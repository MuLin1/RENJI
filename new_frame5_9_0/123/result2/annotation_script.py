import argparse
import concurrent.futures
import json
import math
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# =============================================================================
# Config
# =============================================================================

API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "sk-77a84940c8284ecaa2796c3cfc3399e7"
MODEL_NAME = "deepseek-v3.1"

DATASET_JOBS = [
    {"name": "highrisk", "input_file": "xnk_dataset_highrisk.jsonl", "output_file": "result_highrisk.json"},
    {"name": "chronic", "input_file": "xnk_dataset_chronic.jsonl", "output_file": "result_chronic.json"},
]

REQUEST_TIMEOUT_SEC = 60
DEFAULT_CONCURRENCY = 20
MAX_CONCURRENCY = 20
INTERMEDIATE_SAVE_EVERY = 1
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = [2, 5, 10]
MAX_K_CHECKS = 8
LONG_DIALOGUE_ROUND_THRESHOLD = 8
LOW_CONFIDENCE_SCORE_THRESHOLD = 2.0

script_dir = os.path.dirname(os.path.abspath(__file__))
_thread_local = threading.local()
_save_lock = threading.Lock()

POSITIVE_KEYWORDS = {
    "建议": 2.2,
    "需要": 2.2,
    "复查": 2.0,
    "检查": 1.8,
    "造影": 2.0,
    "住院": 2.0,
    "评估": 1.6,
    "用药": 2.0,
    "药": 0.8,
    "治疗": 2.2,
    "手术": 1.8,
    "诊断": 2.0,
    "考虑": 1.4,
    "方案": 1.6,
    "随访": 1.6,
    "停药": 1.8,
    "调整": 1.6,
}

NEGATIVE_PATTERNS = {
    "你好": -1.0,
    "请问还有": -1.2,
    "是否配药": -1.4,
    "先登记": -1.6,
    "挂号": -1.2,
    "还有问题吗": -1.2,
    "就这": -1.0,
}


def now_iso() -> str:
    return datetime.now().isoformat()


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Rough estimate for mixed Chinese/English prompt.
    return max(1, math.ceil(len(text) / 2.2))


def classify_error(msg: str) -> str:
    m = (msg or "").lower()
    if "timed out" in m or "timeout" in m:
        return "timeout"
    if "rate limit" in m or "429" in m or "too many requests" in m:
        return "rate_limit"
    if "json" in m or "parse" in m:
        return "parse_error"
    if "connection" in m or "network" in m:
        return "network_error"
    return "other_error"


def initialize_model_client() -> Optional[OpenAI]:
    try:
        client = OpenAI(base_url=API_URL, api_key=API_KEY, timeout=REQUEST_TIMEOUT_SEC, max_retries=1)
        print(f"模型客户端初始化成功: {MODEL_NAME} @ {API_URL}")
        return client
    except Exception as e:
        print(f"模型客户端初始化失败: {e}")
        return None


def get_thread_client() -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OpenAI(base_url=API_URL, api_key=API_KEY, timeout=REQUEST_TIMEOUT_SEC, max_retries=1)
        _thread_local.client = client
    return client


def load_input_data(input_file: str) -> Optional[List[Dict[str, Any]]]:
    input_path = os.path.join(script_dir, input_file)
    try:
        data: List[Dict[str, Any]] = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        print(f"成功加载输入数据: {input_file}, 共 {len(data)} 条")
        return data
    except Exception as e:
        print(f"加载输入数据失败: {input_path}, 错误: {e}")
        return None


def normalize_text(text: str) -> str:
    return text.strip() if isinstance(text, str) else ""


def is_patient_turn(text: str) -> bool:
    t = normalize_text(text)
    return t.startswith("患者：") or t.startswith("患者:")


def is_doctor_turn(text: str) -> bool:
    t = normalize_text(text)
    return t.startswith("医生：") or t.startswith("医生:")


def split_dialogue_into_rounds(dialogue: List[str]) -> Tuple[List[Dict[str, str]], List[str]]:
    rounds: List[Dict[str, str]] = []
    all_doctor_turns: List[str] = []
    pending_patient: Optional[str] = None

    for turn in dialogue:
        if not isinstance(turn, str):
            continue
        t = normalize_text(turn)
        if not t:
            continue
        if is_patient_turn(t):
            pending_patient = t
            continue
        if is_doctor_turn(t):
            all_doctor_turns.append(t)
            if pending_patient:
                rounds.append({"patient": pending_patient, "doctor": t})
                pending_patient = None
            continue

    return rounds, all_doctor_turns


def flatten_rounds(rounds: List[Dict[str, str]], k: int) -> List[str]:
    turns: List[str] = []
    for r in rounds[:k]:
        turns.append(r["patient"])
        turns.append(r["doctor"])
    return turns


def compress_turns_for_prompt(rounds: List[Dict[str, str]], k: int) -> Tuple[List[str], Dict[str, Any]]:
    selected_rounds = rounds[:k]
    if len(selected_rounds) <= LONG_DIALOGUE_ROUND_THRESHOLD:
        turns = flatten_rounds(selected_rounds, len(selected_rounds))
        return turns, {"compressed": False, "selected_rounds": len(selected_rounds)}

    keep_idx = set()
    for i in range(min(2, len(selected_rounds))):
        keep_idx.add(i)
    for i in range(max(0, len(selected_rounds) - 4), len(selected_rounds)):
        keep_idx.add(i)

    for i, r in enumerate(selected_rounds):
        text = f"{r['patient']} {r['doctor']}"
        if any(k in text for k in ["症状", "胸闷", "心悸", "建议", "复查", "检查", "治疗", "用药", "住院", "造影"]):
            keep_idx.add(i)

    ordered = sorted(keep_idx)
    turns: List[str] = []
    for idx in ordered:
        turns.append(selected_rounds[idx]["patient"])
        turns.append(selected_rounds[idx]["doctor"])

    return turns, {
        "compressed": True,
        "original_rounds": len(selected_rounds),
        "selected_round_indices_1_based": [i + 1 for i in ordered],
        "selected_rounds": len(ordered),
    }


def score_doctor_text(text: str, round_idx: int, total_rounds: int) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: List[str] = []

    for kw, weight in POSITIVE_KEYWORDS.items():
        if kw in text:
            score += weight
            reasons.append(f"+{weight}:{kw}")

    for pat, weight in NEGATIVE_PATTERNS.items():
        if pat in text:
            score += weight
            reasons.append(f"{weight}:{pat}")

    if "？" in text or "?" in text:
        score -= 0.8
        reasons.append("-0.8:question")

    if "建议" in text and ("复查" in text or "检查" in text or "住院" in text or "用药" in text):
        score += 1.8
        reasons.append("+1.8:action_bundle")

    if total_rounds > 0:
        pos_bonus = (round_idx / total_rounds) * 1.5
        score += pos_bonus
        reasons.append(f"+{pos_bonus:.2f}:position")

    return round(score, 3), reasons


def extract_doctor_diagnosis_by_scoring(
    rounds: List[Dict[str, str]], all_doctor_turns: List[str]
) -> Tuple[str, Optional[int], str, List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    for idx, r in enumerate(rounds, start=1):
        doctor_text = r["doctor"]
        score, reasons = score_doctor_text(doctor_text, idx, len(rounds))
        candidates.append(
            {
                "round": idx,
                "doctor_text": doctor_text,
                "score": score,
                "reasons": reasons,
            }
        )

    if candidates:
        candidates = sorted(candidates, key=lambda x: (x["score"], x["round"]), reverse=True)
        top3 = candidates[:3]
        top = top3[0]
        if top["score"] >= LOW_CONFIDENCE_SCORE_THRESHOLD:
            return top["doctor_text"], top["round"], "scored_candidate", top3
        return rounds[-1]["doctor"], len(rounds), "low_confidence_fallback", top3

    if all_doctor_turns:
        fallback = all_doctor_turns[-1]
        return fallback, None, "fallback_unpaired_doctor", [{"round": None, "doctor_text": fallback, "score": -1.0, "reasons": ["fallback"]}]

    return "", None, "no_doctor_turn", []


def extract_content_from_completion(resp: Any) -> str:
    try:
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return str(resp)


def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json\n", "", 1).strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def call_llm_with_retry(
    client: OpenAI,
    messages: List[Dict[str, str]],
    sample_observe: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    request_log: Dict[str, Any] = {
        "start_at": now_iso(),
        "attempts": [],
        "final_status": "failed",
        "retry_success_at": None,
    }
    joined_prompt = "\n".join([m.get("content", "") for m in messages])
    sample_observe["estimated_input_tokens_total"] += estimate_tokens(joined_prompt)
    sample_observe["total_llm_requests"] += 1

    last_error = ""
    for attempt in range(MAX_RETRIES + 1):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0,
                extra_body={"enable_thinking": False},
            )
            raw_text = extract_content_from_completion(resp)
            elapsed = time.time() - t0
            request_log["attempts"].append({"attempt": attempt + 1, "elapsed_sec": round(elapsed, 3), "status": "success"})
            request_log["final_status"] = "success"
            if attempt > 0:
                request_log["retry_success_at"] = attempt + 1
                sample_observe["retry_success_count"] += 1
            return raw_text, request_log
        except Exception as e:
            msg = str(e)
            err_type = classify_error(msg)
            elapsed = time.time() - t0
            request_log["attempts"].append(
                {"attempt": attempt + 1, "elapsed_sec": round(elapsed, 3), "status": "error", "error_type": err_type, "error_message": msg}
            )
            sample_observe["retry_count"] += 1
            sample_observe["error_type_counter"][err_type] = sample_observe["error_type_counter"].get(err_type, 0) + 1
            if err_type == "timeout":
                sample_observe["timeout_count"] += 1
            last_error = msg
            if attempt < MAX_RETRIES:
                wait_sec = RETRY_BACKOFF_SECONDS[min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)]
                time.sleep(wait_sec)
                continue
            break

    request_log["final_error"] = last_error
    raise RuntimeError(last_error or "LLM request failed")


def judge_diagnosable_with_k(
    client: OpenAI,
    rounds: List[Dict[str, str]],
    k: int,
    sample_observe: Dict[str, Any],
) -> Dict[str, Any]:
    turns, compression_info = compress_turns_for_prompt(rounds, k)
    dialogue_text = "\n".join(turns)
    system_prompt = (
        "你是心内科互联网问诊评估助手。"
        "任务是判断当前对话信息是否足以支持诊疗结论。"
    )
    user_prompt = (
        "仅基于下方对话判断是否可给出诊疗结论（治疗/用药/检查任一项即可）。\n"
        f"对话（前{k}轮）:\n{dialogue_text}\n\n"
        "必须只输出JSON:\n"
        '{"can_diagnose": true/false, "diagnosis_text": "若可诊断则填结论，否则空字符串"}'
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    raw_text, request_log = call_llm_with_retry(client, messages, sample_observe)
    parsed = safe_json_parse(raw_text) or {}
    can = bool(parsed.get("can_diagnose", False))
    diagnosis_text = str(parsed.get("diagnosis_text", "") or "").strip()
    if can and not diagnosis_text:
        can = False

    return {
        "k": k,
        "can_diagnose": can,
        "diagnosis_text": diagnosis_text,
        "raw_response": raw_text,
        "parsed_response": parsed,
        "messages": messages,
        "compression_info": compression_info,
        "request_log": request_log,
    }


def screen_explicit_diagnosis_advice(
    client: OpenAI,
    doctor_diagnosis_raw: str,
    sample_observe: Dict[str, Any],
) -> Dict[str, Any]:
    diagnosis_text = str(doctor_diagnosis_raw or "").strip()
    system_prompt = (
        "You are a strict medical text screener. "
        "Judge whether the doctor's text contains explicit diagnostic advice."
    )
    user_prompt = (
        "Decide whether the doctor text includes explicit diagnostic advice.\n"
        "Explicit diagnostic advice means clear diagnosis direction and/or concrete medical next-step advice "
        "(e.g., suggested examination, treatment, medication adjustment, follow-up plan).\n"
        "Not explicit includes only greetings, pure questioning, registration/process guidance, or generic conversation.\n\n"
        f"Doctor text:\n{diagnosis_text}\n\n"
        "Return JSON only:\n"
        '{"has_explicit_diagnosis_advice": true/false, "advice_screen_reason": "short reason"}'
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    raw_text, request_log = call_llm_with_retry(client, messages, sample_observe)
    parsed = safe_json_parse(raw_text) or {}

    raw_flag = parsed.get("has_explicit_diagnosis_advice", False)
    if isinstance(raw_flag, bool):
        has_explicit = raw_flag
    elif isinstance(raw_flag, str):
        has_explicit = raw_flag.strip().lower() in {"true", "1", "yes", "y"}
    else:
        has_explicit = bool(raw_flag)

    reason = str(parsed.get("advice_screen_reason", "") or "").strip()
    if not reason:
        reason = "explicit_advice_present" if has_explicit else "no_explicit_advice"

    return {
        "has_explicit_diagnosis_advice": has_explicit,
        "advice_screen_reason": reason,
        "raw_response": raw_text,
        "parsed_response": parsed,
        "messages": messages,
        "request_log": request_log,
    }


def build_coarse_k_list(total_rounds: int) -> List[int]:
    ks = list(range(1, total_rounds + 1, 2))
    if total_rounds not in ks:
        ks.append(total_rounds)
    return ks


def find_min_k_with_budget(
    client: OpenAI,
    rounds: List[Dict[str, str]],
    sample_observe: Dict[str, Any],
) -> Tuple[Optional[int], str, List[Dict[str, Any]], str]:
    total_rounds = len(rounds)
    cache: Dict[int, Dict[str, Any]] = {}
    trace: List[Dict[str, Any]] = []
    checks_used = 0
    budget_exhausted = False

    def eval_k(k: int) -> Optional[Dict[str, Any]]:
        nonlocal checks_used, budget_exhausted
        if k in cache:
            return cache[k]
        if checks_used >= MAX_K_CHECKS:
            budget_exhausted = True
            return None
        res = judge_diagnosable_with_k(client, rounds, k, sample_observe)
        cache[k] = res
        checks_used += 1
        trace.append(
            {
                "k": k,
                "can_diagnose": res["can_diagnose"],
                "diagnosis_text": res["diagnosis_text"],
                "request_log": res["request_log"],
                "compression_info": res["compression_info"],
            }
        )
        return res

    coarse_ks = build_coarse_k_list(total_rounds)
    coarse_hit: Optional[int] = None
    coarse_prev_false = 0

    for k in coarse_ks:
        r = eval_k(k)
        if r is None:
            break
        if r["can_diagnose"]:
            coarse_hit = k
            break
        coarse_prev_false = k

    if coarse_hit is None:
        final_k = total_rounds
        if final_k not in cache:
            _ = eval_k(final_k)
        return final_k, "fallback_last_round_no_positive", trace, "budget_exhausted" if budget_exhausted else ""

    search_start = max(1, coarse_prev_false + 1)
    for k in range(search_start, coarse_hit + 1):
        current = eval_k(k)
        if current is None:
            break
        if not current["can_diagnose"]:
            continue

        if k < total_rounds:
            nxt = eval_k(k + 1)
            if nxt is None:
                return k, "consistency_unchecked_due_budget", trace, "budget_exhausted"
            if nxt["can_diagnose"]:
                return k, "model_min_k_consistent", trace, ""
            continue
        return k, "model_min_k_at_last_round", trace, ""

    return coarse_hit, "coarse_hit_fallback", trace, "budget_exhausted" if budget_exhausted else ""


def annotate_single_sample(sample: Dict[str, Any], sample_index: int, total_count: int) -> Dict[str, Any]:
    print(f"处理样本 {sample_index + 1}/{total_count}")
    start_time = time.time()
    pid = sample.get("id", sample.get("pid"))
    dialogue = sample.get("dialogue", [])
    if not isinstance(dialogue, list):
        dialogue = []

    rounds, all_doctor_turns = split_dialogue_into_rounds(dialogue)
    doctor_raw, doctor_round, doctor_source, doctor_candidates = extract_doctor_diagnosis_by_scoring(rounds, all_doctor_turns)
    has_explicit_diagnosis_advice = False
    advice_screen_reason = "screen_not_run"

    sample_observe: Dict[str, Any] = {
        "total_llm_requests": 0,
        "timeout_count": 0,
        "retry_count": 0,
        "retry_success_count": 0,
        "estimated_input_tokens_total": 0,
        "error_type_counter": {},
    }

    if not rounds:
        elapsed = time.time() - start_time
        return {
            "sample_index": sample_index,
            "pid": pid,
            "status": "error",
            "error_message": "No complete patient-doctor rounds found",
            "failure_reason": "invalid_dialogue",
            "k": None,
            "k_source": "none",
            "k_dialogue": [],
            "total_rounds": 0,
            "doctor_diagnosis_raw": doctor_raw,
            "doctor_diagnosis_round": doctor_round,
            "doctor_diagnosis_source": doctor_source,
            "doctor_diagnosis_candidates": doctor_candidates,
            "has_explicit_diagnosis_advice": has_explicit_diagnosis_advice,
            "advice_screen_reason": "invalid_dialogue",
            "processing_time": f"{elapsed:.3f}s",
            "timestamp": now_iso(),
            "observability": sample_observe,
            "debug_info": {"search_trace": [], "advice_screening": {}},
        }

    try:
        client = get_thread_client()
        advice_screen = screen_explicit_diagnosis_advice(client, doctor_raw, sample_observe)
        has_explicit_diagnosis_advice = bool(advice_screen["has_explicit_diagnosis_advice"])
        advice_screen_reason = str(advice_screen["advice_screen_reason"])

        if not has_explicit_diagnosis_advice:
            elapsed = time.time() - start_time
            return {
                "sample_index": sample_index,
                "pid": pid,
                "status": "success",
                "error_message": "",
                "failure_reason": "",
                "k": None,
                "k_source": "screen_no_explicit_advice",
                "k_dialogue": [],
                "total_rounds": len(rounds),
                "doctor_diagnosis_raw": doctor_raw,
                "doctor_diagnosis_round": doctor_round,
                "doctor_diagnosis_source": doctor_source,
                "doctor_diagnosis_candidates": doctor_candidates,
                "has_explicit_diagnosis_advice": has_explicit_diagnosis_advice,
                "advice_screen_reason": advice_screen_reason,
                "processing_time": f"{elapsed:.3f}s",
                "timestamp": now_iso(),
                "observability": sample_observe,
                "debug_info": {
                    "search_trace": [],
                    "search_note": "screen_no_explicit_advice",
                    "advice_screening": {
                        "raw_response": advice_screen.get("raw_response", ""),
                        "parsed_response": advice_screen.get("parsed_response", {}),
                        "messages": advice_screen.get("messages", []),
                        "request_log": advice_screen.get("request_log", {}),
                    },
                },
            }

        selected_k, k_source, search_trace, search_note = find_min_k_with_budget(client, rounds, sample_observe)
        if selected_k is None:
            selected_k = len(rounds)
            k_source = "fallback_last_round_null"

        k_dialogue = flatten_rounds(rounds, selected_k)
        elapsed = time.time() - start_time

        return {
            "sample_index": sample_index,
            "pid": pid,
            "status": "success",
            "error_message": "",
            "failure_reason": "",
            "k": selected_k,
            "k_source": k_source,
            "k_dialogue": k_dialogue,
            "total_rounds": len(rounds),
            "doctor_diagnosis_raw": doctor_raw,
            "doctor_diagnosis_round": doctor_round,
            "doctor_diagnosis_source": doctor_source,
            "doctor_diagnosis_candidates": doctor_candidates,
            "has_explicit_diagnosis_advice": has_explicit_diagnosis_advice,
            "advice_screen_reason": advice_screen_reason,
            "processing_time": f"{elapsed:.3f}s",
            "timestamp": now_iso(),
            "observability": sample_observe,
            "debug_info": {
                "search_trace": search_trace,
                "search_note": search_note,
                "advice_screening": {
                    "raw_response": advice_screen.get("raw_response", ""),
                    "parsed_response": advice_screen.get("parsed_response", {}),
                    "messages": advice_screen.get("messages", []),
                    "request_log": advice_screen.get("request_log", {}),
                },
            },
        }
    except Exception as e:
        elapsed = time.time() - start_time
        msg = str(e)
        return {
            "sample_index": sample_index,
            "pid": pid,
            "status": "error",
            "error_message": msg,
            "failure_reason": classify_error(msg),
            "k": None,
            "k_source": "error",
            "k_dialogue": [],
            "total_rounds": len(rounds),
            "doctor_diagnosis_raw": doctor_raw,
            "doctor_diagnosis_round": doctor_round,
            "doctor_diagnosis_source": doctor_source,
            "doctor_diagnosis_candidates": doctor_candidates,
            "has_explicit_diagnosis_advice": has_explicit_diagnosis_advice,
            "advice_screen_reason": advice_screen_reason if advice_screen_reason != "screen_not_run" else "screening_error",
            "processing_time": f"{elapsed:.3f}s",
            "timestamp": now_iso(),
            "observability": sample_observe,
            "debug_info": {"search_trace": [], "advice_screening": {}},
        }


def build_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]

    timeout_failures = sum(1 for r in failed if r.get("failure_reason") == "timeout")
    low_confidence = sum(1 for r in successful if r.get("doctor_diagnosis_source") == "low_confidence_fallback")
    retries = sum(int(r.get("observability", {}).get("retry_count", 0)) for r in results)
    retry_success = sum(int(r.get("observability", {}).get("retry_success_count", 0)) for r in results)
    requests = sum(int(r.get("observability", {}).get("total_llm_requests", 0)) for r in results)
    timeouts = sum(int(r.get("observability", {}).get("timeout_count", 0)) for r in results)
    screened_out_no_explicit_advice = sum(1 for r in results if r.get("k_source") == "screen_no_explicit_advice")

    avg_retry_per_sample = retries / total if total else 0.0
    timeout_rate = (timeout_failures / total * 100) if total else 0.0
    low_conf_rate = (low_confidence / len(successful) * 100) if successful else 0.0
    screened_out_rate = (screened_out_no_explicit_advice / total * 100) if total else 0.0

    return {
        "total_samples": total,
        "successful_annotations": len(successful),
        "failed_annotations": len(failed),
        "timeout_failures": timeout_failures,
        "timeout_failure_rate": f"{timeout_rate:.2f}%",
        "total_llm_requests": requests,
        "total_retry_count": retries,
        "average_retry_per_sample": round(avg_retry_per_sample, 3),
        "retry_success_count": retry_success,
        "total_timeout_count": timeouts,
        "low_confidence_diagnosis_count": low_confidence,
        "low_confidence_diagnosis_rate": f"{low_conf_rate:.2f}%",
        "screened_out_no_explicit_advice": screened_out_no_explicit_advice,
        "screened_out_no_explicit_advice_rate": f"{screened_out_rate:.2f}%",
    }


def save_intermediate_results(
    output_file: str,
    dataset_name: str,
    input_file: str,
    results: List[Dict[str, Any]],
    processed_count: int,
    concurrency: int,
) -> None:
    summary = build_summary_stats(results)
    partial = {
        "metadata": {
            "dataset_name": dataset_name,
            "input_file": input_file,
            "output_file": output_file,
            "processed_count": processed_count,
            "concurrency": concurrency,
            "model_config": {"url": API_URL, "model_name": MODEL_NAME, "api_key": "***hidden***"},
            "summary": summary,
            "timestamp": now_iso(),
        },
        "results": results,
    }
    partial_path = os.path.join(script_dir, f"{output_file}.partial.json")
    with _save_lock:
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(partial, f, ensure_ascii=False, indent=2)


def run_annotation_for_dataset(
    dataset_job: Dict[str, str],
    limit: Optional[int] = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> Dict[str, Any]:
    dataset_name = dataset_job["name"]
    input_file = dataset_job["input_file"]
    output_file = dataset_job["output_file"]
    print(f"\n{'=' * 70}\n开始标注数据集: {dataset_name}\n输入: {input_file}\n输出: {output_file}\n{'=' * 70}")

    samples = load_input_data(input_file)
    if samples is None:
        return {"dataset_name": dataset_name, "status": "error", "error": "load_input_failed"}

    if limit is not None:
        if limit <= 0:
            raise ValueError(f"--limit must be positive, got: {limit}")
        samples = samples[:limit]
        print(f"limit={limit}, 将处理 {len(samples)} 条样本")

    if concurrency <= 0:
        raise ValueError(f"--concurrency must be positive, got: {concurrency}")
    if concurrency > MAX_CONCURRENCY:
        print(f"concurrency capped: requested={concurrency}, cap={MAX_CONCURRENCY}")
        concurrency = MAX_CONCURRENCY

    total_samples = len(samples)
    if total_samples == 0:
        return {"dataset_name": dataset_name, "status": "error", "error": "no_samples"}

    results_by_index: List[Optional[Dict[str, Any]]] = [None] * total_samples
    completed = 0

    def worker(idx: int, s: Dict[str, Any]) -> Dict[str, Any]:
        return annotate_single_sample(s, idx, total_samples)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(concurrency, total_samples)) as executor:
        future_map = {executor.submit(worker, i, sample): i for i, sample in enumerate(samples)}
        for future in concurrent.futures.as_completed(future_map):
            idx = future_map[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "sample_index": idx,
                    "pid": samples[idx].get("id", samples[idx].get("pid")),
                    "status": "error",
                    "error_message": str(e),
                    "failure_reason": classify_error(str(e)),
                    "k": None,
                    "k_source": "worker_exception",
                    "k_dialogue": [],
                    "total_rounds": 0,
                    "doctor_diagnosis_raw": "",
                    "doctor_diagnosis_round": None,
                    "doctor_diagnosis_source": "worker_exception",
                    "doctor_diagnosis_candidates": [],
                    "has_explicit_diagnosis_advice": False,
                    "advice_screen_reason": "worker_exception",
                    "processing_time": "0.000s",
                    "timestamp": now_iso(),
                    "observability": {
                        "total_llm_requests": 0,
                        "timeout_count": 0,
                        "retry_count": 0,
                        "retry_success_count": 0,
                        "estimated_input_tokens_total": 0,
                        "error_type_counter": {"worker_exception": 1},
                    },
                    "debug_info": {"search_trace": [], "advice_screening": {}},
                }

            results_by_index[idx] = result
            completed += 1

            if completed % INTERMEDIATE_SAVE_EVERY == 0:
                completed_results = [r for r in results_by_index if r is not None]
                save_intermediate_results(
                    output_file=output_file,
                    dataset_name=dataset_name,
                    input_file=input_file,
                    results=completed_results,
                    processed_count=completed,
                    concurrency=concurrency,
                )

    results = [r for r in results_by_index if r is not None]
    summary = build_summary_stats(results)

    total_time = 0.0
    for r in results:
        try:
            total_time += float(str(r.get("processing_time", "0")).rstrip("s"))
        except Exception:
            pass
    avg_time = total_time / total_samples if total_samples else 0.0
    success_rate = (summary["successful_annotations"] / total_samples * 100) if total_samples else 0.0

    final_payload = {
        "metadata": {
            "dataset_name": dataset_name,
            "input_file": input_file,
            "output_file": output_file,
            "total_samples": total_samples,
            "successful_annotations": summary["successful_annotations"],
            "success_rate": f"{success_rate:.2f}%",
            "average_processing_time": f"{avg_time:.3f}s",
            "concurrency": concurrency,
            "max_k_checks": MAX_K_CHECKS,
            "request_timeout_sec": REQUEST_TIMEOUT_SEC,
            "retry_policy": {"max_retries": MAX_RETRIES, "backoff_sec": RETRY_BACKOFF_SECONDS},
            "model_config": {"url": API_URL, "model_name": MODEL_NAME, "api_key": "***hidden***"},
            "summary": summary,
            "timestamp": now_iso(),
        },
        "results": results,
    }

    output_path = os.path.join(script_dir, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)

    print(
        f"数据集 {dataset_name} 标注完成: {summary['successful_annotations']}/{total_samples} "
        f"({success_rate:.2f}%), 输出: {output_path}"
    )
    return {
        "dataset_name": dataset_name,
        "status": "success",
        "output_file": output_file,
        "total_samples": total_samples,
        "successful_annotations": summary["successful_annotations"],
        "success_rate": f"{success_rate:.2f}%",
    }


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate dataset with minimal-k diagnostic turn search.")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N samples per dataset.")
    parser.add_argument("--dataset", choices=["all", "highrisk", "chronic"], default="all", help="Which dataset to run.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Parallel workers, recommended 2-4, capped at {MAX_CONCURRENCY}.",
    )
    return parser.parse_args()


def run_annotation(limit: Optional[int], dataset: str, concurrency: int) -> None:
    warmup_client = initialize_model_client()
    if warmup_client is None:
        raise RuntimeError("Model client initialization failed")

    dataset_jobs = DATASET_JOBS if dataset == "all" else [x for x in DATASET_JOBS if x["name"] == dataset]
    if not dataset_jobs:
        raise ValueError(f"Unknown dataset: {dataset}")

    all_summaries: List[Dict[str, Any]] = []
    for job in dataset_jobs:
        all_summaries.append(run_annotation_for_dataset(job, limit=limit, concurrency=concurrency))

    print(f"\n{'=' * 70}\n全部数据集处理完成\n{'=' * 70}")
    for item in all_summaries:
        if item.get("status") == "success":
            print(
                f"{item['dataset_name']}: 成功 {item['successful_annotations']}/{item['total_samples']} "
                f"({item['success_rate']}) -> {item['output_file']}"
            )
        else:
            print(f"{item['dataset_name']}: 失败 -> {item.get('error')}")


if __name__ == "__main__":
    try:
        args = parse_cli_args()
        run_annotation(limit=args.limit, dataset=args.dataset, concurrency=args.concurrency)
    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"执行异常: {e}")
