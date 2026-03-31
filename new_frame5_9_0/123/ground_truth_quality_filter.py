import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# =============================================================================
# Defaults
# =============================================================================

DEFAULT_INPUT_FILES = [
    "similarity_scores_evaluation_results_Qwen_Qwen3.5-9B_highrisk_deepseek-v3.1.json",
    "similarity_scores_evaluation_results_Qwen_Qwen3.5-9B_chronic_deepseek-v3.1.json",
]

DEFAULT_RAW_JSONL_MAP_ITEMS = [
    "highrisk=xnk_dataset_highrisk.jsonl",
    "chronic=xnk_dataset_chronic.jsonl",
]

DEFAULT_LOW_SCORE_THRESHOLD = 3.5
DEFAULT_JUDGE_MODEL = "deepseek-v3.1"
DEFAULT_JUDGE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_JUDGE_API_KEY_ENV = "sk-77a84940c8284ecaa2796c3cfc3399e7"
DEFAULT_WORKERS = 20
DEFAULT_TIMEOUT_SEC = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = [2, 5, 10]
DEFAULT_POLICY = "strict_rule_plus_llm_v1"

RULE_FILTER_EXACT_TEXTS = {
    "好",
    "好的",
    "不客气",
    "开好了",
    "药开好了",
    "好的 开好了",
    "再试一下",
}

RULE_FILTER_SUBSTRINGS = [
    "给您开好了",
    "给你开好了",
    "您看下",
    "你看下",
    "一次只能",
    "20片/盒",
    "能开2盒",
    "能开几盒",
    "是否快递",
    "快递",
    "没货",
    "付款",
    "后台回复",
    "冷链运输",
    "结束了哈",
]

RULE_FILTER_QUESTION_SUBSTRINGS = [
    "什么问题",
    "最近还有什么症状",
    "最近情况还好吗",
    "有没有复查",
    "有复查过吗",
    "愿意住院复查吗",
    "需要配药吗",
    "现在吃几粒",
    "各要多少",
]

RULE_KEEP_ACTION_KEYWORDS = [
    "建议",
    "复查",
    "检查",
    "造影",
    "住院",
    "复诊",
    "监测",
    "继续",
    "停用",
    "停了",
    "换用",
    "换一个",
    "换别的",
    "减量",
    "加量",
    "维持原剂量",
    "按原剂量",
    "不用加量",
]

RULE_KEEP_RESULT_KEYWORDS = [
    "结果正常",
    "无明显异常",
    "没有明显的异常",
    "没什么大问题",
]

script_dir = os.path.dirname(os.path.abspath(__file__))
_thread_local = threading.local()
_print_lock = threading.Lock()


def safe_print(*args: Any, **kwargs: Any) -> None:
    with _print_lock:
        print(*args, **kwargs)


def now_iso() -> str:
    return datetime.now().isoformat()


def normalize_base_url(url: str) -> str:
    u = (url or "").rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/responses"):
        if u.endswith(suffix):
            u = u[: -len(suffix)]
            break
    return u


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


def extract_response_text(resp: Any) -> str:
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


def get_thread_client(
    judge_base_url: str,
    judge_api_key: str,
    timeout_sec: int,
) -> OpenAI:
    cfg_key = (judge_base_url, judge_api_key, timeout_sec)
    cache_key = getattr(_thread_local, "client_cfg_key", None)
    client = getattr(_thread_local, "client", None)
    if client is None or cache_key != cfg_key:
        client = OpenAI(
            base_url=normalize_base_url(judge_base_url),
            api_key=judge_api_key,
            timeout=timeout_sec,
            max_retries=1,
        )
        _thread_local.client = client
        _thread_local.client_cfg_key = cfg_key
    return client


def strip_role_prefix(text: str) -> str:
    t = (text or "").strip()
    for p in ("医生：", "医生:", "doctor:", "Doctor:"):
        if t.startswith(p):
            return t[len(p):].strip()
    return t


def normalize_rule_text(text: str) -> str:
    return re.sub(r"\s+", " ", strip_role_prefix(text)).strip()


def contains_any(text: str, keywords: List[str]) -> bool:
    return any(k in text for k in keywords)


def classify_ground_truth_by_rules(ground_truth_text: str) -> Optional[Dict[str, Any]]:
    core = normalize_rule_text(ground_truth_text)
    if not core:
        return {
            "decision": "filter",
            "reason": "真值为空，不能形成有效临床指引",
            "evidence_span": "",
            "confidence": 1.0,
        }

    if contains_any(core, RULE_KEEP_ACTION_KEYWORDS):
        return {
            "decision": "keep",
            "reason": "包含明确医疗动作或管理建议，应保留作为有效真值",
            "evidence_span": core[:40],
            "confidence": 0.98,
        }

    if contains_any(core, RULE_KEEP_RESULT_KEYWORDS) and contains_any(
        core, ["复诊", "监测", "检查", "用药", "观察", "门诊", "就诊"]
    ):
        return {
            "decision": "keep",
            "reason": "包含检查结果解读和下一步临床动作，应保留",
            "evidence_span": core[:40],
            "confidence": 0.98,
        }

    if core in RULE_FILTER_EXACT_TEXTS:
        return {
            "decision": "filter",
            "reason": "纯礼貌、纯确认或纯流程短句，信息量过低",
            "evidence_span": core[:40],
            "confidence": 0.99,
        }

    if contains_any(core, RULE_FILTER_SUBSTRINGS):
        return {
            "decision": "filter",
            "reason": "主要是配药、订单、库存、付款或流程确认信息，缺少有效临床动作",
            "evidence_span": core[:40],
            "confidence": 0.98,
        }

    if ("?" in core or "？" in core) and contains_any(core, RULE_FILTER_QUESTION_SUBSTRINGS):
        return {
            "decision": "filter",
            "reason": "主要在收集信息，缺少明确临床建议或管理动作",
            "evidence_span": core[:40],
            "confidence": 0.96,
        }

    if len(core) <= 18 and contains_any(core, ["几粒", "多少", "要吗", "对吧"]):
        return {
            "decision": "filter",
            "reason": "短流程追问或数量确认，缺少临床指导价值",
            "evidence_span": core[:40],
            "confidence": 0.95,
        }

    return None


def build_quality_judge_messages(ground_truth_text: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a medical data quality reviewer. "
        "Judge whether the Ground Truth text is low-quality. "
        "Use only the Ground Truth text itself and ignore prediction. "
        "This policy is strict: pure workflow, prescription logistics, order confirmation, inventory/payment, and most short question-only replies should be filtered. "
        "Return JSON only."
    )

    user_prompt = (
        "Decide whether the following Ground Truth should be filtered.\n\n"
        "Filter as low_quality when it is mainly: workflow/polite closing, prescription logistics, order confirmation, inventory/payment/shipping, or short question-only information collection without a clear medical action.\n\n"
        "Keep as acceptable when it contains a clear clinical action or management step, such as examination, recheck, angiography, admission, follow-up, monitoring, continue/stop/switch medication, explicit dose instruction, or result interpretation plus a next step.\n\n"
        "Return JSON fields only:\n"
        "- should_filter: boolean\n"
        "- quality_label: low_quality or acceptable\n"
        "- reason: short explanation\n"
        "- evidence_span: short quoted span from Ground Truth\n"
        "- confidence: 0~1 number\n\n"
        f"[Ground Truth]\n{ground_truth_text}\n"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_bool_value(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "yes", "1"}:
            return True
        if s in {"false", "no", "0"}:
            return False
    return None


def normalize_quality_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("invalid_payload_type")

    should_filter = parse_bool_value(payload.get("should_filter"))
    if should_filter is None:
        raise ValueError("invalid_or_missing_field:should_filter")

    quality_label = str(payload.get("quality_label", "")).strip()
    if not quality_label:
        quality_label = "low_quality" if should_filter else "acceptable"

    reason = str(payload.get("reason", "")).strip()
    evidence_span = str(payload.get("evidence_span", "")).strip()
    confidence_raw = payload.get("confidence", 0.5)
    try:
        confidence = float(confidence_raw)
    except Exception as e:
        raise ValueError("invalid_field_type:confidence") from e
    if confidence > 1:
        confidence = confidence / 100.0
    confidence = max(0.0, min(1.0, confidence))

    if len(reason) > 240:
        reason = reason[:240]
    if len(evidence_span) > 120:
        evidence_span = evidence_span[:120]

    return {
        "should_filter": should_filter,
        "quality_label": quality_label,
        "reason": reason,
        "evidence_span": evidence_span,
        "confidence": round(confidence, 4),
    }


def call_quality_judge_llm(
    *,
    messages: List[Dict[str, str]],
    judge_model: str,
    judge_base_url: str,
    judge_api_key: str,
    timeout_sec: int,
    max_retries: int,
    retry_backoff_seconds: List[int],
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    if not judge_api_key:
        raise RuntimeError("missing_api_key")

    client = get_thread_client(
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
        timeout_sec=timeout_sec,
    )

    attempts: List[Dict[str, Any]] = []
    last_error = ""
    raw_text = ""

    for i in range(max_retries + 1):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=messages,
                temperature=0,
                extra_body={"enable_thinking": False},
            )
            raw_text = extract_response_text(resp)
            parsed = safe_json_parse(raw_text)
            elapsed = round(time.time() - t0, 3)
            attempts.append({"attempt": i + 1, "status": "success", "elapsed_sec": elapsed})
            if parsed is None:
                raise ValueError("LLM JSON parse failed")
            return parsed, raw_text, {"attempts": attempts, "retry_success_at": (i + 1 if i > 0 else None)}
        except Exception as e:
            msg = str(e)
            last_error = msg
            elapsed = round(time.time() - t0, 3)
            attempts.append(
                {
                    "attempt": i + 1,
                    "status": "error",
                    "elapsed_sec": elapsed,
                    "error_type": classify_error(msg),
                    "error_message": msg,
                }
            )
            if i < max_retries:
                time.sleep(retry_backoff_seconds[min(i, len(retry_backoff_seconds) - 1)])
            else:
                break

    raise RuntimeError(
        json.dumps(
            {"error_message": last_error or "LLM call failed", "raw_text": raw_text, "attempts": attempts},
            ensure_ascii=False,
        )
    )


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_distribution(scores: List[float]) -> Dict[str, int]:
    bins = {f"{x/2:.1f}": 0 for x in range(0, 11)}
    for s in scores:
        key = f"{float(s):.1f}"
        if key not in bins:
            bins[key] = 0
        bins[key] += 1
    return bins


def recompute_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    success_items = [x for x in results if x.get("status") == "success" and x.get("scoring")]
    error_items = [x for x in results if x.get("status") == "error"]
    skipped_items = [x for x in results if x.get("status") == "skipped"]
    final_scores = [float(x["scoring"]["final_score_0_5"]) for x in success_items]
    avg_score = round(sum(final_scores) / len(final_scores), 4) if final_scores else 0.0

    low_score_samples = [
        {
            "sample_index": x.get("sample_index"),
            "pid": x.get("pid"),
            "final_score_0_5": x["scoring"]["final_score_0_5"],
            "is_low_info_ground_truth": x.get("reference_quality", {}).get("is_low_info_ground_truth", False),
            "low_info_reasons": x.get("reference_quality", {}).get("low_info_reasons", []),
            "rationale": x["scoring"].get("rationale", ""),
        }
        for x in success_items
        if float(x["scoring"]["final_score_0_5"]) <= 2.0
    ]

    return {
        "total_records": len(results),
        "success_count": len(success_items),
        "error_count": len(error_items),
        "skipped_count": len(skipped_items),
        "average_final_score_0_5": avg_score,
        "score_distribution": build_distribution(final_scores),
        "low_score_samples_le_2": low_score_samples,
    }


def get_record_key(record: Dict[str, Any]) -> str:
    pid = record.get("pid")
    if pid:
        return f"pid:{pid}"
    return f"sample_index:{record.get('sample_index')}"


def get_score_0_5(record: Dict[str, Any]) -> Optional[float]:
    try:
        scoring = record.get("scoring")
        if not isinstance(scoring, dict):
            return None
        return float(scoring.get("final_score_0_5"))
    except Exception:
        return None


def evaluate_low_score_record(
    *,
    dataset_name: str,
    record: Dict[str, Any],
    judge_model: str,
    judge_base_url: str,
    judge_api_key: str,
    timeout_sec: int,
    max_retries: int,
    retry_backoff_seconds: List[int],
) -> Dict[str, Any]:
    sample_index = record.get("sample_index")
    pid = record.get("pid")
    ground_truth_text = str(record.get("ground_truth_text", "") or "").strip()
    score = get_score_0_5(record)
    key = get_record_key(record)

    base = {
        "dataset_name": dataset_name,
        "sample_index": sample_index,
        "pid": pid,
        "record_key": key,
        "final_score_0_5": score,
        "ground_truth_text": ground_truth_text,
    }

    if not ground_truth_text:
        return {
            **base,
            "judge_status": "rule_based_filter",
            "error_message": "",
            "judge_result": {
                "should_filter": True,
                "quality_label": "low_quality",
                "reason": "??????????????",
                "evidence_span": "",
                "confidence": 1.0,
            },
            "debug_info": {},
        }

    rule_result = classify_ground_truth_by_rules(ground_truth_text)
    if rule_result is not None:
        should_filter = bool(rule_result["decision"] == "filter")
        return {
            **base,
            "judge_status": ("rule_based_filter" if should_filter else "rule_based_keep"),
            "error_message": "",
            "judge_result": {
                "should_filter": should_filter,
                "quality_label": ("low_quality" if should_filter else "acceptable"),
                "reason": str(rule_result["reason"]),
                "evidence_span": str(rule_result["evidence_span"]),
                "confidence": float(rule_result["confidence"]),
            },
            "debug_info": {"rule_result": rule_result},
        }

    messages = build_quality_judge_messages(ground_truth_text)
    t0 = time.time()
    try:
        parsed, raw_text, llm_debug = call_quality_judge_llm(
            messages=messages,
            judge_model=judge_model,
            judge_base_url=judge_base_url,
            judge_api_key=judge_api_key,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        normalized = normalize_quality_payload(parsed)
        elapsed = round(time.time() - t0, 3)
        return {
            **base,
            "judge_status": "success",
            "error_message": "",
            "judge_result": normalized,
            "processing_time_sec": elapsed,
            "debug_info": {
                "messages": messages,
                "raw_llm_response": raw_text,
                "parsed_llm_response": parsed,
                "llm_call": llm_debug,
            },
        }
    except Exception as e:
        elapsed = round(time.time() - t0, 3)
        return {
            **base,
            "judge_status": "judge_error",
            "error_message": str(e),
            "judge_result": {
                "should_filter": False,
                "quality_label": "judge_error",
                "reason": "LLM judge failed, keep the sample conservatively",
                "evidence_span": "",
                "confidence": 0.0,
            },
            "processing_time_sec": elapsed,
            "debug_info": {"messages": messages},
        }


def parse_raw_jsonl_map(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid --raw-jsonl-map item: {item}, expected dataset=path")
        dataset, raw_path = item.split("=", 1)
        dataset = dataset.strip()
        raw_path = raw_path.strip()
        if not dataset or not raw_path:
            raise ValueError(f"invalid --raw-jsonl-map item: {item}")
        out[dataset] = raw_path
    return out


def resolve_judge_api_key(
    judge_api_key_env_or_value: str,
    judge_api_key: Optional[str],
) -> Tuple[str, str]:
    direct_key = (judge_api_key or "").strip()
    if direct_key:
        return direct_key, "arg:--judge-api-key"

    source = (judge_api_key_env_or_value or "").strip()
    if not source:
        return "", "missing"

    env_value = os.getenv(source, "")
    if env_value:
        return env_value, f"env:{source}"

    if source.startswith("sk-"):
        return source, "inline_key_from_--judge-api-key-env"

    return "", f"env:{source}"


def resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(script_dir, path)


def extract_pid(sample: Dict[str, Any], fallback_index: Optional[int] = None) -> Optional[str]:
    pid = sample.get("id", sample.get("pid"))
    if pid is not None and str(pid).strip():
        return str(pid)
    if fallback_index is not None:
        return f"idx_{fallback_index}"
    return None


def infer_dataset_name_from_path(input_path: str) -> Optional[str]:
    """
    Best-effort dataset name inference from filename/path.

    Some upstream scoring artifacts may stamp the same `metadata.dataset_name`
    across different datasets, which can make downstream stats look identical.
    When an unambiguous token exists in the filename, prefer it.
    """

    base = os.path.basename(input_path or "")
    s = base.lower()

    hits: List[Tuple[int, str]] = []
    for token in ("highrisk", "chronic"):
        m = re.search(rf"(^|[^a-z0-9]){re.escape(token)}([^a-z0-9]|$)", s)
        if m:
            hits.append((m.start(), token))

    if hits:
        hits.sort(key=lambda x: x[0])
        return hits[0][1]

    return None


def short_path_hash(input_path: str, n: int = 8) -> str:
    h = hashlib.sha1((input_path or "").encode("utf-8", errors="ignore")).hexdigest()
    return h[: max(4, int(n))]


def choose_decisions_out_file(
    *,
    output_dir: str,
    dataset_name: str,
    input_path: str,
) -> str:
    """
    Prefer the stable name `ground_truth_quality_decisions_{dataset}.json`.

    If that name is already used by a different input file (common when upstream
    metadata.dataset_name is duplicated), switch to a disambiguated filename to
    avoid silent overwrite that makes stats look identical across datasets.
    """

    primary = f"ground_truth_quality_decisions_{dataset_name}.json"
    primary_path = os.path.join(output_dir, primary)
    if not os.path.exists(primary_path):
        return primary

    try:
        existing = load_json(primary_path)
        existing_input = str(existing.get("metadata", {}).get("input_file", "") or "")
        if existing_input == os.path.basename(input_path):
            return primary
    except Exception:
        # If we cannot parse, be conservative and disambiguate.
        pass

    return f"ground_truth_quality_decisions_{dataset_name}__{short_path_hash(input_path)}.json"


def process_similarity_file(
    *,
    input_path: str,
    output_dir: str,
    low_score_threshold: float,
    judge_model: str,
    judge_base_url: str,
    judge_api_key: str,
    judge_api_key_source: str,
    timeout_sec: int,
    max_retries: int,
    retry_backoff_seconds: List[int],
    workers: int,
    max_samples: Optional[int],
) -> Dict[str, Any]:
    payload = load_json(input_path)

    metadata_dataset_name = payload.get("metadata", {}).get("dataset_name")
    metadata_dataset_name = str(metadata_dataset_name).strip() if metadata_dataset_name is not None else ""
    inferred_dataset_name = infer_dataset_name_from_path(input_path) or ""

    dataset_name = inferred_dataset_name or metadata_dataset_name or "unknown_dataset"
    if inferred_dataset_name and metadata_dataset_name and inferred_dataset_name != metadata_dataset_name:
        safe_print(
            f"[WARN] dataset_name mismatch for {os.path.basename(input_path)}: "
            f"metadata={metadata_dataset_name}, inferred={inferred_dataset_name}. Using inferred."
        )
    results = payload.get("results", [])
    if not isinstance(results, list):
        results = []

    original_summary = recompute_summary(results)
    candidates: List[Dict[str, Any]] = []
    for record in results:
        if record.get("status") != "success":
            continue
        score = get_score_0_5(record)
        if score is None:
            continue
        if score <= low_score_threshold:
            candidates.append(record)

    candidate_total = len(candidates)
    if max_samples is not None and max_samples >= 0:
        candidates_to_eval = candidates[:max_samples]
    else:
        candidates_to_eval = candidates
    candidate_skipped_by_max_samples = max(0, candidate_total - len(candidates_to_eval))

    safe_print(
        f"[{dataset_name}] total={len(results)}, low_score_candidates={candidate_total}, "
        f"to_evaluate={len(candidates_to_eval)}"
    )

    decisions: List[Optional[Dict[str, Any]]] = [None] * len(candidates_to_eval)
    progress = [0]
    lock = threading.Lock()

    def _worker(i: int, rec: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        decision = evaluate_low_score_record(
            dataset_name=dataset_name,
            record=rec,
            judge_model=judge_model,
            judge_base_url=judge_base_url,
            judge_api_key=judge_api_key,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        with lock:
            progress[0] += 1
            safe_print(f"[{dataset_name}] judging progress: {progress[0]}/{len(candidates_to_eval)}")
        return i, decision

    if candidates_to_eval:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(_worker, i, rec) for i, rec in enumerate(candidates_to_eval)]
            for fut in concurrent.futures.as_completed(futures):
                i, item = fut.result()
                decisions[i] = item

    decision_items = [x for x in decisions if x is not None]
    decision_key_map = {d["record_key"]: d for d in decision_items}
    filtered_keys = {
        d["record_key"]
        for d in decision_items
        if d.get("judge_status") in {"success", "rule_based_filter"} and d.get("judge_result", {}).get("should_filter")
    }

    filtered_results: List[Dict[str, Any]] = []
    filtered_pids: List[str] = []
    for record in results:
        key = get_record_key(record)
        if key in filtered_keys:
            pid = record.get("pid")
            if pid:
                filtered_pids.append(pid)
            continue
        filtered_results.append(record)

    filtered_summary = recompute_summary(filtered_results)

    judge_status_counts: Dict[str, int] = {}
    for d in decision_items:
        status = str(d.get("judge_status", "unknown"))
        judge_status_counts[status] = judge_status_counts.get(status, 0) + 1

    filter_stats = {
        "input_total_records": len(results),
        "candidate_count_total": candidate_total,
        "candidate_evaluated_count": len(candidates_to_eval),
        "candidate_skipped_by_max_samples": candidate_skipped_by_max_samples,
        "filtered_count": len(filtered_keys),
        "kept_count": len(results) - len(filtered_keys),
        "filter_rate_in_candidates": round(len(filtered_keys) / candidate_total, 4) if candidate_total else 0.0,
        "filter_rate_overall": round(len(filtered_keys) / len(results), 4) if results else 0.0,
        "judge_status_counts": judge_status_counts,
        "rule_based_filter_count": judge_status_counts.get("rule_based_filter", 0),
        "rule_based_keep_count": judge_status_counts.get("rule_based_keep", 0),
        "llm_judged_count": judge_status_counts.get("success", 0),
        "average_before_success": original_summary["average_final_score_0_5"],
        "average_after_success": filtered_summary["average_final_score_0_5"],
        "average_delta_success": round(
            filtered_summary["average_final_score_0_5"] - original_summary["average_final_score_0_5"], 4
        ),
    }

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    similarity_out_file = f"{base_name}_filtered.json"
    decisions_out_file = choose_decisions_out_file(
        output_dir=output_dir,
        dataset_name=dataset_name,
        input_path=input_path,
    )

    similarity_out_path = os.path.join(output_dir, similarity_out_file)
    decisions_out_path = os.path.join(output_dir, decisions_out_file)

    out_payload = copy.deepcopy(payload)
    out_metadata = dict(out_payload.get("metadata", {}))
    out_metadata["summary"] = filtered_summary
    out_metadata["ground_truth_quality_filter"] = {
        "enabled": True,
        "policy": DEFAULT_POLICY,
        "low_score_threshold": low_score_threshold,
        "dataset_name": dataset_name,
        "dataset_name_sources": {
            "metadata_dataset_name": (metadata_dataset_name or None),
            "inferred_dataset_name": (inferred_dataset_name or None),
            "input_file": os.path.basename(input_path),
        },
        "judge_model": {
            "name": judge_model,
            "url": normalize_base_url(judge_base_url),
            "api_key_source": judge_api_key_source,
            "api_key": "***hidden***",
        },
        "filter_stats": filter_stats,
        "decisions_file": decisions_out_file,
        "timestamp": now_iso(),
    }
    out_payload["metadata"] = out_metadata
    out_payload["results"] = filtered_results
    write_json(similarity_out_path, out_payload)

    decisions_payload = {
        "metadata": {
            "dataset_name": dataset_name,
            "input_file": os.path.basename(input_path),
            "policy": DEFAULT_POLICY,
            "low_score_threshold": low_score_threshold,
            "dataset_name_sources": {
                "metadata_dataset_name": (metadata_dataset_name or None),
                "inferred_dataset_name": (inferred_dataset_name or None),
                "input_file": os.path.basename(input_path),
            },
            "judge_model": {
                "name": judge_model,
                "url": normalize_base_url(judge_base_url),
                "api_key_source": judge_api_key_source,
                "api_key": "***hidden***",
            },
            "timestamp": now_iso(),
        },
        "summary": filter_stats,
        "decisions": decision_items,
    }
    write_json(decisions_out_path, decisions_payload)

    safe_print(f"[{dataset_name}] wrote similarity: {similarity_out_path}")
    safe_print(f"[{dataset_name}] wrote decisions: {decisions_out_path}")

    return {
        "dataset_name": dataset_name,
        "input_file": os.path.basename(input_path),
        "input_path": input_path,
        "output_similarity_file": similarity_out_file,
        "output_similarity_path": similarity_out_path,
        "output_decisions_file": decisions_out_file,
        "output_decisions_path": decisions_out_path,
        "original_summary": original_summary,
        "filtered_summary": filtered_summary,
        "filter_stats": filter_stats,
        "filtered_pids": sorted(set(filtered_pids)),
        "evaluated_record_key_count": len(decision_key_map),
    }


def filter_raw_jsonl_by_pids(
    *,
    dataset_name: str,
    raw_jsonl_path: str,
    output_dir: str,
    removed_pids: List[str],
) -> Dict[str, Any]:
    removed_set = {str(x) for x in removed_pids if x is not None and str(x).strip()}
    if not os.path.exists(raw_jsonl_path):
        return {
            "dataset_name": dataset_name,
            "raw_input_path": raw_jsonl_path,
            "raw_output_path": "",
            "status": "missing_input_jsonl",
        }

    base_name = os.path.splitext(os.path.basename(raw_jsonl_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_filtered.jsonl")

    total_lines = 0
    kept_lines = 0
    removed_lines = 0
    removed_found_pids: set = set()
    decode_errors = 0
    sample_index = 0

    with open(raw_jsonl_path, "r", encoding="utf-8") as rf, open(out_path, "w", encoding="utf-8") as wf:
        for line in rf:
            line_strip = line.strip()
            if not line_strip:
                continue
            total_lines += 1
            try:
                obj = json.loads(line_strip)
                pid = extract_pid(obj, sample_index)
                sample_index += 1
            except Exception:
                decode_errors += 1
                wf.write(line)
                kept_lines += 1
                continue

            if pid in removed_set:
                removed_lines += 1
                removed_found_pids.add(pid)
                continue
            wf.write(line)
            kept_lines += 1

    missing_removed_pids = sorted(removed_set - removed_found_pids)
    return {
        "dataset_name": dataset_name,
        "raw_input_path": raw_jsonl_path,
        "raw_output_path": out_path,
        "status": "ok",
        "raw_total_lines": total_lines,
        "raw_kept_lines": kept_lines,
        "raw_removed_lines": removed_lines,
        "removed_pid_count_expected": len(removed_set),
        "removed_pid_count_found_in_raw": len(removed_found_pids),
        "missing_removed_pid_count_in_raw": len(missing_removed_pids),
        "missing_removed_pids_preview": missing_removed_pids[:20],
        "json_decode_error_lines": decode_errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter low-score samples by LLM ground-truth quality judgement.")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=DEFAULT_INPUT_FILES,
        help="Similarity score JSON files to process.",
    )
    parser.add_argument(
        "--raw-jsonl-map",
        nargs="*",
        default=DEFAULT_RAW_JSONL_MAP_ITEMS,
        help="Dataset to raw jsonl mapping, format: dataset=path",
    )
    parser.add_argument("--low-score-threshold", type=float, default=DEFAULT_LOW_SCORE_THRESHOLD)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-base-url", default=DEFAULT_JUDGE_BASE_URL)
    parser.add_argument("--judge-api-key-env", default=DEFAULT_JUDGE_API_KEY_ENV)
    parser.add_argument(
        "--judge-api-key",
        default="",
        help="Direct API key. If provided, it overrides --judge-api-key-env.",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-samples", type=int, default=None, help="Max low-score samples to evaluate per file.")
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--output-dir",
        default=script_dir,
        help="Directory for filtered outputs and summary files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    raw_jsonl_map = parse_raw_jsonl_map(args.raw_jsonl_map)
    judge_api_key, judge_api_key_source = resolve_judge_api_key(args.judge_api_key_env, args.judge_api_key)

    safe_print("=" * 80)
    safe_print("Ground-truth quality filter started")
    safe_print(f"Judge model: {args.judge_model} @ {normalize_base_url(args.judge_base_url)}")
    safe_print(f"Low-score threshold: <= {args.low_score_threshold}")
    safe_print(f"Workers: {args.workers}")
    safe_print(f"Output dir: {output_dir}")
    safe_print(f"API key source: {judge_api_key_source}")
    safe_print("=" * 80)

    if not judge_api_key:
        safe_print(
            "WARNING: judge api key is empty. Judging records will become judge_error=missing_api_key. "
            "Please pass --judge-api-key or set --judge-api-key-env to a valid env var (or inline sk-* key)."
        )

    file_summaries: List[Dict[str, Any]] = []
    raw_filter_summaries: List[Dict[str, Any]] = []

    for input_item in args.inputs:
        input_path = resolve_path(input_item)
        if not os.path.exists(input_path):
            safe_print(f"Skip missing file: {input_path}")
            continue

        summary = process_similarity_file(
            input_path=input_path,
            output_dir=output_dir,
            low_score_threshold=args.low_score_threshold,
            judge_model=args.judge_model,
            judge_base_url=args.judge_base_url,
            judge_api_key=judge_api_key,
            judge_api_key_source=judge_api_key_source,
            timeout_sec=args.timeout_sec,
            max_retries=args.max_retries,
            retry_backoff_seconds=DEFAULT_RETRY_BACKOFF_SECONDS,
            workers=args.workers,
            max_samples=args.max_samples,
        )
        file_summaries.append(summary)

        dataset_name = summary["dataset_name"]
        removed_pids = summary.get("filtered_pids", [])
        raw_input_path = raw_jsonl_map.get(dataset_name)
        if raw_input_path:
            raw_input_path = resolve_path(raw_input_path)
            raw_summary = filter_raw_jsonl_by_pids(
                dataset_name=dataset_name,
                raw_jsonl_path=raw_input_path,
                output_dir=output_dir,
                removed_pids=removed_pids,
            )
            raw_filter_summaries.append(raw_summary)
            safe_print(
                f"[{dataset_name}] raw jsonl filtered: {raw_summary.get('raw_removed_lines', 0)} removed, "
                f"{raw_summary.get('raw_kept_lines', 0)} kept"
            )
        else:
            raw_filter_summaries.append(
                {
                    "dataset_name": dataset_name,
                    "status": "raw_jsonl_map_missing",
                    "message": "Dataset not found in --raw-jsonl-map",
                }
            )

    weighted_before = 0.0
    weighted_after = 0.0
    weighted_count_before = 0
    weighted_count_after = 0
    candidate_total = 0
    candidate_evaluated_total = 0
    filtered_total = 0
    overall_judge_status_counts: Dict[str, int] = {}

    serializable_file_summaries: List[Dict[str, Any]] = []
    for item in file_summaries:
        before_n = item["original_summary"]["success_count"]
        after_n = item["filtered_summary"]["success_count"]
        weighted_before += item["original_summary"]["average_final_score_0_5"] * before_n
        weighted_after += item["filtered_summary"]["average_final_score_0_5"] * after_n
        weighted_count_before += before_n
        weighted_count_after += after_n

        fs = item["filter_stats"]
        candidate_total += fs["candidate_count_total"]
        candidate_evaluated_total += fs["candidate_evaluated_count"]
        filtered_total += fs["filtered_count"]
        for key, value in fs["judge_status_counts"].items():
            overall_judge_status_counts[key] = overall_judge_status_counts.get(key, 0) + int(value)

        item_for_json = dict(item)
        item_for_json.pop("filtered_pids", None)
        serializable_file_summaries.append(item_for_json)

    global_avg_before = round(weighted_before / weighted_count_before, 4) if weighted_count_before else 0.0
    global_avg_after = round(weighted_after / weighted_count_after, 4) if weighted_count_after else 0.0

    summary_payload = {
        "metadata": {
            "policy": DEFAULT_POLICY,
            "judge_model": {
                "name": args.judge_model,
                "url": normalize_base_url(args.judge_base_url),
                "api_key_source": judge_api_key_source,
                "api_key": "***hidden***",
            },
            "low_score_threshold": args.low_score_threshold,
            "workers": args.workers,
            "max_samples": args.max_samples,
            "timestamp": now_iso(),
        },
        "global_summary": {
            "files_processed": len(file_summaries),
            "candidate_count_total": candidate_total,
            "candidate_evaluated_total": candidate_evaluated_total,
            "filtered_count_total": filtered_total,
            "filter_rate_in_candidates": round(filtered_total / candidate_total, 4) if candidate_total else 0.0,
            "global_average_final_score_0_5_before": global_avg_before,
            "global_average_final_score_0_5_after": global_avg_after,
            "global_average_delta_success": round(global_avg_after - global_avg_before, 4),
            "judge_status_counts": overall_judge_status_counts,
        },
        "file_summaries": serializable_file_summaries,
        "raw_jsonl_filter_summaries": raw_filter_summaries,
    }
    summary_path = os.path.join(output_dir, "ground_truth_filter_summary.json")
    write_json(summary_path, summary_payload)

    safe_print("\nDone")
    safe_print(f"Global summary: {summary_path}")


if __name__ == "__main__":
    main()
