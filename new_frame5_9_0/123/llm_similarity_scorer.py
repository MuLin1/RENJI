import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# =============================================================================
# Fixed Scoring Model Config
# =============================================================================

SCORING_MODEL_NAME = "deepseek-v3.1"
SCORING_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
SCORING_API_KEY = os.getenv(
    "SCORING_API_KEY",
    "sk-77a84940c8284ecaa2796c3cfc3399e7",
)

REQUEST_TIMEOUT_SEC = 60
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = [2, 5, 10]
DEFAULT_PARALLEL_WORKERS = 50
LLM_TEMPERATURE = 0

DEFAULT_INPUT_FILES = [
    "evaluation_results_Qwen_Qwen3.5-9B_highrisk.json",
    "evaluation_results_Qwen_Qwen3.5-9B_chronic.json",
]

LOW_INFO_WORKFLOW_KEYWORDS = [
    "开好了",
    "已开好",
    "按原剂量",
    "线下",
    "复诊",
    "登记",
    "挂号",
    "不客气",
    "请问",
    "您好",
    "还有问题吗",
]
LOW_INFO_MEDICAL_ACTION_KEYWORDS = [
    "建议",
    "检查",
    "复查",
    "治疗",
    "用药",
    "停药",
    "加药",
    "住院",
    "造影",
    "手术",
    "评估",
    "诊断",
]

script_dir = os.path.dirname(os.path.abspath(__file__))
_thread_local = threading.local()
_print_lock = threading.Lock()


def safe_print(*args: Any, **kwargs: Any) -> None:
    with _print_lock:
        try:
            print(*args, **kwargs)
        except UnicodeEncodeError:
            # Fallback for terminals with non-UTF8 encodings (e.g., gbk).
            text = " ".join(str(a) for a in args)
            encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            safe_text = text.encode(encoding, errors="backslashreplace").decode(encoding, errors="ignore")
            print(safe_text, **kwargs)


def now_iso() -> str:
    return datetime.now().isoformat()


def normalize_base_url(url: str) -> str:
    u = (url or "").rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/responses"):
        if u.endswith(suffix):
            u = u[: -len(suffix)]
            break
    return u


def get_thread_client() -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OpenAI(
            base_url=normalize_base_url(SCORING_API_URL),
            api_key=SCORING_API_KEY,
            timeout=REQUEST_TIMEOUT_SEC,
            max_retries=1,
        )
        _thread_local.client = client
    return client


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


def strip_role_prefix(text: str) -> str:
    t = (text or "").strip()
    for p in ("医生：", "医生:", "doctor:", "Doctor:"):
        if t.startswith(p):
            return t[len(p):].strip()
    return t
def detect_low_info_ground_truth(text: str, max_len: int, min_score: int) -> Dict[str, Any]:
    t = (text or "").strip()
    reasons: List[str] = []

    if not t:
        return {
            "is_low_info_ground_truth": True,
            "low_info_score": 999,
            "low_info_reasons": ["empty_ground_truth"],
            "ground_truth_len": 0,
        }

    core = strip_role_prefix(t)
    score = 0

    if len(core) <= max_len:
        score += 2
        reasons.append("short_text")
    elif len(core) <= max_len * 2:
        score += 1
        reasons.append("medium_short_text")

    workflow_hits = [k for k in LOW_INFO_WORKFLOW_KEYWORDS if k in core]
    if workflow_hits:
        score += 2
        reasons.append(f"workflow_keywords:{'|'.join(workflow_hits[:3])}")

    if core.endswith(("？", "?")):
        score += 1
        reasons.append("question_form")

    has_action = any(k in core for k in LOW_INFO_MEDICAL_ACTION_KEYWORDS)
    if not has_action:
        score += 1
        reasons.append("no_explicit_medical_action")

    polite_only_candidates = {"好的", "收到", "不客气", "嗯", "可以", "好"}
    if core in polite_only_candidates:
        score += 2
        reasons.append("polite_only_phrase")

    return {
        "is_low_info_ground_truth": score >= min_score,
        "low_info_score": score,
        "low_info_reasons": reasons,
        "ground_truth_len": len(core),
    }
def build_scoring_messages(ground_truth_text: str, prediction_text: str) -> List[Dict[str, str]]:
    system_prompt = (
        '你是一名资深心血管医生。'
        '请采用非对称方向评分：重点评估“预测文本”是否覆盖并遵循“真值文本”。'
        '真值的核心意图是首要标准；只要核心意图保留，额外的无关内容默认不扣分，除非它与真值明确冲突或存在明显安全风险。'
        '更激进、更多步骤、更多细节的处理方式，只要不违背真值核心意图，也默认不扣分。'
        '评分时整体继续从宽，不要因为表达更泛化、更展开、或附加兼容管理动作而轻易给低分。'
        '对于有效管理句、短医疗指令、检查复查住院建议，默认按中高分理解。'
        '对于边界样本默认上调：如果难以判断 3.0 和 3.5，优先给 3.5；如果难以判断 3.5 和 4.0，且方向一致并有实际覆盖，优先给更高分。'
        '只返回包含 final_score_0_5 和 rationale 的 JSON。'
    )

    user_prompt = (
        '请比较 [Ground Truth] 和 [Prediction] 的语义一致性。\n\n'
        '评分规则：\n'
        '1) 评分方向是非对称的：重点看 [Prediction] 是否覆盖并遵循 [Ground Truth]。\n'
        '2) 如果预测文本包含了真值中的核心意图/动作，即使预测更详细、更积极、更完整，也应该给高分。\n'
        '3) 只要是“部分但有意义”的覆盖，通常应达到 3.5 分或以上；不要把这类样本轻易打到 2.5 或以下。\n'
        '4) 预测中的额外诊断、额外建议、额外检查、额外药物或更激进的处理方式，默认不扣分，只要它没有与真值明确冲突，也不是明显不安全的医学建议。\n'
        '5) 0.0-2.5 只能用于以下情况：与真值核心意图基本无关、明确反对或替换了真值动作、换成了完全不同的临床场景、或者存在明显安全风险。\n'
        '6) 对于简短、流程型、管理型真值，只要预测文本保留了兼容的核心管理动作，就默认应给 3.5 分以上，而不是判成“没有重合”。\n'
        '7) 以下真值类型通常都属于有效临床指引，而不是应被苛刻打低分的弱匹配：追问近期症状或复查情况、建议线下复诊、建议住院评估、建议复查造影/检查、续方、按原剂量服药、安排门诊或检查流程。\n'
        '8) 如果真值是短管理句或流程句，但包含有效医疗动作，例如“请按原剂量服药”“建议线下复诊”“建议住院评估/复查造影”“继续吃并监测”，只要预测仍处于同一临床线程并保留该方向，通常应给 3.5 分以上。\n'
        '9) 如果预测把真值中的管理动作展开成更完整的执行方案，例如把“线下复诊”展开为更具体的检查、住院、造影、随访或药物安排，只要不与真值冲突，优先考虑 4.0-5.0。\n'
        '10) “开好了”“您看下”“各要多少”“一次只能 5 个药”这类纯流程、配药确认或数量确认，不要作为高分正例；如果这类弱真值没有额外医疗动作，不应因为语义接近就宽松给高分。\n'
        '11) “药开好了，请按原剂量服药”这类带明确用药指令的内容，不应按纯流程句处理。\n'
        '12) 对边界样本默认上调：如果难以判断 3.0 和 3.5，优先给 3.5；如果难以判断 3.5 和 4.0，且方向一致并有实际覆盖，优先给 4.0。\n'
        '13) 评分参考：\n'
        '   - 4.0-5.0：核心意图完整覆盖，没有冲突；可允许细节扩展、动作展开或更积极但兼容的处理\n'
        '   - 3.5：有意义的部分覆盖，或方向一致但不够具体；对短管理句/流程型有效真值尤其应从宽\n'
        '   - 3.0：存在部分重合，但保留的管理动作较弱或较泛\n'
        '   - 2.0-2.5：仅有少量相关点，且核心动作基本未被保留；这档分数要慎用\n'
        '   - 0.0-1.5：完全不相关、明确冲突、明确反向替代，或存在严重场景偏离/安全风险\n'
        '14) 请输出 0-5 范围内的 final_score_0_5，建议以 0.5 为步长，并给出简短 rationale。\n'
        '15) rationale 主要说明预测文本是否覆盖真值核心意图；不要把“额外无关内容”作为低分主因，除非它造成了冲突或安全问题。\n'
        '16) 不要输出域名、子分项、冲突标签、风险标签或其他额外字段。\n\n'
        '只返回 JSON：\n'
        '{\n'
        '  "final_score_0_5": number >= 0,\n'
        '  "rationale": "简短理由，120 字以内"\n'
        '}\n\n'
        f'[Ground Truth]\n{ground_truth_text}\n\n'
        f'[Prediction]\n{prediction_text}\n'
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
def call_llm_scoring(messages: List[Dict[str, str]]) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    client = get_thread_client()
    attempts: List[Dict[str, Any]] = []
    last_error = ""
    raw_text = ""

    for i in range(MAX_RETRIES + 1):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=SCORING_MODEL_NAME,
                messages=messages,
                temperature=LLM_TEMPERATURE,
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
            if i < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS[min(i, len(RETRY_BACKOFF_SECONDS) - 1)])
            else:
                break

    raise RuntimeError(
        json.dumps(
            {"error_message": last_error or "LLM call failed", "raw_text": raw_text, "attempts": attempts},
            ensure_ascii=False,
        )
    )


def _parse_number_field(payload: Dict[str, Any], key: str) -> float:
    if key not in payload:
        raise ValueError(f"missing_field:{key}")
    value = payload[key]
    if isinstance(value, bool):
        raise ValueError(f"invalid_field_type:{key}:bool")
    try:
        return float(value)
    except Exception as e:
        raise ValueError(f"invalid_field_type:{key}") from e


def normalize_scoring_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("invalid_payload_type")

    final_score = _parse_number_field(payload, "final_score_0_5")
    if final_score < 0:
        raise ValueError("invalid_field_value:final_score_0_5:negative")

    if "rationale" not in payload:
        raise ValueError("missing_field:rationale")
    rationale_raw = payload["rationale"]
    if not isinstance(rationale_raw, str):
        raise ValueError("invalid_field_type:rationale")
    rationale = rationale_raw.strip()

    return {
        "final_score_0_5": final_score,
        "rationale": rationale,
    }
def score_single_record(record: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    sample_index = record.get("sample_index")
    pid = record.get("pid")
    ground_truth_text = str(record.get("ground_truth_text", "") or "").strip()
    prediction_text = str(record.get("prediction_text", "") or "").strip()

    reference_quality = {
        "is_low_info_ground_truth": bool(record.get("_is_low_info_ground_truth", False)),
        "low_info_score": record.get("_low_info_score"),
        "low_info_reasons": record.get("_low_info_reasons", []),
        "ground_truth_len": record.get("_ground_truth_len"),
    }

    base = {
        "dataset_name": dataset_name,
        "sample_index": sample_index,
        "pid": pid,
        "upstream_status": record.get("status"),
        "ground_truth_text": ground_truth_text,
        "prediction_text": prediction_text,
        "reference_quality": reference_quality,
    }

    if not ground_truth_text or not prediction_text:
        return {
            **base,
            "status": "error",
            "error_message": "missing_ground_truth_or_prediction",
            "scoring": None,
            "debug_info": {},
        }

    t0 = time.time()
    messages = build_scoring_messages(ground_truth_text, prediction_text)
    try:
        parsed, raw_text, llm_debug = call_llm_scoring(messages)
        normalized = normalize_scoring_payload(parsed)
        elapsed = round(time.time() - t0, 3)
        return {
            **base,
            "status": "success",
            "error_message": "",
            "scoring": normalized,
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
            "status": "error",
            "error_message": str(e),
            "scoring": None,
            "processing_time_sec": elapsed,
            "debug_info": {"messages": messages},
        }
def load_eval_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_distribution(scores: List[float]) -> Dict[str, int]:
    bins = {f"{x/2:.1f}": 0 for x in range(0, 11)}
    for s in scores:
        k = f"{s:.1f}"
        if k not in bins:
            bins[k] = 0
        bins[k] += 1
    return bins


def summarize_scored_results(scored_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    success_items = [x for x in scored_results if x.get("status") == "success" and x.get("scoring")]
    error_items = [x for x in scored_results if x.get("status") == "error"]
    skipped_items = [x for x in scored_results if x.get("status") == "skipped"]

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
        "total_records": len(scored_results),
        "success_count": len(success_items),
        "error_count": len(error_items),
        "skipped_count": len(skipped_items),
        "average_final_score_0_5": avg_score,
        "score_distribution": build_distribution(final_scores),
        "low_score_samples_le_2": low_score_samples,
    }


def score_file(
    input_path: str,
    output_dir: str,
    max_samples: Optional[int],
    workers: int,
    low_info_max_len: int,
    low_info_min_score: int,
) -> Dict[str, Any]:
    safe_print(f"\n寮€濮嬭瘎鍒嗘枃浠? {input_path}")
    payload = load_eval_file(input_path)
    dataset_name = str(payload.get("metadata", {}).get("dataset_name", "unknown_dataset"))
    records = payload.get("results", [])
    if not isinstance(records, list):
        records = []
    if max_samples is not None and max_samples > 0:
        records = records[:max_samples]

    prepared: List[Dict[str, Any]] = []
    low_info_total = 0
    for rec in records:
        q = detect_low_info_ground_truth(
            text=str(rec.get("ground_truth_text", "") or ""),
            max_len=low_info_max_len,
            min_score=low_info_min_score,
        )
        item = dict(rec)
        item["_is_low_info_ground_truth"] = q["is_low_info_ground_truth"]
        item["_low_info_score"] = q["low_info_score"]
        item["_low_info_reasons"] = q["low_info_reasons"]
        item["_ground_truth_len"] = q["ground_truth_len"]
        if q["is_low_info_ground_truth"]:
            low_info_total += 1
        prepared.append(item)

    # 取消过滤机制：不再排除低信息样本，仍保留识别结果用于后续分析。
    filtered_records = prepared

    scored_results: List[Optional[Dict[str, Any]]] = [None] * len(filtered_records)
    progress_counter = [0]
    progress_lock = threading.Lock()

    def _worker(i: int, rec: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        result = i, score_single_record(rec, dataset_name)
        with progress_lock:
            progress_counter[0] += 1
            safe_print(f"杩涘害: {progress_counter[0]}/{len(filtered_records)}")
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_worker, i, rec) for i, rec in enumerate(filtered_records)]
        for fut in concurrent.futures.as_completed(futures):
            i, item = fut.result()
            scored_results[i] = item

    final_results = [x for x in scored_results if x is not None]
    summary = summarize_scored_results(final_results)
    summary["input_total_count"] = len(prepared)
    summary["low_info_detected_count"] = low_info_total
    summary["excluded_low_info_count"] = 0
    summary["kept_non_low_info_count"] = len(filtered_records)

    model_safe = SCORING_MODEL_NAME.replace("/", "_")
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_file = f"similarity_scores_{base_name}_{model_safe}.json"
    out_path = os.path.join(output_dir, out_file)

    out_payload = {
        "metadata": {
            "input_file": os.path.basename(input_path),
            "dataset_name": dataset_name,
            "scoring_model": {
                "name": SCORING_MODEL_NAME,
                "url": normalize_base_url(SCORING_API_URL),
                "api_key": "***hidden***",
            },
            "rules_version": "pure_llm_direct_score_v5_more_lenient_prompt",
            "score_granularity": "raw_llm",
            "low_info_filter": {
                "mode": "disabled",
                "max_len": low_info_max_len,
                "min_score": low_info_min_score,
            },
            "summary": summary,
            "timestamp": now_iso(),
        },
        "results": final_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    safe_print(f"瀹屾垚: {os.path.basename(input_path)} -> {out_file}")
    return {
        "input_file": os.path.basename(input_path),
        "output_file": out_file,
        "dataset_name": dataset_name,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM medical similarity scorer (direct final score by LLM).")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=DEFAULT_INPUT_FILES,
        help="Evaluation result files to score. Default: highrisk + chronic.",
    )
    parser.add_argument(
        "--output-dir",
        default=script_dir,
        help="Directory for per-file score outputs and global summary.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Only score first N samples per file.")
    parser.add_argument("--workers", type=int, default=DEFAULT_PARALLEL_WORKERS, help="Parallel workers per file.")
    parser.add_argument(
        "--low-info-max-len",
        type=int,
        default=30,
        help="短文本阈值（字符数）。",
    )
    parser.add_argument(
        "--low-info-min-score",
        type=int,
        default=3,
        help="低信息样本判定最小分数。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    safe_print("=" * 80)
    safe_print("Medical similarity scoring started (direct final score by LLM)")
    safe_print(f"Scoring model: {SCORING_MODEL_NAME} @ {normalize_base_url(SCORING_API_URL)}")
    safe_print(f"Workers: {args.workers}")
    safe_print(f"Low-info samples will NOT be excluded (filter disabled).")
    safe_print(f"Low-info detector: max_len={args.low_info_max_len}, min_score={args.low_info_min_score}")
    safe_print("=" * 80)

    input_paths: List[str] = []
    for p in args.inputs:
        input_paths.append(p if os.path.isabs(p) else os.path.join(script_dir, p))

    all_file_summaries: List[Dict[str, Any]] = []
    for input_path in input_paths:
        if not os.path.exists(input_path):
            safe_print(f"Skip missing file: {input_path}")
            continue
        all_file_summaries.append(
            score_file(
                input_path=input_path,
                output_dir=output_dir,
                max_samples=args.max_samples,
                workers=args.workers,
                low_info_max_len=args.low_info_max_len,
                low_info_min_score=args.low_info_min_score,
            )
        )

    global_success = sum(x["summary"]["success_count"] for x in all_file_summaries)
    global_errors = sum(x["summary"]["error_count"] for x in all_file_summaries)
    global_skipped = sum(x["summary"]["skipped_count"] for x in all_file_summaries)
    weighted_scores_sum = 0.0
    weighted_scores_count = 0
    for x in all_file_summaries:
        n = x["summary"]["success_count"]
        weighted_scores_sum += float(x["summary"]["average_final_score_0_5"]) * n
        weighted_scores_count += n
    global_avg = round(weighted_scores_sum / weighted_scores_count, 4) if weighted_scores_count else 0.0

    summary_payload = {
        "metadata": {
            "scoring_model": {
                "name": SCORING_MODEL_NAME,
                "url": normalize_base_url(SCORING_API_URL),
                "api_key": "***hidden***",
            },
            "rules_version": "pure_llm_direct_score_v5_more_lenient_prompt",
            "score_granularity": "raw_llm",
            "low_info_filter": {
                "mode": "disabled",
                "max_len": args.low_info_max_len,
                "min_score": args.low_info_min_score,
            },
            "timestamp": now_iso(),
        },
        "global_summary": {
            "files_processed": len(all_file_summaries),
            "global_success_count": global_success,
            "global_error_count": global_errors,
            "global_skipped_count": global_skipped,
            "global_average_final_score_0_5": global_avg,
        },
        "file_summaries": all_file_summaries,
    }

    summary_path = os.path.join(output_dir, "similarity_scores_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    safe_print("\nScoring completed")
    safe_print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()
