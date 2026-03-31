import argparse
import concurrent.futures
import json
import os
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
DEFAULT_PARALLEL_WORKERS = 4

DEFAULT_INPUT_FILES = [
    "evaluation_results_Qwen_Qwen3.5-9B_highrisk.json",
    "evaluation_results_Qwen_Qwen3.5-9B_chronic.json",
]

DOMAIN_KEYS = ["检查建议", "治疗建议", "用药建议"]
DOMAIN_ALIGN_VALUES = {"一致", "部分一致", "冲突", "未涉及"}

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
        print(*args, **kwargs)


def now_iso() -> str:
    return datetime.now().isoformat()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def round_to_half(v: float) -> float:
    return round(v * 2.0) / 2.0


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


def normalize_domain_alignment(value: Any) -> str:
    v = str(value or "").strip()
    if v in DOMAIN_ALIGN_VALUES:
        return v
    vn = v.lower()
    if v in ("完全一致", "高度一致") or "一致" in v:
        return "一致"
    if "部分" in v or "part" in vn:
        return "部分一致"
    if "冲突" in v or "相反" in v or "conflict" in vn or "opposite" in vn:
        return "冲突"
    return "未涉及"


def normalize_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value or "").strip()
    return [s] if s else []


def strip_role_prefix(text: str) -> str:
    t = (text or "").strip()
    for p in ("医生：", "医生:", "doctor:", "Doctor:"):
        if t.startswith(p):
            return t[len(p) :].strip()
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
        "你是资深心血管临床质控专家，负责对医学问答结果做语义一致性评分。"
        "评分重点是方向一致性，不要求检查/治疗/用药三类建议必须同时出现。"
        "请严格按规则评分，保持保守、稳定、可复核。"
    )

    user_prompt = (
        "请比较【原回答】与【预测回答】的医学语义一致性，并打0-5分（0.5步进）。\n\n"
        "评分规则（方向一致性优先）：\n"
        "1) direction_consistency_score: 0-3分\n"
        "- 3分：主要诊疗方向、病情判断、处理路径基本一致。\n"
        "- 2分：总体方向一致，但细节有偏差。\n"
        "- 1分：仅部分方向一致，核心信息缺失较多。\n"
        "- 0分：方向明显相反或无关。\n\n"
        "2) action_consistency_score: 0-2分（动态计分）\n"
        "- 逐域比较：检查建议、治疗建议、用药建议（每域标记 一致/部分一致/冲突/未涉及）。\n"
        "- 仅对任一文本中“有出现”的域进行评价，不要求三域齐全。\n"
        "- 新增域默认不扣分，除非与整体方向冲突或存在风险。\n\n"
        "3) penalty_score: 0-2分（扣分项）\n"
        "- 方向相反、明显医学错误或潜在高风险误导：重扣。\n"
        "- 与真值无关的大量展开但不冲突：轻扣或不扣。\n\n"
        "最终总分计算：final_score_0_5 = round_to_0.5(clamp(direction_consistency_score + action_consistency_score - penalty_score, 0, 5))。\n\n"
        "请仅返回JSON，不要返回其他文字，格式必须是：\n"
        "{\n"
        '  "direction_consistency_score": 0-3之间的数字(0.5步进),\n'
        '  "action_consistency_score": 0-2之间的数字(0.5步进),\n'
        '  "penalty_score": 0-2之间的数字(0.5步进),\n'
        '  "domain_alignment": {\n'
        '    "检查建议": "一致|部分一致|冲突|未涉及",\n'
        '    "治疗建议": "一致|部分一致|冲突|未涉及",\n'
        '    "用药建议": "一致|部分一致|冲突|未涉及"\n'
        "  },\n"
        '  "major_conflicts": ["若无则返回空数组"],\n'
        '  "risk_flags": ["若无则返回空数组"],\n'
        '  "rationale": "简要解释评分依据，不超过120字"\n'
        "}\n\n"
        f"【原回答】\n{ground_truth_text}\n\n"
        f"【预测回答】\n{prediction_text}\n"
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


def normalize_scoring_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    direction = round_to_half(clamp(float(payload.get("direction_consistency_score", 0.0)), 0.0, 3.0))
    action = round_to_half(clamp(float(payload.get("action_consistency_score", 0.0)), 0.0, 2.0))
    penalty = round_to_half(clamp(float(payload.get("penalty_score", 0.0)), 0.0, 2.0))

    domain_alignment_raw = payload.get("domain_alignment", {})
    if not isinstance(domain_alignment_raw, dict):
        domain_alignment_raw = {}
    domain_alignment: Dict[str, str] = {}
    for k in DOMAIN_KEYS:
        domain_alignment[k] = normalize_domain_alignment(domain_alignment_raw.get(k, "未涉及"))

    major_conflicts = normalize_string_list(payload.get("major_conflicts"))
    risk_flags = normalize_string_list(payload.get("risk_flags"))
    rationale = str(payload.get("rationale", "") or "").strip()

    final_score = round_to_half(clamp(direction + action - penalty, 0.0, 5.0))
    return {
        "final_score_0_5": final_score,
        "direction_consistency_score": direction,
        "action_consistency_score": action,
        "penalty_score": penalty,
        "domain_alignment": domain_alignment,
        "major_conflicts": major_conflicts,
        "risk_flags": risk_flags,
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
        "ground_truth_text": ground_truth_text,
        "prediction_text": prediction_text,
        "reference_quality": reference_quality,
    }

    if record.get("status") != "success":
        return {
            **base,
            "status": "skipped",
            "error_message": f"upstream_status={record.get('status')}",
            "scoring": None,
            "debug_info": {},
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

    low_info_success = [x for x in success_items if x.get("reference_quality", {}).get("is_low_info_ground_truth")]
    non_low_info_success = [x for x in success_items if not x.get("reference_quality", {}).get("is_low_info_ground_truth")]
    low_info_avg = (
        round(sum(float(x["scoring"]["final_score_0_5"]) for x in low_info_success) / len(low_info_success), 4)
        if low_info_success
        else 0.0
    )
    non_low_info_avg = (
        round(sum(float(x["scoring"]["final_score_0_5"]) for x in non_low_info_success) / len(non_low_info_success), 4)
        if non_low_info_success
        else 0.0
    )

    low_score_samples = [
        {
            "sample_index": x.get("sample_index"),
            "pid": x.get("pid"),
            "final_score_0_5": x["scoring"]["final_score_0_5"],
            "is_low_info_ground_truth": x.get("reference_quality", {}).get("is_low_info_ground_truth", False),
            "low_info_reasons": x.get("reference_quality", {}).get("low_info_reasons", []),
            "major_conflicts": x["scoring"].get("major_conflicts", []),
            "risk_flags": x["scoring"].get("risk_flags", []),
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
        "low_info_success_count": len(low_info_success),
        "non_low_info_success_count": len(non_low_info_success),
        "low_info_average_final_score_0_5": low_info_avg,
        "non_low_info_average_final_score_0_5": non_low_info_avg,
        "low_score_samples_le_2": low_score_samples,
    }


def score_file(
    input_path: str,
    output_dir: str,
    max_samples: Optional[int],
    workers: int,
    low_info_mode: str,
    low_info_max_len: int,
    low_info_min_score: int,
) -> Dict[str, Any]:
    safe_print(f"\n开始评分文件: {input_path}")
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

    if low_info_mode == "only":
        filtered_records = [x for x in prepared if x["_is_low_info_ground_truth"]]
    elif low_info_mode == "exclude":
        filtered_records = [x for x in prepared if not x["_is_low_info_ground_truth"]]
    else:
        filtered_records = prepared

    scored_results: List[Optional[Dict[str, Any]]] = [None] * len(filtered_records)

    def _worker(i: int, rec: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        return i, score_single_record(rec, dataset_name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_worker, i, rec) for i, rec in enumerate(filtered_records)]
        for fut in concurrent.futures.as_completed(futures):
            i, item = fut.result()
            scored_results[i] = item

    final_results = [x for x in scored_results if x is not None]
    summary = summarize_scored_results(final_results)
    summary["input_low_info_total"] = low_info_total
    summary["input_non_low_info_total"] = len(prepared) - low_info_total
    summary["kept_records_count"] = len(filtered_records)
    summary["low_info_mode"] = low_info_mode

    model_safe = SCORING_MODEL_NAME.replace("/", "_")
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    mode_suffix = "" if low_info_mode == "all" else f"_lowinfo_{low_info_mode}"
    out_file = f"similarity_scores_{base_name}_{model_safe}{mode_suffix}.json"
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
            "rules_version": "direction_consistency_v2",
            "score_granularity": "0.5",
            "low_info_filter": {
                "mode": low_info_mode,
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

    safe_print(f"完成: {os.path.basename(input_path)} -> {out_file}")
    return {
        "input_file": os.path.basename(input_path),
        "output_file": out_file,
        "dataset_name": dataset_name,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM medical similarity scorer (0-5, direction-consistency-first).")
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
        "--low-info-mode",
        choices=["all", "only", "exclude"],
        default="all",
        help="all=全部样本; only=仅低信息真值样本; exclude=排除低信息真值样本。",
    )
    parser.add_argument("--low-info-max-len", type=int, default=30, help="短文本阈值（字符数）。")
    parser.add_argument("--low-info-min-score", type=int, default=3, help="低信息判定最低分。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    safe_print("=" * 80)
    safe_print("医学相似度评分脚本启动（方向一致性优先）")
    safe_print(f"评分模型: {SCORING_MODEL_NAME} @ {normalize_base_url(SCORING_API_URL)}")
    safe_print(f"并发: {args.workers}")
    safe_print(
        f"低信息真值筛选: mode={args.low_info_mode}, max_len={args.low_info_max_len}, min_score={args.low_info_min_score}"
    )
    safe_print("=" * 80)

    input_paths: List[str] = []
    for p in args.inputs:
        input_paths.append(p if os.path.isabs(p) else os.path.join(script_dir, p))

    all_file_summaries: List[Dict[str, Any]] = []
    for input_path in input_paths:
        if not os.path.exists(input_path):
            safe_print(f"跳过不存在文件: {input_path}")
            continue
        all_file_summaries.append(
            score_file(
                input_path=input_path,
                output_dir=output_dir,
                max_samples=args.max_samples,
                workers=args.workers,
                low_info_mode=args.low_info_mode,
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
            "rules_version": "direction_consistency_v2",
            "score_granularity": "0.5",
            "low_info_filter": {
                "mode": args.low_info_mode,
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

    summary_suffix = "" if args.low_info_mode == "all" else f"_lowinfo_{args.low_info_mode}"
    summary_path = os.path.join(output_dir, f"similarity_scores_summary{summary_suffix}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    safe_print("\n评分完成")
    safe_print(f"汇总文件: {summary_path}")


if __name__ == "__main__":
    main()
