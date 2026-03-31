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
# Config
# =============================================================================

MODELS_TO_TEST = [
    {
        "name": "Qwen3.5-9B",
        # OpenAI SDK 的 base_url 应该是到 /v1（不要写到 /chat/completions）。
        "url": "http://113.59.64.93:9090/v1",
        "key": ""
    }
]

DATASET_JOBS = [
    {"name": "highrisk", "annotation_file": "result_highrisk.json", "raw_dataset_file": "xnk_dataset_highrisk.jsonl"},
    {"name": "chronic", "annotation_file": "result_chronic.json", "raw_dataset_file": "xnk_dataset_chronic.jsonl"},
]

SKIP_ERRORS = True
TEST_SUBSET_SIZE = None
PARALLEL_WORKERS = 50
REQUEST_TIMEOUT_SEC = 45
LLM_TEMPERATURE = 0

script_dir = os.path.dirname(os.path.abspath(__file__))
print_lock = threading.Lock()
thread_storage = threading.local()


def safe_print(*args: Any, **kwargs: Any) -> None:
    with print_lock:
        print(*args, **kwargs)


def normalize_text(text: str) -> str:
    return text.strip() if isinstance(text, str) else ""


def is_patient_turn(text: str) -> bool:
    t = normalize_text(text)
    return t.startswith("患者：") or t.startswith("患者:")


def is_doctor_turn(text: str) -> bool:
    t = normalize_text(text)
    return t.startswith("医生：") or t.startswith("医生:")


def split_dialogue_into_rounds(dialogue: List[str]) -> List[Dict[str, str]]:
    rounds: List[Dict[str, str]] = []
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
            if pending_patient:
                rounds.append({"patient": pending_patient, "doctor": t})
                pending_patient = None
            continue

    return rounds


def flatten_rounds(rounds: List[Dict[str, str]], k: int) -> List[str]:
    turns: List[str] = []
    for r in rounds[:k]:
        turns.append(r["patient"])
        turns.append(r["doctor"])
    return turns


def serialize_raw_response(response: Any) -> str:
    try:
        if hasattr(response, "model_dump_json"):
            return response.model_dump_json()
        if hasattr(response, "model_dump"):
            return json.dumps(response.model_dump(), ensure_ascii=False)
        if isinstance(response, (dict, list)):
            return json.dumps(response, ensure_ascii=False)
    except Exception:
        pass
    try:
        return str(response)
    except Exception:
        return repr(response)


def normalize_base_url(url: str) -> str:
    u = (url or "").rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/responses"):
        if u.endswith(suffix):
            u = u[: -len(suffix)]
            break
    return u


class Qwen3CompletionsWrapper:
    def __init__(self, real_completions: Any):
        self._real_completions = real_completions

    def create(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["temperature"] = LLM_TEMPERATURE
        kwargs.setdefault("timeout", REQUEST_TIMEOUT_SEC)
        kwargs.setdefault("extra_body", {})
        kwargs["extra_body"]["enable_thinking"] = False
        response = self._real_completions.create(*args, **kwargs)
        try:
            thread_storage.captured_raw_responses.append(serialize_raw_response(response))
        except Exception:
            pass
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_completions, name)


class Qwen3ChatWrapper:
    def __init__(self, real_chat: Any):
        self._real_chat = real_chat
        self.completions = Qwen3CompletionsWrapper(real_chat.completions)

    def __getattr__(self, name: str) -> Any:
        if name == "completions":
            return self.completions
        return getattr(self._real_chat, name)


class Qwen3ClientWrapper:
    def __init__(self, real_client: OpenAI):
        self._real_client = real_client
        self.chat = Qwen3ChatWrapper(real_client.chat)

    def __getattr__(self, name: str) -> Any:
        if name == "chat":
            return self.chat
        return getattr(self._real_client, name)


def load_annotation_data(annotation_file: str) -> Optional[List[Dict[str, Any]]]:
    annotation_path = os.path.join(script_dir, annotation_file)
    try:
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        safe_print(f"加载标注数据: {annotation_file}, 共 {len(results)} 条")
        missing_label_samples = [
            (idx, r.get("pid"))
            for idx, r in enumerate(results)
            if "has_explicit_diagnosis_advice" not in r
        ]
        if missing_label_samples:
            preview = ", ".join([f"index={idx},pid={pid}" for idx, pid in missing_label_samples[:5]])
            raise ValueError(
                f"Annotation file missing required field 'has_explicit_diagnosis_advice' in "
                f"{len(missing_label_samples)} samples. Examples: {preview}"
            )

        label_false_count = sum(1 for r in results if r.get("has_explicit_diagnosis_advice") is False)
        label_true_count = sum(1 for r in results if r.get("has_explicit_diagnosis_advice") is True)
        safe_print(f"筛查标签统计: has_explicit=true {label_true_count}, false {label_false_count}")

        if not SKIP_ERRORS:
            filtered = [r for r in results if r.get("has_explicit_diagnosis_advice") is True]
            safe_print(f"仅按筛查标签过滤后样本: {len(filtered)} 条")
            return filtered

        filtered = [
            r
            for r in results
            if r.get("status") == "success"
            and isinstance(r.get("k"), int)
            and r.get("k", 0) > 0
            and r.get("pid")
            and r.get("has_explicit_diagnosis_advice") is True
        ]
        safe_print(
            f"过滤统计: 总样本 {len(results)} 条, 标签为否 {label_false_count} 条, 最终可评估 {len(filtered)} 条"
        )
        return filtered
    except ValueError:
        raise
    except Exception as e:
        safe_print(f"加载标注数据失败: {annotation_path}, 错误: {e}")
        return None


def load_raw_data_mapping(raw_dataset_file: str) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(script_dir, raw_dataset_file)
    mapping: Dict[str, Dict[str, Any]] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pid = obj.get("id", obj.get("pid"))
                if pid:
                    mapping[pid] = obj
        safe_print(f"加载原始数据映射: {raw_dataset_file}, 共 {len(mapping)} 条")
    except Exception as e:
        safe_print(f"加载原始数据失败: {path}, 错误: {e}")
    return mapping


def build_input_from_annotation(annotation_sample: Dict[str, Any], raw_mapping: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    pid = annotation_sample.get("pid")
    raw_sample = raw_mapping.get(pid, {})
    raw_dialogue = raw_sample.get("dialogue", [])
    if not isinstance(raw_dialogue, list):
        raw_dialogue = []

    rounds = split_dialogue_into_rounds(raw_dialogue)
    requested_k = int(annotation_sample.get("k", 1))
    k = max(1, min(requested_k, len(rounds))) if rounds else 0
    k_dialogue = flatten_rounds(rounds, k) if k > 0 else []

    return {
        "pid": pid,
        "k": k,
        "requested_k": requested_k,
        "total_rounds": len(rounds),
        "input_dialogue_k": k_dialogue,
        "ground_truth_text": annotation_sample.get("doctor_diagnosis_raw", ""),
        "has_explicit_diagnosis_advice": bool(annotation_sample.get("has_explicit_diagnosis_advice", False)),
    }


def extract_content_from_completion(resp: Any) -> str:
    try:
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return str(resp)


def generate_diagnosis_text(client: Any, model_name: str, input_dialogue_k: List[str], k: int) -> Tuple[str, List[Dict[str, str]]]:
    dialogue_text = "\n".join(input_dialogue_k)
    system_prompt = (
        "你是心内科医生助手。"
        "请仅根据已给出的对话内容输出诊疗判断，不能假设额外信息。"
    )
    user_prompt = (
        f"以下是前{k}轮患者-医生对话（每轮为一问一答）:\n{dialogue_text}\n\n"
        "请输出一段简洁的诊疗结论（自由文本），尽量涵盖：可能诊断方向、建议检查、建议治疗/用药。"
        "如果信息不足，请明确说明哪些关键信息缺失。"
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=LLM_TEMPERATURE,
    )
    return extract_content_from_completion(resp), messages


def evaluate_single_sample(
    annotation_sample: Dict[str, Any],
    sample_index: int,
    total_count: int,
    model_name: str,
    client: Any,
    raw_mapping: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    thread_storage.captured_raw_responses = []
    safe_print(f"[{model_name}] 评估样本 {sample_index + 1}/{total_count}")
    start_time = time.time()

    input_payload = build_input_from_annotation(annotation_sample, raw_mapping)
    if input_payload["k"] <= 0 or not input_payload["input_dialogue_k"]:
        elapsed = time.time() - start_time
        return {
            "sample_index": sample_index,
            "pid": input_payload["pid"],
            "k": input_payload["k"],
            "has_explicit_diagnosis_advice": input_payload["has_explicit_diagnosis_advice"],
            "status": "error",
            "error_message": "No valid complete rounds for k",
            "processing_time": f"{elapsed:.3f}s",
            "debug_info": {"messages": [], "captured_raw_responses": []},
        }

    try:
        prediction_text, messages = generate_diagnosis_text(
            client=client,
            model_name=model_name,
            input_dialogue_k=input_payload["input_dialogue_k"],
            k=input_payload["k"],
        )
        elapsed = time.time() - start_time
        return {
            "sample_index": sample_index,
            "pid": input_payload["pid"],
            "k": input_payload["k"],
            "requested_k": input_payload["requested_k"],
            "total_rounds": input_payload["total_rounds"],
            "input_dialogue_k": input_payload["input_dialogue_k"],
            "has_explicit_diagnosis_advice": input_payload["has_explicit_diagnosis_advice"],
            "prediction_text": prediction_text,
            "ground_truth_text": input_payload["ground_truth_text"],
            "status": "success",
            "processing_time": f"{elapsed:.3f}s",
            "debug_info": {
                "messages": messages,
                "captured_raw_responses": list(getattr(thread_storage, "captured_raw_responses", [])),
            },
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "sample_index": sample_index,
            "pid": input_payload["pid"],
            "k": input_payload["k"],
            "requested_k": input_payload["requested_k"],
            "total_rounds": input_payload["total_rounds"],
            "input_dialogue_k": input_payload["input_dialogue_k"],
            "has_explicit_diagnosis_advice": input_payload["has_explicit_diagnosis_advice"],
            "status": "error",
            "error_message": str(e),
            "processing_time": f"{elapsed:.3f}s",
            "debug_info": {
                "messages": [],
                "captured_raw_responses": list(getattr(thread_storage, "captured_raw_responses", [])),
            },
        }


def summarize_round_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [r for r in results if r.get("status") == "success"]
    times = []
    for r in successful:
        try:
            times.append(float(str(r.get("processing_time", "0")).rstrip("s")))
        except Exception:
            pass
    avg_time = sum(times) / len(times) if times else 0.0
    return {
        "total_samples": len(results),
        "successful_samples": len(successful),
        "failed_samples": len(results) - len(successful),
        "average_processing_time": f"{avg_time:.3f}s",
    }


def write_raw_responses_log(model_name: str, dataset_name: str, results: List[Dict[str, Any]]) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"raw_llm_responses_{model_name.replace('/', '_')}_{dataset_name}_{ts}.jsonl"
    path = os.path.join(script_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            rec = {
                "sample_index": r.get("sample_index"),
                "pid": r.get("pid"),
                "status": r.get("status"),
                "raw_responses": r.get("debug_info", {}).get("captured_raw_responses", []),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    safe_print(f"[{model_name}] 原始响应日志已保存: {path}")


def run_evaluation_for_model(
    model_config: Dict[str, str],
    dataset_name: str,
    annotation_samples: List[Dict[str, Any]],
    raw_mapping: Dict[str, Dict[str, Any]],
    subset_size: Optional[int] = None,
) -> Dict[str, Any]:
    model_name = model_config["name"]
    safe_print(f"\n{'=' * 80}\n开始评估模型: {model_name}, 数据集: {dataset_name}\n{'=' * 80}")
    try:
        base_url = normalize_base_url(model_config["url"])
        if base_url != (model_config["url"] or "").rstrip("/"):
            safe_print(f"[{model_name}] 注意: 已将 url 归一化为 base_url={base_url}")
        real_client = OpenAI(
            base_url=base_url,
            api_key=model_config["key"],
            timeout=REQUEST_TIMEOUT_SEC,
            max_retries=1,
        )
        if "qwen3" in model_name.lower() or model_name.lower().startswith("qwen/"):
            client = Qwen3ClientWrapper(real_client)
        else:
            client = real_client

        effective_subset_size = subset_size if subset_size is not None else TEST_SUBSET_SIZE
        samples_to_test = annotation_samples[:effective_subset_size] if effective_subset_size else annotation_samples
        total_samples = len(samples_to_test)
        if total_samples == 0:
            return {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "status": "error",
                "error": "No samples to evaluate",
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = [
                executor.submit(
                    evaluate_single_sample,
                    s,
                    i,
                    total_samples,
                    model_name,
                    client,
                    raw_mapping,
                )
                for i, s in enumerate(samples_to_test)
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        sorted_results = sorted(results, key=lambda x: x.get("sample_index", 0))
        summary = summarize_round_results(sorted_results)
        output_file = f"evaluation_results_{model_name.replace('/', '_')}_{dataset_name}.json"
        output_path = os.path.join(script_dir, output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "dataset_name": dataset_name,
                        "model_config": model_config,
                        "summary": summary,
                        "timestamp": datetime.now().isoformat(),
                    },
                    "results": sorted_results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        write_raw_responses_log(model_name, dataset_name, sorted_results)
        safe_print(f"[{model_name}] 评估完成，输出: {output_file}")
        return {
            "dataset_name": dataset_name,
            "model_name": model_name,
            "status": "success",
            "output_file": output_file,
            "summary": summary,
        }
    except Exception as e:
        safe_print(f"[{model_name}] 评估失败: {e}")
        return {"dataset_name": dataset_name, "model_name": model_name, "status": "error", "error": str(e)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按前k轮诊断评估脚本")
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="只评估前 N 条有效样本（过滤后）。默认：评估全部。",
    )
    return parser.parse_args()


def main() -> None:
    safe_print(f"{'=' * 80}\n按前k轮诊断评估脚本\n{'=' * 80}")
    start_time = time.time()
    args = parse_args()
    if args.subset_size is not None and args.subset_size <= 0:
        raise SystemExit("--subset-size must be a positive integer")

    all_results: List[Dict[str, Any]] = []
    for dataset_job in DATASET_JOBS:
        dataset_name = dataset_job["name"]
        annotation_file = dataset_job["annotation_file"]
        raw_dataset_file = dataset_job["raw_dataset_file"]

        safe_print(f"\n{'-' * 80}\n处理数据集: {dataset_name}\n标注文件: {annotation_file}\n原始文件: {raw_dataset_file}\n{'-' * 80}")
        annotation_samples = load_annotation_data(annotation_file)
        raw_mapping = load_raw_data_mapping(raw_dataset_file)
        if not annotation_samples:
            safe_print(f"{dataset_name}: 无可评估样本，跳过")
            continue

        for model_config in MODELS_TO_TEST:
            all_results.append(
                run_evaluation_for_model(
                    model_config,
                    dataset_name,
                    annotation_samples,
                    raw_mapping,
                    subset_size=args.subset_size,
                )
            )

    summary_path = os.path.join(script_dir, "evaluation_results_all_datasets_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "dataset_jobs": DATASET_JOBS,
                "model_jobs": MODELS_TO_TEST,
                "results": all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    safe_print(f"\n汇总文件已保存: {summary_path}")
    safe_print(f"总耗时: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
