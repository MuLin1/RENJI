import argparse
import concurrent.futures
import json
import os
import re
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
        "name": "Qwen/Qwen3.5-9B",
        # OpenAI SDK 的 base_url 应该是到 /v1（不要写到 /chat/completions）。
        "url": "https://api.siliconflow.cn/v1",
        "key": "sk-kwcpucjoyfugtgwslskeueqxcdwbububczpltehjoxelukqy"
    }
]

DATASET_JOBS = [
    {"name": "highrisk", "annotation_file": "result_highrisk.json", "raw_dataset_file": "xnk_dataset_highrisk.jsonl"},
    {"name": "chronic", "annotation_file": "result_chronic.json", "raw_dataset_file": "xnk_dataset_chronic.jsonl"},
]

SKIP_ERRORS = True
TEST_SUBSET_SIZE = None
PARALLEL_WORKERS = 50
REQUEST_TIMEOUT_SEC = 60
LLM_TEMPERATURE = 0
MAX_CONTEXT_ROUNDS = 6
MAX_CONTEXT_CHARS = 5000

script_dir = os.path.dirname(os.path.abspath(__file__))
print_lock = threading.Lock()
thread_storage = threading.local()


def safe_print(*args: Any, **kwargs: Any) -> None:
    with print_lock:
        print(*args, **kwargs)


def normalize_text(text: str) -> str:
    return text.strip() if isinstance(text, str) else ""


def extract_pid(sample: Dict[str, Any], fallback_index: Optional[int] = None) -> Optional[str]:
    pid = sample.get("id", sample.get("pid"))
    if pid is not None and str(pid).strip():
        return str(pid)
    if fallback_index is not None:
        return f"idx_{fallback_index}"
    return None


def extract_dialogue(sample: Dict[str, Any]) -> List[Any]:
    dialogue = sample.get("dialogue")
    if isinstance(dialogue, list):
        return dialogue
    history = sample.get("history")
    if isinstance(history, list):
        return history
    return []


def is_patient_turn(text: str) -> bool:
    t = normalize_text(text)
    return t.startswith("患者：") or t.startswith("患者:")


def is_doctor_turn(text: str) -> bool:
    t = normalize_text(text)
    return t.startswith("医生：") or t.startswith("医生:")


def split_dialogue_into_rounds(dialogue: List[Any]) -> List[Dict[str, str]]:
    rounds: List[Dict[str, str]] = []
    pending_patient: Optional[str] = None

    for turn in dialogue:
        if isinstance(turn, dict):
            role = normalize_text(str(turn.get("role", ""))).lower()
            content = normalize_text(str(turn.get("content", "")))
            if not content:
                continue
            if role == "patient":
                pending_patient = content
                continue
            if role == "doctor":
                if pending_patient:
                    rounds.append({"patient": pending_patient, "doctor": content})
                    pending_patient = None
                continue
            continue
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


def limit_rounds_for_prompt(rounds: List[Dict[str, str]], k: int) -> Tuple[List[Dict[str, str]], bool]:
    if not rounds:
        return [], False

    usable_rounds = rounds[:k]
    truncated = False
    if MAX_CONTEXT_ROUNDS and len(usable_rounds) > MAX_CONTEXT_ROUNDS:
        usable_rounds = usable_rounds[-MAX_CONTEXT_ROUNDS:]
        truncated = True
    return usable_rounds, truncated


def limit_turns_for_prompt(turns: List[str], max_chars: int) -> Tuple[List[str], bool]:
    if not turns or max_chars <= 0:
        return turns, False

    kept: List[str] = []
    total_chars = 0
    truncated = False
    for turn in reversed(turns):
        text = normalize_text(turn)
        if not text:
            continue
        if len(text) > max_chars:
            text = "..." + text[-(max_chars - 3):] if max_chars > 3 else text[-max_chars:]
            truncated = True
        add_len = len(text) + (1 if kept else 0)
        if kept and total_chars + add_len > max_chars:
            truncated = True
            break
        kept.append(text)
        total_chars += add_len

    kept.reverse()
    return kept, truncated


def build_prompt_dialogue(rounds: List[Dict[str, str]], k: int) -> Tuple[str, List[str], Dict[str, Any]]:
    prompt_rounds, rounds_truncated = limit_rounds_for_prompt(rounds, k)
    prompt_turns = flatten_rounds(prompt_rounds, len(prompt_rounds))
    prompt_turns, chars_truncated = limit_turns_for_prompt(prompt_turns, MAX_CONTEXT_CHARS)
    prompt_text = "\n".join(prompt_turns)
    meta = {
        "prompt_rounds_used": len(prompt_rounds),
        "prompt_turns_used": len(prompt_turns),
        "prompt_rounds_truncated": rounds_truncated,
        "prompt_chars_truncated": chars_truncated,
    }
    return prompt_text, prompt_turns, meta


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

        # 取消过滤机制：不再按 status / k / pid / has_explicit_diagnosis_advice 过滤样本。
        # 仅做基本健壮性处理：跳过非 dict 的异常条目。
        if isinstance(results, list):
            results = [r for r in results if isinstance(r, dict)]
        else:
            results = []

        label_false_count = sum(1 for r in results if r.get("has_explicit_diagnosis_advice") is False)
        label_true_count = sum(1 for r in results if r.get("has_explicit_diagnosis_advice") is True)
        safe_print(f"筛查标签统计: has_explicit=true {label_true_count}, false {label_false_count}")
        return results
    except ValueError:
        raise
    except Exception as e:
        safe_print(f"加载标注数据失败: {annotation_path}, 错误: {e}")
        return None


def load_raw_data_mapping(raw_dataset_file: str) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(script_dir, raw_dataset_file)
    mapping: Dict[str, Dict[str, Any]] = {}
    try:
        sample_index = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pid = extract_pid(obj, sample_index)
                sample_index += 1
                if pid:
                    mapping[pid] = obj
        safe_print(f"加载原始数据映射: {raw_dataset_file}, 共 {len(mapping)} 条")
    except Exception as e:
        safe_print(f"加载原始数据失败: {path}, 错误: {e}")
    return mapping


def build_input_from_annotation(annotation_sample: Dict[str, Any], raw_mapping: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    pid = annotation_sample.get("pid")
    raw_sample = raw_mapping.get(pid, {})
    if not raw_sample and pid is not None:
        raw_sample = raw_mapping.get(str(pid), {})
    raw_dialogue = extract_dialogue(raw_sample)

    rounds = split_dialogue_into_rounds(raw_dialogue)
    try:
        requested_k = int(annotation_sample.get("k", 1))
    except Exception:
        requested_k = 1
    k = max(1, min(requested_k, len(rounds))) if rounds else 0
    k_dialogue = flatten_rounds(rounds, k) if k > 0 else []
    prompt_dialogue_text, prompt_dialogue_k, prompt_meta = build_prompt_dialogue(rounds, k)

    return {
        "pid": pid,
        "k": k,
        "requested_k": requested_k,
        "total_rounds": len(rounds),
        "input_dialogue_k": k_dialogue,
        "prompt_dialogue_text": prompt_dialogue_text,
        "prompt_dialogue_k": prompt_dialogue_k,
        "prompt_rounds_used": prompt_meta["prompt_rounds_used"],
        "prompt_turns_used": prompt_meta["prompt_turns_used"],
        "prompt_rounds_truncated": prompt_meta["prompt_rounds_truncated"],
        "prompt_chars_truncated": prompt_meta["prompt_chars_truncated"],
        "ground_truth_text": annotation_sample.get("doctor_diagnosis_raw", ""),
        "has_explicit_diagnosis_advice": bool(annotation_sample.get("has_explicit_diagnosis_advice", False)),
    }


def extract_content_from_completion(resp: Any) -> str:
    try:
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return str(resp)


def clean_prediction_text_regex(text: str) -> str:
    raw = normalize_text(text)
    if not raw:
        return ""

    answer_pattern = re.compile(r"(?:最终答案|答案|结论|Final Answer|Final|Answer)\s*[:：]\s*", re.IGNORECASE)
    matches = list(answer_pattern.finditer(raw))
    if matches:
        candidate = normalize_text(raw[matches[-1].end():])
        if candidate:
            return candidate

    reasoning_patterns = [
        r"(?is)^\s*(?:Thinking\s*Process|Analysis|Reasoning|思考过程)\s*[:：]?\s*.*?(?=(?:\n\s*\n)|\Z)",
        r"(?is)^\s*<think>.*?</think>\s*",
        r"(?is)^\s*<analysis>.*?</analysis>\s*",
        r"(?is)^\s*<reasoning>.*?</reasoning>\s*",
    ]
    cleaned = raw
    for pattern in reasoning_patterns:
        cleaned = re.sub(pattern, "", cleaned).strip()
    if cleaned and cleaned != raw:
        return cleaned

    paragraph_candidates = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if len(paragraph_candidates) > 1:
        tail = paragraph_candidates[-1]
        if not re.match(r"(?is)^(?:thinking\s*process|analysis|reasoning|思考过程)\s*[:：]?", tail):
            return tail

    line_candidates = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(line_candidates) > 1:
        for line in reversed(line_candidates):
            if re.match(r"(?is)^(?:thinking|analysis|reasoning|思考过程)\b", line):
                continue
            return line

    return raw


def generate_concise_diagnosis_text_regex(
    client: Any,
    model_name: str,
    prompt_dialogue_text: str,
    k: int,
) -> Tuple[str, List[Dict[str, str]]]:
    system_prompt = (
        "你是医疗对话分析助手。"
        "只输出医生最终会给出的核心诊疗建议。"
    )
    user_prompt = (
        f"下面是前{k}轮患者-医生对话（每轮一问一答）：\n{prompt_dialogue_text}\n\n"
        "请直接给出医生最终会说的那一句核心诊疗建议：\n"
        "1. 不要输出标题、编号、markdown。\n"
        "2. 保留具体的检查、治疗、用药、复查、住院、手术或随访建议。\n"
        "3. 不要补充对话里没有的信息，不要写免责声明。\n"
        "4. 如果只是确认、预约、流程沟通或单纯问药量，就输出最核心的一句原意。\n"
        "5. 字数精简到60字以内，精简包含核心信息。"
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=LLM_TEMPERATURE,
    )
    return clean_prediction_text_regex(extract_content_from_completion(resp)), messages


def clean_prediction_text(text: str) -> str:
    raw = normalize_text(text)
    if not raw:
        return ""

    final_markers = [
        "最终答案：",
        "最终答案:",
        "答案：",
        "答案:",
        "结论：",
        "结论:",
        "Final Answer:",
        "Final:",
        "Answer:",
    ]
    for marker in final_markers:
        idx = raw.rfind(marker)
        if idx != -1:
            candidate = normalize_text(raw[idx + len(marker):])
            if candidate:
                return candidate

    thinking_markers = [
        "Thinking Process:",
        "Thinking Process：",
        "思考过程：",
        "Analysis:",
        "Analysis：",
        "Reasoning:",
        "Reasoning：",
    ]
    for marker in thinking_markers:
        if raw.startswith(marker):
            tail = normalize_text(raw[len(marker):])
            paragraphs = [p.strip() for p in tail.split("\n\n") if p.strip()]
            if paragraphs:
                return paragraphs[-1]
            lines = [line.strip() for line in tail.splitlines() if line.strip()]
            if lines:
                return lines[-1]
            return tail

    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs[-1]

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) > 1:
        for line in reversed(lines):
            if line.lower().startswith(("thinking", "analysis", "reasoning")):
                continue
            return line
    if lines:
        return lines[-1]
    return raw


def generate_concise_diagnosis_text(
    client: Any,
    model_name: str,
    prompt_dialogue_text: str,
    k: int,
) -> Tuple[str, List[Dict[str, str]]]:
    system_prompt = (
        "你是医疗对话结果抽取器。"
        "只输出医生最终会给出的核心诊疗建议，不要展开分析，不要输出思考过程。"
    )
    user_prompt = (
        f"下面是前{k}轮患者-医生对话（每轮一问一答）：\n{prompt_dialogue_text}\n\n"
        "请直接给出医生最终会说的那一句核心诊疗建议：\n"
        "1. 不要输出思考过程、分析步骤、标题、编号、markdown。\n"
        "2. 保留具体的检查、治疗、用药、复查、住院、手术或随访建议。\n"
        "3. 不要补充对话里没有的信息，不要写免责声明。\n"
        "4. 如果只是确认、预约、流程沟通或单纯问药量，就输出最核心的一句原意。\n"
        "5. 结论尽量简洁精炼，不超过60字。"
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=LLM_TEMPERATURE,
    )
    return clean_prediction_text(extract_content_from_completion(resp)), messages


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
        prediction_text, messages = generate_concise_diagnosis_text_regex(
            client=client,
            model_name=model_name,
            prompt_dialogue_text=input_payload["prompt_dialogue_text"],
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
            "prompt_dialogue_k": input_payload["prompt_dialogue_k"],
            "prompt_dialogue_text": input_payload["prompt_dialogue_text"],
            "prompt_rounds_used": input_payload["prompt_rounds_used"],
            "prompt_turns_used": input_payload["prompt_turns_used"],
            "prompt_rounds_truncated": input_payload["prompt_rounds_truncated"],
            "prompt_chars_truncated": input_payload["prompt_chars_truncated"],
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
            "prompt_dialogue_k": input_payload["prompt_dialogue_k"],
            "prompt_dialogue_text": input_payload["prompt_dialogue_text"],
            "prompt_rounds_used": input_payload["prompt_rounds_used"],
            "prompt_turns_used": input_payload["prompt_turns_used"],
            "prompt_rounds_truncated": input_payload["prompt_rounds_truncated"],
            "prompt_chars_truncated": input_payload["prompt_chars_truncated"],
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

        worker_count = max(1, min(PARALLEL_WORKERS, total_samples))
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
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
