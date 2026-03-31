import json
import os
import sys
import time
from datetime import datetime
from unittest.mock import patch
import threading
import concurrent.futures
import statistics

# =============================================================================
# 可配置区域 - 请根据需要修改以下参数
# =============================================================================

MODELS_TO_TEST = [
    {
        "name": "Qwen3.5-9B",
        "url": "http://113.59.64.93:9090/v1",
        "key": ""
    }
]

DATASET_JOBS = [
    {"name": "highrisk", "annotation_file": "result_highrisk.json", "raw_dataset_file": "xnk_dataset_highrisk.jsonl"},
    {"name": "chronic", "annotation_file": "result_chronic.json", "raw_dataset_file": "xnk_dataset_chronic.jsonl"},
]
TEST_SUBSET_SIZE = None
SKIP_ERRORS = True
NUM_ROUNDS = 1
PARALLEL_WORKERS = 5
REQUEST_TIMEOUT_SEC = 45

# =============================================================================
# OpenAI 客户端包装器 (用于注入 Qwen3 特定参数)
# =============================================================================

class Qwen3CompletionsWrapper:
    """精准包装 'completions' 对象，拦截 'create' 调用"""
    def __init__(self, real_completions):
        self._real_completions = real_completions

    def create(self, *args, **kwargs):
        """在调用真正的 create 方法前，注入 extra_body 参数"""
        if 'temperature' not in kwargs:
            kwargs['temperature'] = 0
        if 'timeout' not in kwargs:
            kwargs['timeout'] = REQUEST_TIMEOUT_SEC
        if 'extra_body' not in kwargs:
            kwargs['extra_body'] = {}
        kwargs['extra_body']["enable_thinking"] = False
        response = self._real_completions.create(*args, **kwargs)
        try:
            raw_text = serialize_raw_response(response)
            thread_storage.captured_raw_responses.append(raw_text)
            safe_print(f"【原始返回】{raw_text}")
        except Exception:
            pass
        return response

    def __getattr__(self, name):
        """将所有其他属性访问（如 acreate）都传递给真实对象"""
        return getattr(self._real_completions, name)

class Qwen3ChatWrapper:
    """包装 'chat' 对象"""
    def __init__(self, real_chat):
        self._real_chat = real_chat
        # 用我们的包装器替换真实的 completions 对象
        self.completions = Qwen3CompletionsWrapper(real_chat.completions)

    def __getattr__(self, name):
        """确保对 self.completions 的访问返回的是我们的包装器"""
        if name == "completions":
            return self.completions
        return getattr(self._real_chat, name)

class Qwen3ClientWrapper:
    """包装顶层 OpenAI 客户端对象"""
    def __init__(self, real_client):
        self._real_client = real_client
        # 用我们的包装器替换真实的 chat 对象
        self.chat = Qwen3ChatWrapper(real_client.chat)

    def __getattr__(self, name):
        """确保对 self.chat 的访问返回的是我们的包装器"""
        if name == "chat":
            return self.chat
        return getattr(self._real_client, name)

class RawCaptureCompletionsWrapper:
    def __init__(self, real_completions):
        self._real_completions = real_completions

    def create(self, *args, **kwargs):
        if 'temperature' not in kwargs:
            kwargs['temperature'] = 0
        if 'timeout' not in kwargs:
            kwargs['timeout'] = REQUEST_TIMEOUT_SEC
        response = self._real_completions.create(*args, **kwargs)
        try:
            raw_text = serialize_raw_response(response)
            thread_storage.captured_raw_responses.append(raw_text)
            safe_print(f"【原始返回】{raw_text}")
        except Exception:
            pass
        return response

    def __getattr__(self, name):
        return getattr(self._real_completions, name)

class RawCaptureChatWrapper:
    def __init__(self, real_chat):
        self._real_chat = real_chat
        self.completions = RawCaptureCompletionsWrapper(real_chat.completions)

    def __getattr__(self, name):
        if name == "completions":
            return self.completions
        return getattr(self._real_chat, name)

class RawCaptureClientWrapper:
    def __init__(self, real_client):
        self._real_client = real_client
        self.chat = RawCaptureChatWrapper(real_client.chat)

    def __getattr__(self, name):
        if name == "chat":
            return self.chat
        return getattr(self._real_client, name)

# =============================================================================
# 并行化与系统集成
# =============================================================================

print_lock = threading.Lock()
def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
module_candidates = [
    os.path.join(project_root, 'new_frame'),
    os.path.join(project_root, 'paper_version')
]
new_frame_path = next((path for path in module_candidates if os.path.exists(path)), None)
if new_frame_path and new_frame_path not in sys.path:
    sys.path.insert(0, new_frame_path)

safe_print("正在导入模块...")
try:
    import new_app
    from new_app import (
        select_best_matching_tree, process_medical_archive_info,
        get_template_LLM_res as original_get_template_LLM_res, department_manager
    )
    from openai import OpenAI
    safe_print("✓ 成功导入 new_frame 模块")
except ImportError as e:
    safe_print(f"✗ 无法从 'new_frame' 导入模块: {e}")
    sys.exit(1)

thread_storage = threading.local()
request_storage = threading.local()

def get_request_logger():
    import logging
    return logging.getLogger(__name__)

def serialize_raw_response(response):
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

# =============================================================================
# 核心功能函数
# =============================================================================

def get_template_LLM_res_wrapper(template, patient_say, history, **kwargs):
    if isinstance(history, list):
        history_text = "\n".join(history) if history else "无"
    else:
        history_text = str(history) if history else "无"
    template_params = {'history': history_text, 'patient_say': patient_say, **kwargs}
    try:
        formatted_prompt = template.format(**template_params)
        messages = [{"role": "system", "content": "你是一个有用的医生助手，使用纯文本输出，不要用markdown格式。"},
                    {"role": "user", "content": formatted_prompt}]
        thread_storage.captured_prompts.append(json.dumps(messages, ensure_ascii=False, indent=2))
    except KeyError as e:
        thread_storage.captured_prompts.append(f"PROMPT_FORMAT_ERROR: Missing key {e}")
    
    raw_output = None
    try:
        raw_output = original_get_template_LLM_res(template, patient_say, history, **kwargs)
    except Exception:
        safe_print(f"!!!!!!!!!! 调用大模型API时捕获到异常 !!!!!!!!!!")
        with print_lock:
            import traceback
            traceback.print_exc()
        safe_print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    thread_storage.captured_outputs.append(str(raw_output))
    return raw_output

def load_annotation_data(annotation_file):
    annotation_path = os.path.join(script_dir, annotation_file)
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f: data = json.load(f)
        results = data.get('results', [])
        safe_print(f"成功加载标注数据，共 {len(results)} 个样本")
        if SKIP_ERRORS:
            # =================================================================
            # 修改点 1: 移除了 r.get('selected_tree') != '其他' 的条件
            # 现在只过滤状态不为 'success' 或没有 'selected_tree' 字段的样本
            # =================================================================
            valid_results = [r for r in results if r.get('status') == 'success' and r.get('selected_tree')]
            safe_print(f"过滤后有效样本: {len(valid_results)} 个")
            return valid_results
        # =====================================================================
        # 修改点 2: 移除了这里的过滤条件，直接返回所有结果
        # =====================================================================
        return results
    except Exception as e:
        safe_print(f"错误: 加载标注文件 {annotation_path} 失败: {e}"); return None

def load_raw_data_mapping(raw_dataset_file):
    raw_path = os.path.join(script_dir, raw_dataset_file)
    mapping = {}
    try:
        with open(raw_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line.strip())
                pid = obj.get('id', obj.get('pid'))
                if pid:
                    mapping[pid] = obj
        safe_print(f"成功加载原始数据映射，共 {len(mapping)} 条记录")
    except Exception as e:
        safe_print(f"加载原始数据失败: {e}")
    return mapping

def convert_annotation_to_input(annotation_sample, raw_mapping):
    pid = annotation_sample.get('pid')
    raw_sample = raw_mapping.get(pid, {})
    
    dialogue = raw_sample.get('dialogue', [])
    patient_say = annotation_sample.get('patient_say', '')
    history = raw_sample.get('history', [])
    full_dialogue = []
    if isinstance(dialogue, list):
        full_dialogue = [turn for turn in dialogue if isinstance(turn, str) and turn.strip()]
    
    if full_dialogue:
        history = full_dialogue
    elif not history and isinstance(dialogue, list) and dialogue:
        patient_turns = [i for i, turn in enumerate(dialogue) if isinstance(turn, str) and turn.startswith("患者：")]
        if patient_turns:
            last_turn_idx = patient_turns[-1]
            history = dialogue[:last_turn_idx]
        else:
            history = dialogue[:-1]
            
    return {
        'patient_say': patient_say, 
        'history': history, 
        'full_dialogue': full_dialogue,
        'report': raw_sample.get('report', {}),
        'medical_orders': raw_sample.get('medical_orders', []), 
        'department': annotation_sample.get('department', '心内科')
    }

def evaluate_single_sample(annotation_sample, sample_index, total_count, model_name, raw_mapping):
    thread_storage.captured_prompts = []
    thread_storage.captured_outputs = []
    thread_storage.captured_raw_responses = []
    safe_print(f"[{model_name}] 评估样本 {sample_index + 1}/{total_count}")
    ground_truth = annotation_sample.get('selected_tree')
    if not ground_truth:
        return {"sample_index": sample_index, "status": "error", "error_message": "No ground truth"}
    
    input_data = convert_annotation_to_input(annotation_sample, raw_mapping)
    patient_say, history, department = input_data['patient_say'], input_data.get('history', []), input_data.get('department', '心内科')
    request_storage.department = department
    
    start_time = time.time()
    try:
        # 获取科室配置树
        all_trees = department_manager.get_all_trees_for_department(department)
            
        predicted_tree = select_best_matching_tree(
            patient_say=patient_say, history=history,
            tree_category=all_trees,
            atomic_conditions=department_manager.get_department_config(department).get("atomic_conditions", {}),
            report_info=process_medical_archive_info(input_data)
        )
        end_time = time.time()
        return {"sample_index": sample_index, "patient_say": patient_say, "full_dialogue": input_data.get('full_dialogue', []), "department": department,
                "ground_truth": ground_truth, "predicted_tree": predicted_tree, "is_correct": (predicted_tree == ground_truth),
                "processing_time": f"{end_time - start_time:.3f}s", "status": "success",
                "debug_info": {"captured_prompts": thread_storage.captured_prompts, "captured_outputs": thread_storage.captured_outputs, "captured_raw_responses": thread_storage.captured_raw_responses}}
    except Exception as e:
        end_time = time.time()
        return {"sample_index": sample_index, "patient_say": patient_say, "full_dialogue": input_data.get('full_dialogue', []), "status": "error", "error_message": str(e),
                "processing_time": f"{end_time - start_time:.3f}s",
                "debug_info": {"captured_prompts": thread_storage.captured_prompts, "captured_outputs": thread_storage.captured_outputs, "captured_raw_responses": thread_storage.captured_raw_responses}}

def calculate_accuracy_metrics(results):
    successful_samples = [r for r in results if r['status'] == 'success']
    correct_samples = [r for r in successful_samples if r['is_correct']]
    accuracy = (len(correct_samples) / len(successful_samples) * 100) if successful_samples else 0
    processing_times = [float(r['processing_time'].rstrip('s')) for r in successful_samples if 's' in r.get('processing_time', '')]
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    return {'total_samples': len(results), 'successful_samples': len(successful_samples), 'correct_samples': len(correct_samples),
            'accuracy': accuracy, 'average_processing_time': avg_time}

def write_raw_responses_log(model_name, dataset_name, round_results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"raw_llm_responses_{model_name.replace('/', '_')}_{dataset_name}_{timestamp}.jsonl"
    path = os.path.join(script_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        for round_block in round_results:
            round_number = round_block.get("metadata", {}).get("round_number")
            for r in round_block.get("results", []):
                record = {
                    "round_number": round_number,
                    "sample_index": r.get("sample_index"),
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "status": r.get("status"),
                    "raw_responses": r.get("debug_info", {}).get("captured_raw_responses", [])
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    safe_print(f"[{model_name}] 原始返回已保存: {path}")

def run_evaluation_for_model(model_config, dataset_name, annotation_samples, raw_mapping):
    model_name = model_config['name']
    safe_print("\n" + "="*80 + f"\n开始评估模型: {model_name}\n" + "="*80)

    try:
        # 1. 创建真实的 OpenAI 客户端
        real_client = OpenAI(
            base_url=model_config['url'],
            api_key=model_config['key'],
            timeout=REQUEST_TIMEOUT_SEC,
            max_retries=1
        )
        
        model_name_normalized = model_name.lower()
        if "qwen3" in model_name_normalized or model_name_normalized.startswith("qwen/"):
            safe_print(f"[{model_name}] 检测到 Qwen3 模型，应用 'enable_thinking: True' 包装器。")
            local_client = Qwen3ClientWrapper(real_client)
        else:
            local_client = RawCaptureClientWrapper(real_client)

        round_results = []
        with patch('new_app.client_7b', local_client), \
             patch('new_app.MODEL_NAME_7B', model_name), \
             patch('new_app.get_template_LLM_res', get_template_LLM_res_wrapper):
            
            for round_num in range(1, NUM_ROUNDS + 1):
                safe_print(f"\n[{model_name}] 开始第 {round_num}/{NUM_ROUNDS} 轮测试...")
                samples_to_test = annotation_samples[:TEST_SUBSET_SIZE] if TEST_SUBSET_SIZE else annotation_samples
                total_samples = len(samples_to_test)
                results = []

                if model_name == "Llama-3.1-8B-Instruct":
                    safe_print(f"[{model_name}] 以串行方式执行样本评估...")
                    results = [evaluate_single_sample(s, i, total_samples, model_name, raw_mapping) for i, s in enumerate(samples_to_test)]
                else:
                    safe_print(f"[{model_name}] 以并行方式执行样本评估 (并行度: {PARALLEL_WORKERS})...")
                    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                        futures = [executor.submit(evaluate_single_sample, s, i, total_samples, model_name, raw_mapping) for i, s in enumerate(samples_to_test)]
                        results = [future.result() for future in concurrent.futures.as_completed(futures)]
                
                metrics = calculate_accuracy_metrics(results)
                safe_print(f"[{model_name}] 第 {round_num} 轮完成. 准确率: {metrics['accuracy']:.2f}%")
                round_results.append({
                    "metadata": {"round_number": round_num, "metrics": metrics, "timestamp": datetime.now().isoformat()},
                    "results": sorted(results, key=lambda x: x['sample_index'])
                })

        if round_results:
            average_filename = f"evaluation_results_{model_name.replace('/', '_')}_{dataset_name}.json"
            with open(os.path.join(script_dir, average_filename), 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        "metadata": {
                            "dataset_name": dataset_name,
                            "model_config": model_config
                        },
                        "round_results": round_results
                    },
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            write_raw_responses_log(model_name, dataset_name, round_results)
            safe_print(f"[{model_name}] ✓✓✓ 评估成功！结果已保存到: {average_filename}")
            return {"dataset_name": dataset_name, "model_name": model_name, "output_file": average_filename, "status": "success"}
    
    except Exception as e:
        safe_print(f"[{model_name}] ✗✗✗ 模型评估过程中发生严重错误: {e}")
        with print_lock:
            import traceback
            traceback.print_exc()
        return {"dataset_name": dataset_name, "model_name": model_name, "status": "error", "error": str(e)}

    return {"dataset_name": dataset_name, "model_name": model_name, "status": "error", "error": "未知错误"}

def main():
    safe_print("="*80 + "\n医疗对话树选择模型 - 混合模式评估脚本\n" + "="*80)
    
    safe_print("\n正在加载共享资源...")
    try:
        department_manager.load_all_departments()
        safe_print("✓ 共享资源加载完成")
    except Exception as e:
        safe_print(f"✗ 资源加载失败: {e}"); return

    start_time = time.time()
    all_results = []
    for dataset_job in DATASET_JOBS:
        dataset_name = dataset_job["name"]
        annotation_file = dataset_job["annotation_file"]
        raw_dataset_file = dataset_job["raw_dataset_file"]

        safe_print("\n" + "-"*80)
        safe_print(f"开始处理数据集: {dataset_name}")
        safe_print(f"标注文件: {annotation_file}")
        safe_print(f"原始数据文件: {raw_dataset_file}")
        safe_print("-"*80)

        annotation_samples = load_annotation_data(annotation_file)
        raw_mapping = load_raw_data_mapping(raw_dataset_file)
        if not annotation_samples:
            safe_print(f"✗ 数据集 {dataset_name} 的标注数据加载失败，跳过")
            continue

        for model_config in MODELS_TO_TEST:
            result = run_evaluation_for_model(model_config, dataset_name, annotation_samples, raw_mapping)
            all_results.append(result)

    end_time = time.time()
    summary_file = os.path.join(script_dir, "evaluation_results_all_datasets_summary.json")
    try:
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "dataset_jobs": DATASET_JOBS,
                    "model_jobs": MODELS_TO_TEST,
                    "results": all_results
                },
                f,
                ensure_ascii=False,
                indent=2
            )
        safe_print(f"汇总结果已保存: {summary_file}")
    except Exception as e:
        safe_print(f"保存汇总结果失败: {e}")

    safe_print("\n" + "="*80 + "\n所有模型的评估任务已全部完成！\n" + f"总耗时: {end_time - start_time:.2f} 秒\n" + "="*80)

if __name__ == "__main__":
    main()
