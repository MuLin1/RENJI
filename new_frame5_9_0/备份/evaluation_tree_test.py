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
        "name": "Llama-3.1-8B-Instruct",
        "url": "http://113.59.64.94:8088/v1",
        "key": "sk-xx"
    },
    {
        "name": "qwen2.5-7b-instruct",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key": "sk-2d81b01e2b4c4fd3b9f164afeb95a11d"
    },
    {
        "name": "qwen2.5-3b-instruct",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key": "sk-2d81b01e2b4c4fd3b9f164afeb95a11d"
    },
    {
        "name": "qwen3-8b",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key": "sk-2d81b01e2b4c4fd3b9f164afeb95a11d"
    },
    {
        "name": "qwen3-4b",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key": "sk-2d81b01e2b4c4fd3b9f164afeb95a11d"
    },
    {
        "name": "qwen3-0.6b",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key": "sk-2d81b01e2b4c4fd3b9f164afeb95a11d"
    }
]

ANNOTATION_FILE = "xnk_filtered_raw.jsonl"
TEST_SUBSET_SIZE = None
SKIP_ERRORS = True
NUM_ROUNDS = 1
PARALLEL_WORKERS = 5

# =============================================================================
# OpenAI 客户端包装器 (用于注入 Qwen3 特定参数)
# =============================================================================

class Qwen3CompletionsWrapper:
    """精准包装 'completions' 对象，拦截 'create' 调用"""
    def __init__(self, real_completions):
        self._real_completions = real_completions

    def create(self, *args, **kwargs):
        """在调用真正的 create 方法前，注入 extra_body 参数"""
        if 'extra_body' not in kwargs:
            kwargs['extra_body'] = {}
        kwargs['extra_body']["enable_thinking"] = False
        return self._real_completions.create(*args, **kwargs)

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

# =============================================================================
# 并行化与系统集成
# =============================================================================

print_lock = threading.Lock()
def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

script_dir = os.path.dirname(os.path.abspath(__file__))
paper_root = os.path.dirname(script_dir)
project_root = os.path.dirname(paper_root)
new_frame_path = os.path.join(project_root, 'paper_version')
if new_frame_path not in sys.path:
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

def extract_patient_say(sample):
    patient_say = sample.get('patient_say', '')
    dialogue = sample.get('dialogue', [])
    if patient_say:
        return patient_say
    if isinstance(dialogue, list):
        for turn in dialogue:
            if isinstance(turn, str) and turn.startswith("患者："):
                return turn.replace("患者：", "", 1).strip()
        if dialogue and isinstance(dialogue[-1], str):
            return dialogue[-1]
    return ""

def get_request_logger():
    import logging
    return logging.getLogger(__name__)

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

def load_annotation_data():
    annotation_path = os.path.join(script_dir, ANNOTATION_FILE)
    try:
        if annotation_path.lower().endswith(".jsonl"):
            results = []
            with open(annotation_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    sample = json.loads(line)
                    results.append({
                        "patient_say": extract_patient_say(sample),
                        "history": sample.get("dialogue", []),
                        "department": sample.get("department", "心内科"),
                        "selected_tree": sample.get("selected_tree"),
                        "status": "success"
                    })
            safe_print(f"成功加载JSONL数据，共 {len(results)} 个样本")
            return results

        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        results = data.get('results', [])
        safe_print(f"成功加载标注数据，共 {len(results)} 个样本")
        if SKIP_ERRORS:
            valid_results = [r for r in results if r.get('status') == 'success' and r.get('selected_tree')]
            safe_print(f"过滤后有效样本: {len(valid_results)} 个")
            return valid_results
        return results
    except Exception as e:
        safe_print(f"错误: 加载标注文件 {annotation_path} 失败: {e}"); return None

def convert_annotation_to_input(annotation_sample):
    return {'patient_say': annotation_sample.get('patient_say', ''), 'history': annotation_sample.get('history', []), 'report': {},
            'medical_orders': [], 'department': annotation_sample.get('department', '心内科')}

def evaluate_single_sample(annotation_sample, sample_index, total_count, model_name):
    thread_storage.captured_prompts = []
    thread_storage.captured_outputs = []
    safe_print(f"[{model_name}] 评估样本 {sample_index + 1}/{total_count}")
    ground_truth = annotation_sample.get('selected_tree')
    
    input_data = convert_annotation_to_input(annotation_sample)
    patient_say, history, department = input_data['patient_say'], input_data.get('history', []), input_data.get('department', '心内科')
    request_storage.department = department
    
    start_time = time.time()
    try:
        predicted_tree = select_best_matching_tree(
            patient_say=patient_say, history=history,
            tree_category=department_manager.get_all_trees_for_department(department),
            atomic_conditions=department_manager.get_department_config(department).get("atomic_conditions", {}),
            report_info=process_medical_archive_info(input_data)
        )
        end_time = time.time()
        result_status = "success" if ground_truth else "success_no_ground_truth"
        return {"sample_index": sample_index, "patient_say": patient_say, "department": department,
                "ground_truth": ground_truth, "predicted_tree": predicted_tree, "is_correct": (predicted_tree == ground_truth) if ground_truth else None,
                "processing_time": f"{end_time - start_time:.3f}s", "status": result_status,
                "debug_info": {"captured_prompts": thread_storage.captured_prompts, "captured_outputs": thread_storage.captured_outputs}}
    except Exception as e:
        end_time = time.time()
        return {"sample_index": sample_index, "patient_say": patient_say, "status": "error", "error_message": str(e),
                "processing_time": f"{end_time - start_time:.3f}s",
                "debug_info": {"captured_prompts": thread_storage.captured_prompts, "captured_outputs": thread_storage.captured_outputs}}

def calculate_accuracy_metrics(results):
    successful_samples = [r for r in results if r['status'] in ('success', 'success_no_ground_truth')]
    scored_samples = [r for r in successful_samples if r.get('is_correct') is not None]
    correct_samples = [r for r in scored_samples if r['is_correct']]
    accuracy = (len(correct_samples) / len(scored_samples) * 100) if scored_samples else 0
    processing_times = [float(r['processing_time'].rstrip('s')) for r in successful_samples if 's' in r.get('processing_time', '')]
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    return {'total_samples': len(results), 'successful_samples': len(successful_samples), 'scored_samples': len(scored_samples),
            'correct_samples': len(correct_samples), 'accuracy': accuracy, 'average_processing_time': avg_time}

def run_evaluation_for_model(model_config, annotation_samples):
    model_name = model_config['name']
    safe_print("\n" + "="*80 + f"\n开始评估模型: {model_name}\n" + "="*80)

    try:
        # 1. 创建真实的 OpenAI 客户端
        real_client = OpenAI(base_url=model_config['url'], api_key=model_config['key'])
        
        # 2. 如果是 qwen3 模型，则用我们的包装器将其“包”起来
        if model_name.startswith("qwen3"):
            safe_print(f"[{model_name}] 检测到 Qwen3 模型，应用 'enable_thinking: False' 包装器。")
            local_client = Qwen3ClientWrapper(real_client)
        else:
            local_client = real_client

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
                    results = [evaluate_single_sample(s, i, total_samples, model_name) for i, s in enumerate(samples_to_test)]
                else:
                    safe_print(f"[{model_name}] 以并行方式执行样本评估 (并行度: {PARALLEL_WORKERS})...")
                    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                        futures = [executor.submit(evaluate_single_sample, s, i, total_samples, model_name) for i, s in enumerate(samples_to_test)]
                        results = [future.result() for future in concurrent.futures.as_completed(futures)]
                
                metrics = calculate_accuracy_metrics(results)
                safe_print(f"[{model_name}] 第 {round_num} 轮完成. 准确率: {metrics['accuracy']:.2f}%")
                round_results.append({
                    "metadata": {"round_number": round_num, "metrics": metrics, "timestamp": datetime.now().isoformat()},
                    "results": sorted(results, key=lambda x: x['sample_index'])
                })

        if round_results:
            average_filename = f"evaluation_results_{model_name.replace('/', '_')}.json"
            with open(os.path.join(script_dir, average_filename), 'w', encoding='utf-8') as f:
                json.dump({"metadata": {"model_config": model_config}, "round_results": round_results}, f, ensure_ascii=False, indent=2)
            safe_print(f"[{model_name}] ✓✓✓ 评估成功！结果已保存到: {average_filename}")
    
    except Exception as e:
        safe_print(f"[{model_name}] ✗✗✗ 模型评估过程中发生严重错误: {e}")
        with print_lock:
            import traceback
            traceback.print_exc()

def main():
    safe_print("="*80 + "\n医疗对话树选择模型 - 混合模式评估脚本\n" + "="*80)
    
    safe_print("\n正在加载共享资源...")
    try:
        department_manager.load_all_departments()
        annotation_samples = load_annotation_data()
        if not annotation_samples:
            safe_print("✗ 标注数据加载失败，程序终止"); return
        safe_print("✓ 共享资源加载完成")
    except Exception as e:
        safe_print(f"✗ 资源加载失败: {e}"); return

    start_time = time.time()
    for model_config in MODELS_TO_TEST:
        run_evaluation_for_model(model_config, annotation_samples)

    end_time = time.time()
    safe_print("\n" + "="*80 + "\n所有模型的评估任务已全部完成！\n" + f"总耗时: {end_time - start_time:.2f} 秒\n" + "="*80)

if __name__ == "__main__":
    main()
