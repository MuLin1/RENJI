import json
import os
import sys
import time
from datetime import datetime
from unittest.mock import patch
import threading

# =============================================================================
# 可配置区域 - 请根据需要修改以下参数
# =============================================================================

# 7B模型API配置
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 请修改为您的模型API URL
API_KEY = "sk-b2c19e80cf174cdd9262c51a9d70a859"                # 请修改为您的API Key
MODEL_NAME = "deepseek-v3.1"  # 请修改为您的模型名称

# 输入输出文件配置
DATASET_JOBS = [
    {"name": "highrisk", "input_file": "xnk_dataset_highrisk.jsonl", "output_file": "result_highrisk.json"},
    {"name": "chronic", "input_file": "xnk_dataset_chronic.jsonl", "output_file": "result_chronic.json"},
]

# =============================================================================
# 系统集成和初始化
# =============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
module_candidates = [
    os.path.join(project_root, 'new_frame'),
    os.path.join(project_root, 'paper_version')
]
new_frame_path = next((path for path in module_candidates if os.path.exists(path)), None)
if new_frame_path and new_frame_path not in sys.path:
    sys.path.insert(0, new_frame_path)

# 导入必要的模块
print("正在导入模块...")
print(f"new_frame路径: {new_frame_path}")
print(f"路径是否存在: {bool(new_frame_path and os.path.exists(new_frame_path))}")

try:
    import new_app
    from new_app import (
        select_best_matching_tree,
        process_medical_archive_info,
        get_template_LLM_res as original_get_template_LLM_res,
        department_manager
    )
    from openai import OpenAI
    print("✓ 成功导入 new_frame 模块")
except ImportError as e:
    print(f"✗ 无法从 'new_frame' 导入模块。请确保 'new_frame/new_app.py' 存在。")
    print(f"错误详情: {e}")
    print(f"当前 sys.path: {sys.path}")
    sys.exit(1)

# 全局变量用于捕获调试信息
captured_prompts = []
captured_model_outputs = []

# 线程本地存储，用于兼容原代码中的日志记录
request_storage = threading.local()

def get_request_logger():
    """模拟logger以避免对原日志代码的依赖"""
    import logging
    return logging.getLogger(__name__)

# =============================================================================
# 核心功能函数
# =============================================================================

def get_template_LLM_res_wrapper(template, patient_say, history, **kwargs):
    """
    包装函数，用于捕获prompt和模型输出进行调试
    """
    global captured_prompts, captured_model_outputs
    
    # 格式化历史记录
    if isinstance(history, list):
        history_text = "\n".join(history) if history else "无"
    else:
        history_text = str(history) if history else "无"
    
    template_params = {'history': history_text, 'patient_say': patient_say, **kwargs}
    
    try:
        # 格式化prompt并记录
        formatted_prompt = template.format(**template_params)
        messages = [
            {"role": "system", "content": "你是一个有用的医生助手，使用纯文本输出，不要用markdown格式。"},
            {"role": "user", "content": formatted_prompt}
        ]
        captured_prompts.append(json.dumps(messages, ensure_ascii=False, indent=2))
    except KeyError as e:
        print(f"格式化prompt时出错，缺少键: {e}")
        captured_prompts.append(f"PROMPT_FORMAT_ERROR: Missing key {e}")

    # 调用原始函数并捕获输出
    raw_output = original_get_template_LLM_res(template, patient_say, history, **kwargs)
    captured_model_outputs.append(raw_output)
    
    return raw_output

def initialize_model_client():
    """初始化模型客户端"""
    try:
        print(f"正在初始化模型客户端...")
        print(f"API_URL: {API_URL}")
        print(f"MODEL_NAME: {MODEL_NAME}")
        
        client = OpenAI(base_url=API_URL, api_key=API_KEY)
        new_app.client_7b = client
        new_app.MODEL_NAME_7B = MODEL_NAME
        print(f"✓ 已初始化模型客户端 - URL: {API_URL}, Model: {MODEL_NAME}")
        return True
    except Exception as e:
        print(f"✗ 初始化模型客户端失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_input_data(input_file):
    """加载输入数据"""
    input_path = os.path.join(script_dir, input_file)
    try:
        if input_path.endswith(".jsonl"):
            data = []
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data.append(json.loads(line))
        else:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("samples"), list):
                extracted = []
                for item in data["samples"]:
                    if isinstance(item, dict) and isinstance(item.get("raw_entry"), dict):
                        extracted.append(item["raw_entry"])
                if extracted:
                    data = extracted
            elif isinstance(data, dict) and isinstance(data.get("results"), list):
                data = data["results"]
        print(f"成功加载输入数据，共 {len(data)} 个样本")
        return data
    except FileNotFoundError:
        print(f"错误: 输入文件 {input_path} 未找到")
        return None
    except json.JSONDecodeError:
        print(f"错误: 输入文件 {input_path} 格式无效")
        return None

def convert_sample_format(sample):
    """
    将样本格式转换为系统需要的格式
    """
    # 构建输入数据格式
    dialogue = sample.get('dialogue', [])
    patient_say = sample.get('patient_say', '')
    history = sample.get('history', [])
    if isinstance(dialogue, list) and dialogue:
        history = dialogue
        patient_say = ""

    input_data = {
        'pid': sample.get('id', sample.get('pid')),
        'patient_say': patient_say,
        'history': history if isinstance(history, list) else [],
        'report': sample.get('report', {}),
        'medical_orders': sample.get('medical_orders', []),
        'department': sample.get('department', '心内科')
    }
    
    return input_data

def annotate_single_sample(sample, sample_index, total_count):
    """
    标注单个样本
    """
    global captured_prompts, captured_model_outputs
    
    # 重置捕获的调试信息
    captured_prompts = []
    captured_model_outputs = []
    
    print(f"--- 正在处理样本 {sample_index + 1}/{total_count} ---")
    
    # 转换格式
    input_data = convert_sample_format(sample)
    patient_say = input_data['patient_say']
    history = input_data.get('history', [])
    department = input_data.get('department', '心内科')
    
    # 设置线程存储
    request_storage.department = department
    
    print(f"患者输入: {patient_say[:100]}...")
    print(f"科室: {department}")
    
    start_time = time.time()
    
    try:
        # 获取科室配置
        all_trees = department_manager.get_all_trees_for_department(department)
        
        dept_config = department_manager.get_department_config(department)
        atomic_conditions = dept_config.get("atomic_conditions", {})
        
        # 处理就医档案信息
        report_info = process_medical_archive_info(input_data)
        
        # 调用树选择函数
        selected_tree = select_best_matching_tree(
            patient_say=patient_say,
            history=history,
            tree_category=all_trees,
            atomic_conditions=atomic_conditions,
            report_info=report_info
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"选择的树: {selected_tree}")
        print(f"处理时间: {processing_time:.3f}秒")
        
        # 构建结果
        result = {
            "sample_index": sample_index,
            "pid": input_data.get('pid'),
            "patient_say": patient_say,
            "department": department,
            "selected_tree": selected_tree,
            "processing_time": f"{processing_time:.3f}s",
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "debug_info": {
                "prompts_count": len(captured_prompts),
                "model_outputs_count": len(captured_model_outputs),
                "captured_prompts": captured_prompts,
                "captured_outputs": captured_model_outputs
            }
        }
        
        return result
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"处理样本时发生错误: {e}")
        
        result = {
            "sample_index": sample_index,
            "pid": input_data.get('pid'),
            "patient_say": patient_say,
            "department": department,
            "selected_tree": None,
            "processing_time": f"{processing_time:.3f}s",
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error_message": str(e),
            "debug_info": {
                "prompts_count": len(captured_prompts),
                "model_outputs_count": len(captured_model_outputs),
                "captured_prompts": captured_prompts,
                "captured_outputs": captured_model_outputs
            }
        }
        
        return result

def run_annotation_for_dataset(dataset_job):
    """
    运行单个数据集的标注流程
    """
    dataset_name = dataset_job["name"]
    input_file = dataset_job["input_file"]
    output_file = dataset_job["output_file"]

    print("="*60)
    print("医疗对话树选择标注脚本 v1.0")
    print("="*60)
    print(f"数据集: {dataset_name}")
    print(f"模型配置: {MODEL_NAME} @ {API_URL}")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"脚本目录: {script_dir}")
    print("="*60)
    
    # 1. 初始化模型客户端
    if not initialize_model_client():
        print("模型客户端初始化失败，程序终止")
        return
    
    # 2. 加载科室配置
    print("正在加载科室配置...")
    try:
        department_manager.load_all_departments()
        print("科室配置加载完成")
    except Exception as e:
        print(f"科室配置加载失败: {e}")
        return
    
    # 3. 加载输入数据
    samples = load_input_data(input_file)
    if samples is None:
        print("输入数据加载失败，程序终止")
        return None
    
    total_count = len(samples)
    print(f"开始处理 {total_count} 个样本...")
    
    # 4. 处理样本
    results = []
    success_count = 0
    total_time = 0
    
    # 创建包装器以禁用thinking模式
    if hasattr(new_app, 'client_7b') and new_app.client_7b:
        original_create = new_app.client_7b.chat.completions.create
        
        def create_wrapper(*args, **kwargs):
            """包装函数，添加extra_body参数"""
            kwargs['extra_body'] = {"enable_thinking": False}
            return original_create(*args, **kwargs)
    else:
        create_wrapper = None
    
    # 使用patch应用包装器
    patches = [
        patch('new_app.get_template_LLM_res', get_template_LLM_res_wrapper),
        patch('new_app.get_request_logger', get_request_logger)
    ]
    
    if create_wrapper:
        patches.append(patch('new_app.client_7b.chat.completions.create', create_wrapper))
    
    # 使用多个patch的正确方式
    from contextlib import ExitStack
    with ExitStack() as stack:
        for patch_obj in patches:
            stack.enter_context(patch_obj)
        for i, sample in enumerate(samples):
            result = annotate_single_sample(sample, i, total_count)
            results.append(result)
            
            if result['status'] == 'success':
                success_count += 1
            
            # 解析处理时间
            try:
                time_value = float(result['processing_time'].rstrip('s'))
                total_time += time_value
            except:
                pass
            
            # 定期保存中间结果（每50个样本）
            if (i + 1) % 50 == 0:
                save_intermediate_results(results, i + 1, dataset_name)
    
    # 5. 计算统计信息
    avg_time = total_time / total_count if total_count > 0 else 0
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    
    print("\n" + "="*60)
    print("标注完成统计:")
    print(f"总样本数: {total_count}")
    print(f"成功标注: {success_count}")
    print(f"成功率: {success_rate:.2f}%")
    print(f"平均处理时间: {avg_time:.3f}秒")
    print("="*60)
    
    # 6. 保存最终结果
    final_results = {
        "metadata": {
            "dataset_name": dataset_name,
            "input_file": input_file,
            "output_file": output_file,
            "total_samples": total_count,
            "successful_annotations": success_count,
            "success_rate": f"{success_rate:.2f}%",
            "average_processing_time": f"{avg_time:.3f}s",
            "model_config": {
                "url": API_URL,
                "model_name": MODEL_NAME,
                "api_key": "***hidden***"
            },
            "timestamp": datetime.now().isoformat()
        },
        "results": results
    }
    
    output_path = os.path.join(script_dir, output_file)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_path}")
        return {
            "dataset_name": dataset_name,
            "input_file": input_file,
            "output_file": output_file,
            "total_samples": total_count,
            "successful_annotations": success_count,
            "success_rate": f"{success_rate:.2f}%",
            "average_processing_time": f"{avg_time:.3f}s"
        }
    except Exception as e:
        print(f"保存结果失败: {e}")
        return {
            "dataset_name": dataset_name,
            "input_file": input_file,
            "output_file": output_file,
            "error": str(e)
        }

def save_intermediate_results(results, processed_count, dataset_name):
    """保存中间结果"""
    intermediate_file = f"annotation_results_{dataset_name}_intermediate_{processed_count}.json"
    output_path = os.path.join(script_dir, intermediate_file)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "processed_count": processed_count,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
        print(f"中间结果已保存: {intermediate_file}")
    except Exception as e:
        print(f"保存中间结果失败: {e}")

def run_annotation():
    """
    依次运行三个数据集的标注流程
    """
    all_summaries = []
    for dataset_job in DATASET_JOBS:
        summary = run_annotation_for_dataset(dataset_job)
        if summary is not None:
            all_summaries.append(summary)

    print("\n" + "="*60)
    print("全部数据集标注完成")
    print("="*60)
    for item in all_summaries:
        if "error" in item:
            print(f"{item['dataset_name']}: 失败 - {item['error']}")
        else:
            print(
                f"{item['dataset_name']}: 成功 {item['successful_annotations']}/{item['total_samples']} "
                f"(成功率 {item['success_rate']}) -> {item['output_file']}"
            )
    print("="*60)

# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    print("开始执行标注任务...")
    try:
        run_annotation()
    except KeyboardInterrupt:
        print("\n用户中断程序执行")
    except Exception as e:
        print(f"程序执行过程中发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("程序执行完毕")
