import json
import os
import sys
import time
import threading
from datetime import datetime
from typing import Dict, List

# =============================================================================
# 可配置区域 - 请根据需要修改以下参数
# =============================================================================

# 模型API配置
API_URL = "http://localhost:8080/v1"    # 【重要】请确保这里是您要测试的模型的API地址
#API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "sk-xx"                       # API密钥
MODEL_NAME = "Qwen3-0.6B"     # 【重要】请确保这里是您要测试的模型的名称

# 文件路径配置
LABEL_TO_ID_FILE = "label-to-id.json"
TEST_DATA_FILE = "xnk_filtered_raw.jsonl"
EVALUATION_OUTPUT = f"evaluation_results_condition_node_{MODEL_NAME}.json"
PAPER_VERSION_DIR = "paper_version"

# 测试参数配置
TEST_SUBSET_SIZE = None     # 测试节点数量限制，None=全部，数字=限制数量（如：5表示只测试前5个节点）
NUM_ROUNDS = 5              # 每个节点的测试轮数
SKIP_ERRORS = True          # 是否跳过格式错误的样本
REQUEST_INTERVAL = 0.1      # 每次API请求后的间隔时间（秒），避免请求过于频繁

# =============================================================================
# 系统集成和初始化
# =============================================================================

# 将 paper_version 目录添加到 sys.path
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    paper_version_path = os.path.join(project_root, PAPER_VERSION_DIR)
    if paper_version_path not in sys.path:
        sys.path.insert(0, paper_version_path)

    # 动态导入模块
    print("正在导入模块...")
    print(f"paper_version 路径: {paper_version_path}")
    if not os.path.exists(paper_version_path):
        raise FileNotFoundError(f"找不到 '{PAPER_VERSION_DIR}' 目录: {paper_version_path}")

    import new_app
    from new_app import (
        judge_node_process,
        get_template_LLM_res as original_get_template_LLM_res
    )
    from openai import OpenAI
    print("✓ 成功导入 paper_version 模块")

except (ImportError, FileNotFoundError) as e:
    print(f"✗ 无法从 '{PAPER_VERSION_DIR}' 导入模块。请确保 '{PAPER_VERSION_DIR}/new_app.py' 存在且路径正确。")
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
    
    if isinstance(history, list):
        history_text = "\n".join(history) if history else "无"
    else:
        history_text = str(history) if history else "无"
    
    template_params = {'history': history_text, 'patient_say': patient_say, **kwargs}
    
    try:
        formatted_prompt = template.format(**template_params)
        messages = [
            {"role": "system", "content": "你是一个有用的医生助手，使用纯文本输出，不要用markdown格式。"},
            {"role": "user", "content": formatted_prompt}
        ]
        captured_prompts.append(json.dumps(messages, ensure_ascii=False, indent=2))
    except KeyError as e:
        captured_prompts.append(f"PROMPT_FORMAT_ERROR: Missing key {e}")

    try:
        result = original_get_template_LLM_res(template, patient_say, history, **kwargs)
        captured_model_outputs.append(str(result))
        return result
    except Exception as e:
        error_message = f"MODEL_CALL_ERROR: {str(e)}"
        captured_model_outputs.append(error_message)
        raise

def initialize_model_client():
    """初始化并应用测试所需的模型客户端"""
    try:
        print("\n2. 初始化模型客户端...")
        client = OpenAI(base_url=API_URL, api_key=API_KEY)
        new_app.client_7b = client
        new_app.MODEL_NAME_7B = MODEL_NAME
        print(f"✓ 已将 new_app 的模型客户端重定向至 -> URL: {API_URL}, Model: {MODEL_NAME}")
        return True
    except Exception as e:
        print(f"✗ 初始化模型客户端失败: {e}")
        return False

def resolve_data_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(script_dir, path)

def load_label_to_id_data():
    """加载标注数据（label-to-id.json）"""
    try:
        label_path = resolve_data_path(LABEL_TO_ID_FILE)
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        nodes = data.get('nodes', {})
        print(f"✓ 成功加载标注数据，包含 {len(nodes)} 个节点")
        return nodes
    except FileNotFoundError:
        print(f"✗ 找不到标注文件: {resolve_data_path(LABEL_TO_ID_FILE)}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"✗ 标注文件格式错误: {e}")
        sys.exit(1)

def load_test_data() -> Dict[str, Dict]:
    """加载测试数据（xnk_filtered_raw.jsonl）并以id作为key存入字典"""
    try:
        test_data = {}
        test_data_path = resolve_data_path(TEST_DATA_FILE)
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    # 确保 'id' 字段存在
                    if 'id' in data:
                        test_data[data['id']] = data
                    elif 'pid' in data: # 兼容pid
                        test_data[data['pid']] = data
                    else:
                         if not SKIP_ERRORS:
                            print(f"✗ 第{line_num}行数据缺少 'id' 或 'pid' 字段")
                            sys.exit(1)
                         print(f"⚠ 跳过第{line_num}行（缺少id/pid）")

                except json.JSONDecodeError:
                    if not SKIP_ERRORS:
                        print(f"✗ 第{line_num}行数据格式错误")
                        sys.exit(1)
                    print(f"⚠ 跳过第{line_num}行（格式错误）")
        print(f"✓ 成功加载测试数据，包含 {len(test_data)} 个样本")
        return test_data
    except FileNotFoundError:
        print(f"✗ 找不到测试数据文件: {resolve_data_path(TEST_DATA_FILE)}")
        sys.exit(1)

def convert_dialogue_to_history(dialogue: List[str]) -> List[str]:
    """确保对话是列表格式"""
    return dialogue if isinstance(dialogue, list) else []

def create_mock_judge_node(content: str, labels: List[str]) -> Dict:
    """创建模拟的判断节点对象以传入 judge_node_process"""
    return {"content": content, "label": labels, "type": "条件节点"}

def test_single_node(node_id: str, node_data: Dict, test_data_dict: Dict, round_num: int) -> Dict:
    """测试单个条件节点的所有PID"""
    global captured_prompts, captured_model_outputs
    
    labels = node_data.get('label', [])
    pids = node_data.get('pid', [])
    
    if len(pids) != len(labels) + 1:
        print(f"    ✗ 错误：节点 {node_id} 的PID数量({len(pids)})不等于label数量({len(labels)})+1")
        return {
            "node_id": node_id, "status": "error", "accuracy": 0.0,
            "error_message": f"PID count mismatch: got {len(pids)}, expected {len(labels) + 1}"
        }

    mock_judge_node = create_mock_judge_node(node_data['content'], labels)
    
    total_tests = 0
    correct_tests = 0.0
    test_details = []

    for idx, pid in enumerate(pids):
        captured_prompts, captured_model_outputs = [], []
        if pid not in test_data_dict:
            print(f"    ⚠ 跳过PID {pid[:8]}...：在测试数据中未找到")
            continue

        dialogue_data = test_data_dict[pid].get('dialogue', [])
        history = convert_dialogue_to_history(dialogue_data)
        
        if idx < len(labels):
            expected_label = labels[idx]
            expected_result, expected_choice = "能", str(idx + 1)
        else:
            expected_label = "不确定"
            expected_result, expected_choice = "不能", None
        
        try:
            actual_result, actual_choice = judge_node_process(
                dialogue="\n".join(history), judge_node=mock_judge_node, flag=False, args=None,
                patient_say="", history=history, tree_name="test_tree", node_id=node_id, report_info=""
            )
            
            score = 0.0
            if expected_result == "能":
                if actual_result == "能" and actual_choice == expected_choice:
                    score = 1.0  # 完全正确
                elif actual_result != "能":
                    score = 0.5  # 本应判断为"能"，但模型判断为"不能"，算半对
                # score 保持 0.0 表示模型判断为"能"但选项错误
            else:  # 期望"不能"
                if actual_result != "能":
                    score = 1.0  # 正确判断为"不能"
                # score 保持 0.0 表示模型在应为"不能"时判断为"能"

            correct_tests += score
            
            if score == 1.0:
                status = "✓ 正确 (1.0)"
            elif score == 0.5:
                status = "½ 半对 (0.5)"
            else:
                status = "✗ 错误 (0.0)"
            
            print(f"    - PID {pid[:8]}... -> 期望:[{expected_label}] -> 实际:[{actual_result}, {actual_choice}] -> {status}")
            
            test_details.append({
                "pid": pid, "expected_label": expected_label, "expected_result": expected_result,
                "expected_choice": expected_choice, "actual_result": actual_result,
                "actual_choice": actual_choice, "score": score,
                "dialogue_length": len(history),
                "debug_info": {"prompts": captured_prompts, "outputs": captured_model_outputs}
            })
            total_tests += 1
            time.sleep(REQUEST_INTERVAL)

        except Exception as e:
            print(f"    - PID {pid[:8]}... -> 异常: {e}")
            test_details.append({
                "pid": pid, "expected_label": expected_label, "error": str(e), "score": 0.0
            })
            total_tests += 1
    
    accuracy = correct_tests / total_tests if total_tests > 0 else 0.0
    print(f"    准确率: {correct_tests}/{total_tests} = {accuracy:.2%}")
    
    return {
        "node_id": node_id, "content": node_data['content'], "labels": labels,
        "status": "completed", "total_tests": total_tests, "correct_tests": correct_tests,
        "accuracy": accuracy, "test_details": test_details
    }

def run_evaluation():
    """运行完整的评估流程"""
    print("=" * 60)
    print(f"条件节点判断过程测试 ({MODEL_NAME})")
    print("=" * 60)
    
    print("\n1. 加载数据...")
    label_data = load_label_to_id_data()
    test_data_dict = load_test_data()
    
    if not initialize_model_client():
        sys.exit(1)

    # 兼容Qwen3系列模型：禁用thinking功能
    qwen3_models = ["Qwen3-8B", "Qwen3-4B"]
    if MODEL_NAME in qwen3_models:
        if hasattr(new_app, 'client_7b') and new_app.client_7b:
            original_create = new_app.client_7b.chat.completions.create
            def create_wrapper(*args, **kwargs):
                """包装函数，添加extra_body参数以禁用Qwen3系列模型的thinking功能"""
                if 'extra_body' not in kwargs:
                    kwargs['extra_body'] = {}
                kwargs['extra_body']['chat_template_kwargs'] = {"enable_thinking": False}
                return original_create(*args, **kwargs)
            new_app.client_7b.chat.completions.create = create_wrapper
            print(f"✓ 已为Qwen3系列模型 '{MODEL_NAME}' 禁用thinking功能")
        else:
            print(f"⚠ 警告: 模型 '{MODEL_NAME}' 需要禁用thinking功能，但client_7b不可用")
        
    print("\n3. 设置 new_app 模块...")
    # 使用 patch 来确保在测试期间替换掉原始函数
    original_func = new_app.get_template_LLM_res
    new_app.get_template_LLM_res = get_template_LLM_res_wrapper
    # 模拟日志记录器
    new_app.get_request_logger = get_request_logger
    print("✓ 已成功应用函数包装器和日志模拟器")

    test_nodes = dict(list(label_data.items())[:TEST_SUBSET_SIZE]) if TEST_SUBSET_SIZE else label_data
    print(f"\n4. 开始测试 {len(test_nodes)} 个条件节点 (每节点 {NUM_ROUNDS} 轮)...")
    
    all_results = []
    start_time = time.time()
    
    for node_index, (node_id, node_data) in enumerate(test_nodes.items(), 1):
        print(f"\n--- 节点 {node_index}/{len(test_nodes)}: {node_data.get('content', 'N/A')} ---")
        node_round_results = [test_single_node(node_id, node_data, test_data_dict, r) for r in range(1, NUM_ROUNDS + 1)]
        
        avg_accuracy = sum(r.get('accuracy', 0) for r in node_round_results) / NUM_ROUNDS if NUM_ROUNDS > 0 else 0
        print(f"  节点 {node_id} 平均准确率: {avg_accuracy:.2%}")
        
        all_results.append({
            "node_id": node_id, "node_content": node_data['content'],
            "node_labels": node_data.get('label', []), "rounds": node_round_results,
            "average_accuracy": avg_accuracy
        })
    
    total_time = time.time() - start_time
    
    print("\n5. 生成评估报告...")
    overall_accuracy = sum(r['average_accuracy'] for r in all_results) / len(all_results) if all_results else 0
    
    final_report = {
        "evaluation_info": {
            "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": MODEL_NAME, "total_nodes_tested": len(test_nodes),
            "rounds_per_node": NUM_ROUNDS, "overall_accuracy": overall_accuracy,
            "total_time_seconds": total_time,
            "config": {"api_url": API_URL, "test_subset_size": TEST_SUBSET_SIZE}
        },
        "node_results": all_results
    }
    
    output_path = resolve_data_path(EVALUATION_OUTPUT)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=4)
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print(f"总体准确率: {overall_accuracy:.2%}")
    print(f"总用时: {total_time:.1f}秒")
    print(f"结果已保存到: {output_path}")
    print("=" * 60)

    # 恢复原始函数
    new_app.get_template_LLM_res = original_func

if __name__ == "__main__":
    label_path = resolve_data_path(LABEL_TO_ID_FILE)
    test_path = resolve_data_path(TEST_DATA_FILE)
    if not all(os.path.exists(f) for f in [label_path, test_path]):
        print("✗ 必需的测试文件不存在，请检查路径配置。")
        if not os.path.exists(label_path): print(f"  - 找不到: {label_path}")
        if not os.path.exists(test_path): print(f"  - 找不到: {test_path}")
        sys.exit(1)
        
    try:
        run_evaluation()
    except KeyboardInterrupt:
        print("\n用户中断了评估过程。")
    except Exception as e:
        print(f"\n评估过程中发生未捕获的错误: {e}")
        import traceback
        traceback.print_exc()
