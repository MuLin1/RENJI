# 流式输出功能改造方案

本方案为问诊系统增加流式输出（SSE）支持，think 和 answer 均流式输出，同时保持原有 API 完全兼容。

---

## 改动概览

| 文件 | 改动类型 | 预估代码量 |
|------|---------|-----------|
| `new_app.py` | 新增函数 | +150 行 |
| `chat_api.py` | 新增流式端点逻辑 | +80 行 |
| `api接口文档.md` | 补充流式接口文档 | +120 行 |

**无需新增依赖**，Docker 部署不受影响。

---

## 一、`new_app.py` 改动

### 1.1 新增 `StreamTagParser` 类（约 60 行）

位置：`main_llm` 函数之前

```python
class StreamTagParser:
    """
    流式解析 <think></think> 和 <answer></answer> 标签的状态机
    
    状态流转: init -> in_think -> between -> in_answer -> done
    """
    def __init__(self):
        self.buffer = ""
        self.state = "init"  # init, in_think, between, in_answer, done
        
    def feed(self, chunk: str):
        """
        喂入新的 token，返回可以推送的事件列表
        返回格式: [("think", "内容"), ("answer", "内容"), ...]
        """
        self.buffer += chunk
        events = []
        
        while True:
            if self.state == "init":
                # 等待 <think> 标签
                if "<think>" in self.buffer:
                    self.buffer = self.buffer.split("<think>", 1)[1]
                    self.state = "in_think"
                else:
                    # 保留可能不完整的标签前缀
                    break
                    
            elif self.state == "in_think":
                if "</think>" in self.buffer:
                    think_content, self.buffer = self.buffer.split("</think>", 1)
                    if think_content:
                        events.append(("think", think_content))
                    self.state = "between"
                else:
                    # 安全输出：保留可能的 </think> 前缀
                    safe_len = len(self.buffer) - len("</think>")
                    if safe_len > 0:
                        events.append(("think", self.buffer[:safe_len]))
                        self.buffer = self.buffer[safe_len:]
                    break
                    
            elif self.state == "between":
                # 等待 <answer> 标签
                if "<answer>" in self.buffer:
                    self.buffer = self.buffer.split("<answer>", 1)[1]
                    self.state = "in_answer"
                else:
                    break
                    
            elif self.state == "in_answer":
                if "</answer>" in self.buffer:
                    answer_content, self.buffer = self.buffer.split("</answer>", 1)
                    if answer_content:
                        events.append(("answer", answer_content))
                    self.state = "done"
                else:
                    safe_len = len(self.buffer) - len("</answer>")
                    if safe_len > 0:
                        events.append(("answer", self.buffer[:safe_len]))
                        self.buffer = self.buffer[safe_len:]
                    break
                    
            elif self.state == "done":
                break
                
        return events
    
    def flush(self):
        """处理结束时，输出缓冲区剩余内容"""
        events = []
        if self.buffer.strip():
            if self.state == "in_think":
                events.append(("think", self.buffer))
            elif self.state in ("between", "in_answer"):
                events.append(("answer", self.buffer))
        self.buffer = ""
        return events
```

### 1.2 新增 `main_llm_stream` 生成器函数（约 50 行）

位置：`main_llm` 函数之后

```python
def main_llm_stream(patient_say, history, tips=None, report_info="", department="心内科"):
    """
    流式版本的 main_llm，返回生成器
    
    Yields:
        dict: {"type": "think"|"answer", "delta": "文本片段"}
              或 {"type": "done", "full_think": "...", "full_answer": "..."}
    """
    # ... 构建 messages（与 main_llm 相同的前置逻辑）
    
    # 流式调用
    stream = client_14b.chat.completions.create(
        model=MODEL_NAME_14B,
        messages=messages,
        temperature=0.0,
        top_p=1.0,
        stream=True  # 关键：启用流式
    )
    
    parser = StreamTagParser()
    full_think = ""
    full_answer = ""
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            events = parser.feed(token)
            
            for event_type, content in events:
                if event_type == "think":
                    full_think += content
                    yield {"type": "think", "delta": content}
                elif event_type == "answer":
                    full_answer += content
                    yield {"type": "answer", "delta": content}
    
    # 处理剩余缓冲
    for event_type, content in parser.flush():
        if event_type == "think":
            full_think += content
            yield {"type": "think", "delta": content}
        elif event_type == "answer":
            full_answer += content
            yield {"type": "answer", "delta": content}
    
    # 对 answer 做后处理（clean_markdown）
    full_answer = clean_markdown(full_answer.split("医生：")[-1])
    
    yield {"type": "done", "full_think": full_think, "full_answer": full_answer}
```

### 1.3 新增 `main_app_start_with_history_stream` 生成器函数（约 40 行）

位置：`main_app_start_with_history` 函数之后

```python
def main_app_start_with_history_stream(patient_say, history=None, current_tree=None, 
                                        current_node=None, node_flag=False, 
                                        report_info="", medical_orders_info="", 
                                        department="心内科"):
    """
    流式版本的主函数，生成器
    
    Yields:
        dict: 各阶段事件
            - {"stage": "preprocessing", "risk_detection": {...}}
            - {"stage": "tree", "tree": "...", "tree_node": "...", "node_flag": ...}
            - {"stage": "think", "delta": "..."}
            - {"stage": "answer", "delta": "..."}
            - {"stage": "done", "full_think": "...", "full_answer": "..."}
    """
    # ===== 阶段1：前置步骤（复用 main_app_start_with_history 的逻辑到 main_llm 调用之前）=====
    # 执行：代问诊检测、摘要pipeline、决策树选择/遍历
    # 这部分代码从 main_app_start_with_history 提取，执行到获得 tips 为止
    
    # 推送前置完成事件
    yield {
        "stage": "preprocessing",
        "risk_detection": risk_result
    }
    
    # ===== 阶段2：决策树状态 =====
    yield {
        "stage": "tree",
        "tree": tree_to_return,
        "tree_node": node_to_return,
        "node_flag": flag_to_return
    }
    
    # ===== 阶段3：流式模型生成 =====
    for event in main_llm_stream(patient_say, backend_history, tips, report_info, department):
        if event["type"] == "think":
            yield {"stage": "think", "delta": event["delta"]}
        elif event["type"] == "answer":
            yield {"stage": "answer", "delta": event["delta"]}
        elif event["type"] == "done":
            yield {
                "stage": "done",
                "full_think": event["full_think"],
                "full_answer": event["full_answer"]
            }
```

---

## 二、`chat_api.py` 改动

### 2.1 新增导入

```python
from flask import Response
from new_app import main_app_start_with_history_stream  # 新增
```

### 2.2 修改 `/api` 端点，支持 `stream` 参数

在现有 `chat()` 函数中增加分支：

```python
@app.route('/api', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        # ... 现有验证逻辑不变 ...
        
        # 新增：检查是否为流式请求
        stream_mode = data.get('stream', False)
        if isinstance(stream_mode, str):
            stream_mode = stream_mode.lower() == 'true'
        
        if stream_mode:
            # ===== 流式响应分支 =====
            return stream_response(data, patient_say, history, current_tree, 
                                   current_node, node_flag, report_info, 
                                   medical_orders_info, department, log_extra)
        else:
            # ===== 原有非流式逻辑，完全不变 =====
            result = main_app_start_with_history(...)
            # ...
```

### 2.3 新增 `stream_response` 函数

```python
def stream_response(data, patient_say, history, current_tree, current_node, 
                    node_flag, report_info, medical_orders_info, department, log_extra):
    """
    处理流式响应的函数
    """
    def generate():
        try:
            for event in main_app_start_with_history_stream(
                patient_say=patient_say,
                history=history,
                current_tree=current_tree,
                current_node=current_node,
                node_flag=node_flag,
                report_info=report_info,
                medical_orders_info=medical_orders_info,
                department=department
            ):
                # SSE 格式：event: <type>\ndata: <json>\n\n
                event_type = event.get("stage", "message")
                event_data = json.dumps(event, ensure_ascii=False)
                yield f"event: {event_type}\ndata: {event_data}\n\n"
                
        except Exception as e:
            error_event = {"stage": "error", "message": str(e)}
            yield f"event: error\ndata: {json.dumps(error_event, ensure_ascii=False)}\n\n"
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'  # 禁用 nginx 缓冲
        }
    )
```

---

## 三、`api接口文档.md` 改动

### 3.1 更新目录

在目录中新增：
```
- [3.4 问诊对话（流式）POST /api (stream=true)](#34-问诊对话流式-post-api-streamtrue)
```

### 3.2 新增流式接口文档章节

```markdown
### 3.4 问诊对话（流式）POST /api (stream=true)

与 3.3 使用相同端点，通过 `stream` 参数启用流式输出。

**请求体**（在原有字段基础上新增）:

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `stream` | boolean | 否 | `false` | 是否启用流式输出 |

**响应格式**: `text/event-stream` (SSE)

**事件类型**:

| 事件 | 说明 | 数据示例 |
|------|------|---------|
| `preprocessing` | 前置处理完成 | `{"stage":"preprocessing","risk_detection":{...}}` |
| `tree` | 决策树状态 | `{"stage":"tree","tree":"胸痛问诊","tree_node":"1","node_flag":false}` |
| `think` | 思考内容片段 | `{"stage":"think","delta":"患者描述..."}` |
| `answer` | 回复内容片段 | `{"stage":"answer","delta":"您好，"}` |
| `done` | 完成 | `{"stage":"done","full_think":"...","full_answer":"..."}` |
| `error` | 错误 | `{"stage":"error","message":"..."}` |

**curl 测试示例**:

\`\`\`bash
curl -N http://localhost:8060/api \
  -H "Content-Type: application/json" \
  -d '{"patient_say": "我胸口疼", "history": [], "stream": true}'
\`\`\`

**JavaScript 调用示例**:

\`\`\`javascript
const eventSource = new EventSource('/api?' + new URLSearchParams({...}));
// 或使用 fetch + ReadableStream

fetch('/api', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({patient_say: "你好", history: [], stream: true})
}).then(response => {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    // 逐块读取...
});
\`\`\`
```

---

## 四、兼容性保证

| 场景 | 行为 |
|------|------|
| 不传 `stream` 参数 | 与当前完全一致，返回 JSON |
| `stream: false` | 与当前完全一致，返回 JSON |
| `stream: true` | 启用 SSE 流式响应 |
| `stream: "true"` | 启用 SSE（字符串兼容） |

**原有 curl 测试完全不受影响**。

---

## 五、实现步骤

1. **Step 1**: 在 `new_app.py` 中新增 `StreamTagParser` 类
2. **Step 2**: 在 `new_app.py` 中新增 `main_llm_stream` 函数
3. **Step 3**: 在 `new_app.py` 中新增 `main_app_start_with_history_stream` 函数
4. **Step 4**: 在 `chat_api.py` 中新增流式响应逻辑
5. **Step 5**: 更新 `api接口文档.md`
6. **Step 6**: 本地测试验证

---

## 六、风险与注意事项

| 风险点 | 应对措施 |
|--------|---------|
| 流式传输中途断开 | 前端需实现重连/错误处理 |
| Nginx 缓冲影响 | 响应头已包含 `X-Accel-Buffering: no` |
| 日志无法记录完整响应 | `done` 事件中包含完整内容，可在此处记录 |
| 超长连接超时 | 建议 nginx 配置 `proxy_read_timeout 300s` |

---

## 七、测试验证

```bash
# 非流式（兼容性验证）
curl http://localhost:8060/api -H "Content-Type: application/json" \
  -d '{"patient_say": "你好", "history": []}'

# 流式
curl -N http://localhost:8060/api -H "Content-Type: application/json" \
  -d '{"patient_say": "你好", "history": [], "stream": true}'
```

预期流式输出：
```
event: preprocessing
data: {"stage":"preprocessing","risk_detection":{"is_proxy":false,...}}

event: tree
data: {"stage":"tree","tree":null,"tree_node":null,"node_flag":true}

event: think
data: {"stage":"think","delta":"患者"}

event: think
data: {"stage":"think","delta":"打招呼"}

event: answer
data: {"stage":"answer","delta":"您好"}

event: answer
data: {"stage":"answer","delta":"！请问"}

event: done
data: {"stage":"done","full_think":"患者打招呼...","full_answer":"您好！请问有什么不舒服？"}
```
