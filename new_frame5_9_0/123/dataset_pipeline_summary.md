# 数据集构造、打分与筛选流程总结

本文基于以下脚本梳理整条数据处理链路：

- `build_xnk_datasets.py`
- `annotation_script.py`
- `evaluation_tree_test.py`
- `llm_similarity_scorer.py`
- `ground_truth_quality_filter.py`
- `adjust_accuracy.py`

目标不是复述每一行实现，而是解释这套流程**为什么这样设计、每一步依据什么标准、最终如何形成可用数据集**。

---

## 1. 总体流程概览

整条链路可以概括为 5 个主阶段 + 1 个可选后处理阶段：

1. **从原始对话中构造候选数据集**  
   用规则从原始 JSONL 中筛出目标病种/场景样本，并保证树状主题覆盖。

2. **为每个样本抽取“医生最终诊疗建议”，并确定最小有效上下文 `k`**  
   即：判断在前几轮对话里，医生已经足以给出诊疗结论。

3. **让评测模型只看前 `k` 轮对话，生成预测诊疗结论**  
   这个阶段模拟“模型在有限上下文下是否已经能得出接近医生结论的建议”。

4. **用独立评分模型比较预测结论与真值结论的语义一致性**  
   得到 0–5 的相似度分数。

5. **对低分样本做真值质量复核，并回写过滤后的结果与原始 JSONL**  
   核心思想是：低分不一定代表模型差，也可能是真值本身信息量不足。

6. **可选：按目标准确率做人工裁剪**  
   `adjust_accuracy.py` 属于额外后处理，更像人为调指标脚本，不是主流程的核心组成部分。

---

## 2. 阶段一：原始数据集构造

对应脚本：`build_xnk_datasets.py`

### 2.1 输入与输出

- 输入：`xnk_filtered_raw.jsonl`
- 规则文件：`xnk_regex_rules.json`
- 输出：
  - `xnk_dataset_highrisk.jsonl`
  - `xnk_dataset_chronic.jsonl`
  - 以及对应的 meta 文件

### 2.1.1 输入条目格式

`xnk_filtered_raw.jsonl` 是逐行 `json`。  
脚本对原始条目本身要求不算严格，但至少默认会读取这些字段：

- `id` 或 `pid`：样本唯一标识
- `dialogue` 或其他原始业务字段：脚本这里不会深解析，仅原样保留

可抽象理解为：

```json
{
  "id": "sample_001",
  "dialogue": [...],
  "...": "其他原始字段"
}
```

也就是说，这一阶段更关注：

- 整条原始记录的文本内容是否命中规则
- 原始记录是否能原样写回新数据集

### 2.1.2 输出条目格式

`xnk_dataset_highrisk.jsonl` 和 `xnk_dataset_chronic.jsonl` 的**每一行仍是原始样本本身**，没有把筛选证据直接写进 jsonl。

即输出数据集条目格式基本保持：

```json
{
  "id": "sample_001",
  "dialogue": [...],
  "...": "其他原始字段"
}
```

筛选证据、树覆盖、命中关键词等信息，会写到对应的 meta 文件里。

### 2.1.3 Meta 条目格式

对应的 meta 文件不是逐行 jsonl，而是一个完整 `json`，核心结构可以概括为：

```json
{
  "dataset_name": "xnk_highrisk",
  "created_from": ".../xnk_filtered_raw.jsonl",
  "rule_file": ".../xnk_regex_rules.json",
  "construction_method": "regex_auto_filter_from_source_jsonl",
  "focus": ["high_risk_disease"],
  "tree_coverage": ["tree_a", "tree_b"],
  "stats": {
    "eligible_count": 0,
    "selected_count": 0,
    "tree_distribution": {},
    "dataset_policy": {}
  },
  "samples": [
    {
      "id": "sample_001",
      "source_line": 12,
      "labels": ["high_risk_disease"],
      "primary_tree": "xxx",
      "matched_trees": ["xxx", "yyy"],
      "evidence_keywords": ["关键词1", "关键词2"],
      "raw_entry": { "...": "原始样本" }
    }
  ]
}
```

当前脚本实际启用的是：

- `high_risk_disease`
- `chronic_disease`

`acute_disease` 和 `multisystem_disease` 虽然保留了配置位，但默认未启用。

### 2.2 核心筛选逻辑

脚本先对每条原始记录做两类匹配：

1. **表型标签匹配（phenotype label）**    
   样本中只要命中对应关键词，就被视为该标签候选。

2. **主题树匹配（tree match）**  
   样本还必须命中某个医学主题树或子方向。

只有同时满足：

- 命中至少一个目标标签
- 命中至少一个主题树

才会进入候选集。

### 2.3 候选排序标准

单个样本会被打一个规则分数：

`rank = 10 * 标签命中数 + 3 * 目标树重叠数`

这意味着：

- **标签证据**比树覆盖更重要
- 但**树覆盖广度**仍然会影响优先级

### 2.4 最终入选标准

每个标签的数据集不是简单按总分截断，而是分两步选：

#### 第一步：先保证树覆盖下限

脚本先按目标树逐个补齐，确保每棵目标树至少达到 `per_tree_target_min`。

#### 第二步：再在不超过上限的前提下补足总量

在每棵树不超过 `per_tree_target_max` 的情况下，继续按排序高低填充，直到达到 `target_count_per_dataset`。

### 2.5 这一阶段的参考因素

这一阶段主要依赖**规则证据强弱**与**分布均衡性**：

- 是否命中目标标签关键词
- 是否命中对应主题树关键词
- 标签命中数量
- 树覆盖数量
- 各目标树的最少覆盖要求
- 各目标树的最大占比限制

### 2.6 可信性判断

这一阶段的逻辑是可信的，因为它不是单纯“多命中就保留”，而是同时兼顾：

- **主题相关性**
- **标签证据强度**
- **树覆盖均衡**

这能降低数据集被某个高频子场景“挤占”的风险。

---

## 3. 阶段二：医生真值抽取与最小诊断轮次 `k` 标注

对应脚本：`annotation_script.py`

这个阶段本质上是在回答两个问题：

1. 医生在整段对话中，哪一句最像“最终诊疗建议”？
2. 前多少轮对话已经足以支撑这条建议？

### 3.1 对话预处理标准

脚本先把原始对话拆成配对轮次：

- 一轮 = 患者一句 + 医生一句
- 如果对话结构不完整，无法形成轮次，则直接记为错误样本

因此，该阶段默认认为：

- **成对轮次**比散乱消息更适合作为诊疗推断单位

### 3.1.1 输入条目格式

这一阶段读取的是阶段一输出的 `jsonl`，每一条样本至少需要能支持：

- `id` 或 `pid`
- `dialogue`，或者兼容字段 `history`

脚本兼容两类对话格式。

#### 格式 A：结构化轮次

```json
{
  "id": "sample_001",
  "dialogue": [
    {"role": "patient", "content": "......"},
    {"role": "doctor", "content": "......"}
  ]
}
```

#### 格式 B：字符串轮次

```json
{
  "id": "sample_001",
  "dialogue": [
    "患者：......",
    "医生：......"
  ]
}
```

如果 `dialogue` 不存在，脚本会尝试读取 `history`。

### 3.2 医生“最终诊疗建议”如何抽取

脚本并不直接把最后一句医生回复当真值，而是对每轮医生回复做启发式打分。

#### 正向加分因素

如果医生回复中出现下列内容，会加分：

- 建议
- 复查 / 检查
- 造影
- 住院
- 评估
- 用药 / 治疗 / 手术
- 诊断 / 方案 / 调整 / 随访

#### 负向扣分因素

如果更像流程话术或寒暄，会扣分，例如：

- 你好
- 还有问题吗
- 是否配药
- 先登记 / 挂号

#### 其他辅助因素

- **问句会扣分**：因为问句通常不是最终结论
- **行动组合会加分**：例如“建议 + 复查/检查/住院/用药”同时出现
- **后位轮次加分**：越靠后越可能接近最终结论

最终若最高分候选超过阈值 `2.0`，就取该句作为真值；否则退化为：

- 取最后一轮医生回复
- 或最后一个未配对医生回复

### 3.3 显式诊疗建议筛查

得到候选真值后，脚本不会立刻进入 `k` 搜索，而是先做一轮 LLM 审核：

判断该句是否含有**显式诊疗建议**。

保留的典型内容包括：

- 明确诊断方向
- 具体检查建议
- 具体治疗 / 用药 / 调整 / 随访 / 住院计划

排除的典型内容包括：

- 问候
- 纯提问
- 挂号、登记、流程说明
- 泛泛聊天

如果没有显式诊疗建议，该样本会被保留在结果文件中，但：

- `k = None`
- `k_source = screen_no_explicit_advice`

也就是说，它不会进入真正的“最小诊断轮次”计算。

### 3.4 最小诊断轮次 `k` 的定义

脚本用 LLM 判断：

> 在前 `k` 轮对话中，信息是否已经足以支撑一个诊疗结论？

输出格式被严格约束为 JSON：

- `can_diagnose`: 是否已经可以诊断/给出诊疗建议
- `diagnosis_text`: 若可以，则返回结论文本

### 3.5 `k` 的搜索策略

不是暴力搜索全部轮次，而是：

1. **粗搜索**：先按 1、3、5… 这样的步长检测
2. 找到第一个可诊断的粗粒度位置后
3. **细搜索**：在附近做逐轮确认
4. 再检查 `k+1` 是否也稳定可诊断，避免偶然命中

同时受 `MAX_K_CHECKS = 8` 的预算限制。

### 3.6 长对话压缩标准

当轮次过长时，提示词不会机械截断，而是优先保留：

- 前 2 轮
- 后 4 轮
- 以及含关键医学词的轮次

例如与以下因素相关的轮次更容易保留：

- 症状
- 建议
- 复查 / 检查
- 治疗 / 用药
- 住院 / 造影

### 3.7 这一阶段的参考因素

这个阶段同时使用了**规则因素**和**LLM 判断因素**。

规则因素：

- 医生话术中是否包含明确行动建议
- 是否是问句/流程句
- 所处轮次位置
- 是否形成患者-医生完整轮次

LLM 判断因素：

- 当前上下文是否足以形成诊疗结论
- 真值是否属于显式诊疗建议
- 对压缩后关键信息的理解是否仍然稳定

### 3.8 可信性判断

这一步比“直接拿最后一句话当标签”更可信，原因在于它解决了两个常见噪声：

- **最后一句未必是真正的诊疗建议**
- **并非整段对话都对诊断必要**

因此，`doctor_diagnosis_raw` 和 `k` 的组合，实际上是在构造：

- 一个更像最终临床建议的真值
- 一个与之对应的最小有效上下文

### 3.9 输出条目格式

`annotation_script.py` 输出的是 `result_highrisk.json` / `result_chronic.json`，整体是一个完整 `json`，外层结构大致为：

```json
{
  "metadata": {
    "dataset_name": "highrisk",
    "input_file": "xnk_dataset_highrisk.jsonl",
    "output_file": "result_highrisk.json",
    "total_samples": 0,
    "successful_annotations": 0,
    "success_rate": "0.00%",
    "average_processing_time": "0.000s",
    "concurrency": 20,
    "summary": {}
  },
  "results": [...]
}
```

其中 `results` 中每个条目的核心格式可概括为：

```json
{
  "sample_index": 0,
  "pid": "sample_001",
  "status": "success",
  "error_message": "",
  "failure_reason": "",
  "k": 3,
  "k_source": "model_min_k_consistent",
  "k_dialogue": [
    "患者：......",
    "医生：......",
    "患者：......",
    "医生：......"
  ],
  "total_rounds": 6,
  "doctor_diagnosis_raw": "建议复查冠脉CTA并门诊随访",
  "doctor_diagnosis_round": 5,
  "doctor_diagnosis_source": "scored_candidate",
  "doctor_diagnosis_candidates": [
    {
      "round": 5,
      "doctor_text": "......",
      "score": 4.2,
      "reasons": ["+2.2:建议", "+1.8:action_bundle"]
    }
  ],
  "has_explicit_diagnosis_advice": true,
  "advice_screen_reason": "explicit_advice_present",
  "processing_time": "2.341s",
  "timestamp": "2026-03-30T12:34:56",
  "observability": {
    "total_llm_requests": 0,
    "timeout_count": 0,
    "retry_count": 0,
    "retry_success_count": 0,
    "estimated_input_tokens_total": 0,
    "error_type_counter": {}
  },
  "debug_info": {
    "search_trace": [],
    "search_note": "",
    "advice_screening": {}
  }
}
```

其中最关键的监督字段是：

- `pid`
- `k`
- `k_dialogue`
- `doctor_diagnosis_raw`
- `has_explicit_diagnosis_advice`

---

## 4. 阶段三：前 `k` 轮诊疗能力评测

对应脚本：`evaluation_tree_test.py`

这个阶段的目标是：

> 给定前 `k` 轮对话，评测模型能不能生成接近医生真值的诊疗建议。

### 4.1 输入来源

它会同时读取：

- `result_highrisk.json` / `result_chronic.json`：上一步的标注结果
- `xnk_dataset_highrisk.jsonl` / `xnk_dataset_chronic.jsonl`：原始筛后数据集

然后通过 `pid` 回溯原始对话。

### 4.1.1 输入条目格式

这一阶段需要两路输入。

#### 输入 1：标注结果 `result_*.json`

它从 `results` 中取单个标注样本，依赖的关键字段包括：

```json
{
  "sample_index": 0,
  "pid": "sample_001",
  "k": 3,
  "doctor_diagnosis_raw": "建议复查……",
  "has_explicit_diagnosis_advice": true
}
```

#### 输入 2：原始筛后数据集 `xnk_dataset_*.jsonl`

通过 `pid` 回查原始对话，条目格式仍是：

```json
{
  "id": "sample_001",
  "dialogue": [...],
  "...": "其他原始字段"
}
```

### 4.2 评测输入构造标准

从原始对话中截取前 `k` 轮，再做提示词裁剪：

- 最多保留 `MAX_CONTEXT_ROUNDS = 6`
- 最多保留 `MAX_CONTEXT_CHARS = 5000`

优先保证末尾上下文可见，避免长对话把关键结论前的诊断信息全部冲掉。

### 4.3 评测模型的输出要求

评测模型不要求写长分析，而是要抽取一句**简洁核心诊疗建议**：

- 不要标题
- 不要分析过程
- 尽量保留检查、治疗、用药、复查、住院、手术、随访等核心动作
- 不要补充对话中没有的信息
- 输出尽量压缩成一句核心建议

### 4.4 这一阶段的参考因素

这一步主要在测：

- 模型是否能利用前 `k` 轮完成临床建议抽取
- 模型是否抓住核心诊疗动作，而不是泛泛总结
- 在有限上下文中，是否保留与真值一致的临床方向

### 4.5 可信性判断

这一阶段合理之处在于：

- 输入不是整段对话，而是“最小可诊断上下文”
- 输出不是开放式长文本，而是“核心诊疗建议”

这样能把评测问题收缩成更清晰的语义对齐任务，为下一阶段打分创造条件。

### 4.6 输出条目格式

这一阶段输出 `evaluation_results_Qwen_Qwen3.5-9B_highrisk.json` 等文件
其中 `results` 的单条格式可概括为：

```json
{
  "sample_index": 0,
  "pid": "sample_001",
  "k": 3,
  "requested_k": 3,
  "total_rounds": 6,
  "input_dialogue_k": ["患者：......", "医生：......"],
  "prompt_dialogue_k": ["患者：......", "医生：......"],
  "prompt_dialogue_text": "患者：......\n医生：......",
  "prompt_rounds_used": 3,
  "prompt_turns_used": 6,
  "prompt_rounds_truncated": false,
  "prompt_chars_truncated": false,
  "has_explicit_diagnosis_advice": true,
  "prediction_text": "建议复查冠脉CTA并随访",
  "ground_truth_text": "建议复查冠脉CTA并门诊随访",
  "status": "success",
  "processing_time": "1.245s",
  "debug_info": {
    "messages": [],
    "captured_raw_responses": []
  }
}
```

其中最关键的是三组字段：

- 输入：`k`、`input_dialogue_k`、`prompt_dialogue_text`
- 预测：`prediction_text`
- 真值：`ground_truth_text`

---

## 5. 阶段四：预测-真值语义相似度打分

对应脚本：`llm_similarity_scorer.py`

这一步不是词面匹配，而是让另一个 LLM 做**非对称语义评分**。

### 5.1 非对称评分的核心思想

脚本明确要求评分模型重点判断：

> 预测文本是否覆盖并遵循真值文本的核心意图。

这意味着它不是双向“完全一致”标准，而是偏向判断：

- 预测有没有保留真值中的核心管理动作
- 即使预测更详细、更积极，只要不冲突，默认不扣分

### 5.2 评分标准

输出分值范围为 `0–5`，建议以 `0.5` 为步长。

脚本给出的评分参考可以概括为：

- **4.0–5.0**：核心意图完整覆盖，无明显冲突
- **3.5**：有意义的部分覆盖，方向一致
- **3.0**：有部分重合，但核心动作较弱
- **2.0–2.5**：仅少量相关，核心动作基本未保留
- **0.0–1.5**：基本无关、明显冲突、场景偏离或有安全风险

其中一个很重要的偏置是：

- **默认从宽**
- **边界样本倾向上调**

比如难区分 `3.0` 和 `3.5` 时，优先给 `3.5`。

### 5.3 对“低信息真值”的处理

脚本会先检测真值是否属于低信息文本，例如：

- 太短
- 更像流程词
- 缺少明确医疗动作

但当前版本中，这个检测**只做标记，不做过滤**。

也就是说：

- 低信息真值仍然会参与打分
- 只是会在结果里留下 `reference_quality` 供后续分析

### 5.3.1 输入条目格式

这一阶段读取的是阶段三输出的 `evaluation_results_*.json`。  
进入评分器的单条样本至少需要：

```json
{
  "sample_index": 0,
  "pid": "sample_001",
  "status": "success",
  "prediction_text": "建议复查……",
  "ground_truth_text": "建议复查……"
}
```

### 5.4 这一阶段的参考因素

评分模型重点参考：

- 真值的核心临床动作是否被覆盖
- 方向是否一致
- 是否存在冲突或安全风险
- 预测是否只是扩展、细化，而不是偏离
- 真值是否属于短管理句，但仍有有效临床动作

### 5.5 可信性判断

相比严格字面重合，这个评分设计更接近真实医疗对话场景，因为医生表达常常有：

- 简写
- 展开
- 替代表达
- 兼容性的附加建议

因此，“非对称 + 从宽”的规则，适合拿来评估是否**抓住临床意图**，而不是是否**逐字复现答案**。

### 5.6 输出条目格式

这一阶段输出 `similarity_scores_*.json`。外层结构大致为：

```json
{
  "metadata": {
    "input_file": "evaluation_results_....json",
    "dataset_name": "highrisk",
    "scoring_model": {
      "name": "deepseek-v3.1",
      "url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "api_key": "******"
    },
    "rules_version": "pure_llm_direct_score_v5_more_lenient_prompt",
    "summary": {},
    "timestamp": "2026-03-30T12:34:56"
  },
  "results": [...]
}
```

其中 `results` 的单条格式可概括为：

```json
{
  "dataset_name": "highrisk",
  "sample_index": 0,
  "pid": "sample_001",
  "upstream_status": "success",
  "ground_truth_text": "建议复查冠脉CTA并门诊随访",
  "prediction_text": "建议复查冠脉CTA并随访",
  "reference_quality": {
    "is_low_info_ground_truth": false,
    "low_info_score": 1,
    "low_info_reasons": [],
    "ground_truth_len": 18
  },
  "status": "success",
  "error_message": "",
  "scoring": {
    "final_score_0_5": 4.5,
    "rationale": "预测覆盖了真值核心动作，方向一致"
  },
  "processing_time_sec": 1.103,
  "debug_info": {
    "messages": [],
    "raw_llm_response": "{\"final_score_0_5\":4.5,\"rationale\":\"...\"}",
    "parsed_llm_response": {},
    "llm_call": {}
  }
}
```

其中最关键字段是：

- `ground_truth_text`
- `prediction_text`
- `scoring.final_score_0_5`
- `scoring.rationale`
- `reference_quality`

---

## 6. 阶段五：低分样本的真值质量复核与最终筛选

对应脚本：`ground_truth_quality_filter.py`

这一步是整条链路里很关键的“纠偏步骤”。

核心思路是：

> 低分样本不一定说明模型错，也可能是真值本身过弱、过流程化、信息量太低，不适合作为高质量监督样本。

### 6.1 哪些样本会进入复核

只有分数不高于阈值的样本会进入复核：

- 默认阈值：`final_score_0_5 <= 3.5`

这个阈值的含义是：

- `4.0` 以上通常认为对齐较好
- `3.5` 及以下则值得检查是否存在真值问题

### 6.1.1 输入条目格式

这一阶段输入的是 `similarity_scores_*.json`。  
其中会重点读取 `results` 里的以下字段：

```json
{
  "sample_index": 0,
  "pid": "sample_001",
  "status": "success",
  "ground_truth_text": "建议复查……",
  "prediction_text": "建议随访……",
  "scoring": {
    "final_score_0_5": 2.5,
    "rationale": "仅部分覆盖"
  }
}
```

只有 `status=success` 且 `final_score_0_5 <= 3.5` 的样本会进入低分复核候选集。

### 6.2 规则优先的真值质量判断

脚本先用规则判断真值是否应该直接过滤。

#### 直接保留的典型模式

如果真值含有明确医学动作，会优先保留，例如：

- 建议
- 复查 / 检查 / 造影
- 住院 / 复诊 / 监测
- 继续、停用、换药、减量、加量
- 按原剂量服药

如果是“结果解释 + 下一步动作”，也会保留。

#### 直接过滤的典型模式

若真值更像以下内容，则倾向过滤：

- 礼貌结束语
- 纯确认
- 配药 / 订单 / 库存 / 支付 / 配送等流程信息
- 仅收集信息的短问句
- 数量确认、是否要药之类短流程问答

### 6.3 规则无法判断时的 LLM 复核

如果规则没有给出明确结论，才调用 LLM 判断：

- 是否应过滤
- 质量标签是 `low_quality` 还是 `acceptable`
- 给出简短原因、证据片段和置信度

这一层属于**规则兜底后的语义判断**，目的是减少纯规则漏判。

### 6.4 最终过滤动作

如果某条低分样本被判断为低质量真值，则会：

1. 从相似度打分结果中删除该样本
2. 输出过滤后的相似度结果文件：`*_filtered.json`
3. 输出判定明细：`ground_truth_quality_decisions_{dataset}.json`
4. 根据被删 `pid` 回写原始 JSONL，生成 `*_filtered.jsonl`

也就是说，最终留下来的不仅是“分数更高”的样本，更是“真值本身也更像有效临床监督信号”的样本。

### 6.4.1 输出 1：过滤后的相似度结果格式

输出文件形如：`similarity_scores_xxx_filtered.json`。  
它的整体结构与原始 `similarity_scores_*.json` 基本一致，只是：

- `results` 中删除了被过滤样本
- `metadata` 中新增 `ground_truth_quality_filter`

例如：

```json
{
  "metadata": {
    "...": "...",
    "summary": {},
    "ground_truth_quality_filter": {
      "enabled": true,
      "policy": "strict_rule_plus_llm_v1",
      "low_score_threshold": 3.5,
      "dataset_name": "highrisk",
      "judge_model": {
        "name": "deepseek-v3.1",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_source": "..."
      },
      "filter_stats": {},
      "decisions_file": "ground_truth_quality_decisions_highrisk.json",
      "timestamp": "2026-03-30T12:34:56"
    }
  },
  "results": [...]
}
```

### 6.4.2 输出 2：低分样本质量判定明细格式

输出文件形如：`ground_truth_quality_decisions_{dataset}.json`。  
结构大致为：

```json
{
  "metadata": {
    "dataset_name": "highrisk",
    "input_file": "similarity_scores_xxx.json",
    "policy": "strict_rule_plus_llm_v1",
    "low_score_threshold": 3.5,
    "judge_model": {},
    "timestamp": "2026-03-30T12:34:56"
  },
  "summary": {},
  "decisions": [...]
}
```

其中 `decisions` 单条格式可概括为：

```json
{
  "dataset_name": "highrisk",
  "sample_index": 0,
  "pid": "sample_001",
  "record_key": "pid:sample_001",
  "final_score_0_5": 2.5,
  "ground_truth_text": "建议复查……",
  "judge_status": "rule_based_filter",
  "error_message": "",
  "judge_result": {
    "should_filter": true,
    "quality_label": "low_quality",
    "reason": "主要是流程或短问句，缺少有效临床动作",
    "evidence_span": "请先登记复诊",
    "confidence": 0.98
  },
  "debug_info": {}
}
```

### 6.4.3 输出 3：最终过滤后的原始数据集格式

输出文件形如：`xnk_dataset_highrisk_filtered.jsonl`。  
其**每一行仍然是原始样本结构**，只是在样本级别删除了被判定应过滤的 `pid`：

```json
{
  "id": "sample_001",
  "dialogue": [...],
  "...": "其他原始字段"
}
```

因此，这一步对原始数据格式没有做 schema 改造，只做了样本删除。

### 6.5 这一阶段的参考因素

主要参考两类因素：

#### 第一类：真值本身是否有监督价值

- 是否包含临床动作
- 是否只是流程信息
- 是否只是礼貌/结束语
- 是否仅是问句或收集信息
- 是否包含结果解释及下一步计划

#### 第二类：低分是否可能由真值质量差造成

- 相似度分数是否较低
- 真值是否短、弱、流程化
- 真值与临床管理动作是否脱钩

### 6.6 可信性判断

这是整条流程中最能提升数据可靠性的步骤之一，因为它避免了一个常见问题：

- **把“评测低分”误当成“模型差”**

实际上有些低分来自：

- 真值太短
- 真值只是在走流程
- 真值不是有效临床建议

把这些样本过滤掉后，保留下来的数据更适合：

- 做训练监督
- 做模型对齐评估
- 做高质量 benchmark

---

## 7. 可选阶段：按目标准确率裁剪数据

对应脚本：`adjust_accuracy.py`

这个脚本的目标不是提升数据语义质量，而是：

- 通过删除错误样本，让数据集表观准确率达到目标值

其默认目标是：

- `highrisk` 达到 `0.90`
- `chronic` 达到 `0.85`

### 7.1 实现方式

它会：

1. 读取评测结果
2. 统计当前正确样本数 `C`、总样本数 `N`
3. 计算至少要删掉多少个错误样本，才能让 `C / (N-k)` 高于目标
4. 删除前 `k` 个错误样本
5. 同步更新：
   - evaluation 结果
   - 原始 JSONL
   - annotation 结果文件

### 7.1.1 输入与输出格式说明

`adjust_accuracy.py` 假设存在一套和当前主流程并不完全一致的评测结果格式。  
它读取的 `evaluation_results` 预期包含：

```json
{
  "round_results": [
    {
      "metadata": {
        "metrics": {
          "total_samples": 100,
          "correct_samples": 85
        }
      },
      "results": [
        {
          "sample_index": 0,
          "is_correct": true
        }
      ]
    }
  ]
}
```

并同步改写：

- `evaluation_results_*.json`
- `xnk_dataset_*.jsonl`
- `result_*.json`

其中：

- 对 `json` 文件，主要是删除对应 `sample_index` 并重排索引
- 对 `jsonl` 文件，主要是按行删除对应样本

### 7.2 风险与局限

这个脚本更像“指标修正工具”，不是严格的数据质量筛选逻辑，原因有三点：

1. **它按是否答错删除，不按样本质量删除**
2. **它假设 evaluation 文件结构固定**，与当前其他脚本的输出结构并不完全一致
3. **它会直接改写源文件**，可追溯性较弱

因此，更合理的定位是：

- 可选、人工控制的后处理工具
- 不应作为主流程的核心质量标准

---

## 8. 整条流程中的“标准”和“参考因素”汇总

如果要把整条链路压缩成一套清晰标准，可以概括为下面三层。

### 8.1 数据集构造标准

关注“样本是否属于目标任务范围”：

- 是否命中目标疾病/表型标签
- 是否命中目标主题树
- 证据是否足够强
- 分布是否覆盖多个子树，而不是集中在少数场景

### 8.2 打分标准

关注“模型输出是否保留医生核心临床意图”：

- 是否覆盖真值中的核心诊疗动作
- 是否方向一致
- 是否存在冲突或风险
- 即使更详细，只要兼容也不轻易扣分
- 对短管理句、有效管理指令采取偏宽松解释

### 8.3 最终筛选标准

关注“样本本身是否值得保留为高质量监督信号”：

- 真值是否为有效临床动作
- 真值是否只是流程、配药、礼貌、问句
- 低分是否源于真值质量差，而非模型能力差
- 过滤后是否仍能回溯到原始 JSONL 并形成干净子集

---

## 9. 一个更通顺且可信的流程结论

如果用一句话概括，这套方案并不是简单的“筛数据 + 打分”，而是：

> 先用规则保证任务相关性和主题覆盖，再用启发式 + LLM 找到医生的有效诊疗结论及最小诊断上下文，再让评测模型在该上下文下生成建议，用独立 LLM 做临床语义对齐评分，最后把低分但真值质量差的样本剔除掉，从而留下更适合作为医疗监督信号的数据集。

它之所以相对可信，是因为每一步都在处理一种常见噪声：

- **构造阶段**处理“任务不相关 / 分布失衡”
- **标注阶段**处理“真值句不准 / 上下文冗余”
- **评测阶段**处理“有限上下文能否形成结论”
- **打分阶段**处理“临床语义等价但表达不一致”
- **筛选阶段**处理“低分其实是真值质量差”

最终留下的数据更像是：

- 任务边界清晰
- 真值更接近真实临床建议
- 评测上下文更合理
- 评分更贴近临床语义
- 低质量监督信号被进一步清除

---

## 10. 建议如何使用这套结果

如果目标是做训练或评测，建议优先使用：

1. `ground_truth_quality_filter.py` 之后输出的 `*_filtered.jsonl`
2. 对应的 `*_filtered.json` 打分结果
3. 决策明细文件 `ground_truth_quality_decisions_{dataset}.json`

原因是这一步之后的数据同时满足：

- 已做任务相关性筛选
- 已做最小诊断轮次标注
- 已做模型预测与真值的语义打分
- 已剔除一部分低质量真值样本

而 `adjust_accuracy.py` 更适合单独评估是否要用，不建议默认纳入主流程。
