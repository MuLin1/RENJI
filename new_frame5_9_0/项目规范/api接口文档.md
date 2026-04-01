# 模型 API 接口文档

> **服务端口**: 8060  
> **协议**: HTTP  
> **数据格式**: JSON (UTF-8)
> **生产环境api调用端口**： http://113.59.64.88:8060/api

---

## 目录

1. [概述](#1-概述)
2. [通用说明](#2-通用说明)
3. [接口列表](#3-接口列表)
   - [3.1 服务状态 GET /](#31-服务状态-get-)
   - [3.2 健康检查 GET /health](#32-健康检查-get-health)
   - [3.3 问诊对话 POST /api](#33-问诊对话-post-api)
4. [请求字段详细说明](#4-请求字段详细说明)
   - [4.1 patient_say — 患者发言](#41-patient_say--患者发言)
   - [4.2 history — 对话历史](#42-history--对话历史)
   - [4.3 tree / tree_node / node_flag — 决策树状态](#43-tree--tree_node--node_flag--决策树状态)
   - [4.4 report — 就医档案](#44-report--就医档案)
   - [4.5 medical_orders — 过往医嘱](#45-medical_orders--过往医嘱)
   - [4.6 department — 科室](#46-department--科室)
5. [响应格式说明](#5-响应格式说明)
   - [5.1 成功响应](#51-成功响应)
   - [5.2 错误响应](#52-错误响应)
6. [多轮对话状态管理](#6-多轮对话状态管理)
7. [完整调用示例](#7-完整调用示例)

---

## 1. 概述

本系统提供基于大语言模型的智能问诊对话服务。调用方通过 HTTP POST 请求发送患者发言、对话历史、就医档案等信息，系统返回医生回复及相关决策树状态。

**核心交互流程**：
1. 调用方发送患者本轮发言及历史对话
2. 系统返回医生回复、决策树状态、风险检测结果
3. 调用方在下一轮请求中 **原样回传** 上一轮返回的 `tree`、`tree_node`、`node_flag` 字段

---

## 2. 通用说明

| 项目 | 说明 |
|------|------|
| 请求方式 | HTTP POST（主接口 `/api`） |
| Content-Type | `application/json; charset=utf-8` |
| 响应编码 | UTF-8，中文不转义 |
| 超时建议 | ≥ 60 秒（模型推理耗时较长） |

---

## 3. 接口列表

### 3.1 服务状态 GET /

检查服务是否在线。

**请求**: 无参数

**响应示例**:
```json
{
    "status": "success",
    "message": "New Frame V* API is running",
    "endpoints": {
        "api": "/api [POST]",
        "health": "/health [GET]"
    }
}
```

---

### 3.2 健康检查 GET /health

健康探针接口，可用于负载均衡器或容器健康检查。

**请求**: 无参数

**响应**:
```json
{
    "status": "healthy",
    "service": "new_frame_v*"
}
```

---

### 3.3 问诊对话 POST /api

**核心接口**。接收患者发言及上下文信息，返回AI医生回复。

**URL**: `POST /api`

**请求头**:
```
Content-Type: application/json
```

**请求体概览**:

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `patient_say` | string | 条件必填¹ | `""` | 当前患者输入 |
| `history` | array | 条件必填¹ | `[]` | 对话历史记录 |
| `tree` | string \| null | 否 | `null` | 当前决策树名称 |
| `tree_node` | string \| null | 否 | `null` | 当前决策树节点ID |
| `node_flag` | boolean | 否 | `false` | 节点标志 |
| `report` | object \| null | 否 | `null` | 就医档案数据 |
| `medical_orders` | array \| null | 否 | `null` | 过往医嘱信息 |
| `department` | string | 否 | `"心内科"` | 科室名称 |

> ¹ `patient_say` 和 `history` **至少有一个不为空**，否则返回 400 错误。

---

## 4. 请求字段详细说明

### 4.1 patient_say — 患者发言

| 属性 | 说明 |
|------|------|
| 类型 | `string` |
| 必填 | 与 `history` 至少有一个不为空 |
| 说明 | 患者本轮的完整输入文本 |

- 允许为空字符串 `""`（此时 `history` 不能为空）。
- 若患者上传了图片，图片描述文本应拼接在 `patient_say` 内，格式为：`"...正常文字..., 患者上传了图片[图片内容描述]"`。

**示例**:
```json
"patient_say": "周三开始心脏这边背部有点疼痛，主要集中在下午"
```

---

### 4.2 history — 对话历史

| 属性 | 说明 |
|------|------|
| 类型 | `array` of `object` |
| 必填 | 与 `patient_say` 至少有一个不为空 |
| 默认值 | `[]` |

**数组元素结构**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `role` | string | 是 | 角色，取值为 `"patient"` 或 `"doctor"` |
| `content` | string | 是 | 该轮的发言内容 |

#### 格式约束（违反将导致错误）

1. **角色严格交替**：相邻两条记录的 `role` 必须不同，即 `patient` → `doctor` → `patient` → `doctor` → ...
2. **多条记录时必须以 `doctor` 结尾**：当 `history` 包含 2 条及以上记录时，最后一条的 `role` 必须是 `"doctor"`。
3. **允许的特殊情况**：
   - **空数组 `[]`**：首次问诊，无历史记录。
   - **仅一条 `patient` 记录**：患者预填信息场景。
   - **仅一条 `doctor` 记录**：医生开场白场景。

#### 合法示例

**首次问诊（无历史）**:
```json
"history": []
```

**单条患者预填信息**:
```json
"history": [
    {"role": "patient", "content": "我有高血压3年，一直在服药"}
]
```

**多轮对话历史**（注意：以 `doctor` 结尾）:
```json
"history": [
    {"role": "patient", "content": "医生我这个需要进一步检查吗？"},
    {"role": "doctor", "content": "目前在服用什么降压药？身高体重多少？"},
    {"role": "patient", "content": "目前是替米沙坦，160cm 155斤"},
    {"role": "doctor", "content": "超重比较明显，建议完成冠脉CT评估"}
]
```

#### 非法示例（会导致错误）

```json
// ❌ 相邻角色重复
"history": [
    {"role": "patient", "content": "我头疼"},
    {"role": "patient", "content": "还有点恶心"}
]

// ❌ 多条记录但未以 doctor 结尾
"history": [
    {"role": "patient", "content": "我头疼"},
    {"role": "doctor", "content": "什么时候开始的？"},
    {"role": "patient", "content": "昨天开始的"}
]
```

---

### 4.3 tree / tree_node / node_flag — 决策树状态

这三个字段用于维护 **决策树遍历状态**，调用方应将上一轮响应返回的值 **原样回传**。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tree` | string \| null | `null` | 当前决策树名称 |
| `tree_node` | string \| null | `null` | 当前节点ID |
| `node_flag` | boolean | `false` | 节点完成标志 |

**注意事项**:
- `node_flag` 兼容字符串类型：`"true"` / `"True"` 均视为 `true`，其余视为 `false`。
- 首次请求时不传或传 `null`，由系统自动选择决策树。
- **务必将每次响应中的 `tree`、`tree_node`、`node_flag` 保存并在下一次请求中回传**，否则会导致对话流程重置。

**示例（非首次请求）**:
```json
{
    "tree": "有冠脉CT检查",
    "tree_node": "4",
    "node_flag": true
}
```

---

### 4.4 report — 就医档案

| 属性 | 说明 |
|------|------|
| 类型 | `object` \| `null` |
| 必填 | 否 |

当患者有就医档案时传入此字段。传 `null` 或不传表示无档案信息。

#### report 对象结构

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `patientSex` | string | 否 | 患者性别。`"1"` = 男，`"2"` = 女 |
| `patientAge` | string | 否 | 患者年龄（纯数字字符串，如 `"48"`） |
| `assayMD` | string | 否 | 检验报告，**Markdown 格式**（见下方格式约束） |
| `examineMD` | string | 否 | 检查报告，**Markdown 格式**（见下方格式约束） |

> **重要**: 旧版 `assayList` / `examineList` 字段已废弃，请使用 `assayMD` / `examineMD`。

#### 4.4.1 assayMD 字段 — 检验报告 Markdown 格式约束

`assayMD` 是一个 Markdown 格式的字符串，用于传递患者的检验（化验）报告数据。

**格式规范**:

1. **每份检验报告以一级标题开头**，格式为：
   ```
   # [<检验日期>] <科室> <检验项目名>
   ```
2. **标题下方紧跟 Markdown 表格**，包含以下四列：

   | 列名 | 说明 |
   |------|------|
   | 项目 | 检验指标名称 |
   | 结果 | 检测数值及单位 |
   | 参考范围 | 正常参考范围 |
   | 提示 | 异常标记（H=偏高, L=偏低, ↑=偏高, ↓=偏低, 空=正常） |

3. 多份检验报告在同一个字符串中 **依次排列**，以标题分隔。

**assayMD 格式示例**:
```
# [2025-07-28 07:59] 检验科 肝功能(新3)+血脂4项+肾功能

| 项目 | 结果 | 参考范围 | 提示 |
|------|------|---------|------|
| 总胆红素 | 25.8μmol/L | 5.1-22.2μmol/L | H |
| 直接胆红素 | 6.8μmol/L | 0-8.6μmol/L | |
| γ-谷氨酰转肽酶 | 174U/L | 7-45U/L | H |
| 谷氨酸脱氢酶 | 17.6U/L | 0-8.0U/L | H |
| 总胆固醇 | 4.62mmol/L | 3.1-5.7mmol/L | |

# [2025-07-28 07:54] 检验科 PSA、FPSA组合①

| 项目 | 结果 | 参考范围 | 提示 |
|------|------|---------|------|
| 总前列腺特异性抗原(t-PSA) | 4.180ng/ml | 0-4.0ng/ml | H |
| 游离前列腺特异性抗原(f-PSA) | 0.685ng/ml | 0-0.93ng/ml | |
| 游离PSA/总PSA | 0.16 | >0.15 | |
```

#### 4.4.2 examineMD 字段 — 检查报告 Markdown 格式约束

`examineMD` 是一个 Markdown 格式的字符串，用于传递患者的检查（影像/超声/心电图等）报告数据。

**格式规范**:

1. **每份检查报告以一级标题开头**，格式为：
   ```
   # [<检查日期>] <报告类型>
   ```
2. **标题下方包含以下信息字段**（使用键值对格式）：
   - `检查部位`: 检查的身体部位
   - `检查结论`: 报告的主要结论文本

3. 多份检查报告在同一个字符串中 **依次排列**。

**examineMD 格式示例**:
```
# [2025-07-28 08:42] 彩超检查报告

检查部位: 肾脏, 肝，胆，胰，脾（需空腹）
检查结论: 胆囊壁稍毛糙
胰腺回声增高
肝脏,脾脏未见明显异常
双侧肾脏未见明显异常

# [2025-08-08 12:08] CT报告

检查部位: 光子冠脉CTA
检查结论: 左前降支中段可疑轻度非钙化斑块，请结合临床、必要时完善DSA检查。

# [2025-08-07 12:39] 动态心电报告

检查部位: 心脏
检查结论: 全程基础心律为窦性心律；平均心率62bpm；最快心率120bpm；最慢心率43bpm。
```

#### report 完整示例

```json
"report": {
    "patientSex": "1",
    "patientAge": "48",
    "assayMD": "# [2025-07-28 07:59] 检验科 肝功能(新3)+血脂4项+肾功能\n\n| 项目 | 结果 | 参考范围 | 提示 |\n|------|------|---------|------|\n| 总胆红素 | 25.8μmol/L | 5.1-22.2μmol/L | H |\n| γ-谷氨酰转肽酶 | 174U/L | 7-45U/L | H |",
    "examineMD": "# [2025-07-28 08:42] 彩超检查报告\n\n检查部位: 肾脏, 肝，胆，胰，脾\n检查结论: 胆囊壁稍毛糙\n胰腺回声增高"
}
```

---

### 4.5 medical_orders — 过往医嘱

| 属性 | 说明 |
|------|------|
| 类型 | `array` of `object` \| `null` |
| 必填 | 否 |

传入患者的历史医嘱记录。传 `null` 或不传表示无医嘱。

#### 医嘱对象结构

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `medicalDate` | string | 否 | 就诊日期，格式 `"YYYY-MM-DD"`（如 `"2025-06-30"`） |
| `deptName` | string | 否 | 就诊科室名称（如 `"消化科-网医"`） |
| `diagnose` | string | 否 | 主要诊断（如 `"胃炎"`） |
| `differTime` | string | 否 | 建议复诊时间（如 `"六个月内"`） |
| `recommendDiagnose` | array | 否 | 鉴别诊断列表 |
| `prescriptionDrugList` | array | 否 | 处方药品列表 |

#### recommendDiagnose 数组元素结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `diagnoseName` | string | 诊断名称（如 `"高血压病"`） |

#### prescriptionDrugList 数组元素结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `drugName` | string | 药品名称（如 `"(甲)泮托拉唑钠肠溶片"`） |
| `drugFrequency` | string | 用药频率代码（见下方频率代码表） |
| `drugDose` | string | 单次剂量数值（如 `"40"`） |
| `drugDoseUnit` | string | 剂量单位（如 `"mg"`） |
| `usage` | string | 给药途径（如 `"口服"`） |

#### 用药频率代码对照表

| 代码 | 含义 |
|------|------|
| `QD` | 每日一次 |
| `BID` | 每日两次 |
| `TID` | 每日三次 |
| `QID` | 每日四次 |
| `Q8H` | 每8小时一次 |
| `Q12H` | 每12小时一次 |
| `PRN` | 必要时 |
| `QOD` | 隔日一次 |

#### medical_orders 完整示例

```json
"medical_orders": [
    {
        "medicalDate": "2025-06-30",
        "deptName": "消化科-网医",
        "diagnose": "胃炎",
        "differTime": "六个月内",
        "recommendDiagnose": [
            {"diagnoseName": "高血压病"},
            {"diagnoseName": "2型糖尿病"}
        ],
        "prescriptionDrugList": [
            {
                "drugName": "(甲)泮托拉唑钠肠溶片",
                "drugFrequency": "QD",
                "drugDose": "40",
                "drugDoseUnit": "mg",
                "usage": "口服"
            }
        ]
    }
]
```

---

### 4.6 department — 科室

| 属性 | 说明 |
|------|------|
| 类型 | `string` |
| 默认值 | `"心内科"` |
| 必填 | 否 |

传入患者当前挂号/就诊的科室名称。如不传则默认为 `"心内科"`。

**若传入不支持的科室名称，将返回 400 错误，并附带当前支持的科室列表。务必保证传入参数与列表中的科室名称严格一致。**

#### 当前支持的科室

| 科室名称 |
|---------|
| 心内科 |
| 肝脏外科 |
| 消化科 |
| 胆胰外科 |
| 风湿科 |
| 骨关节外科 |
| 生殖医学科 |
| 疼痛科 |
| 泌尿科 |
| 全科医学科 |
| 胃肠外科 |
| 妇产科 |
| 内分泌科 |
| 乳腺外科 |
| 神经内科 |
| 神经外科 |
| 肾内科 |
| 血液科 |

> 注意：科室列表可能随系统更新而变化，请以实际返回的错误信息中的列表为准。

---

## 5. 响应格式说明

### 5.1 成功响应

**HTTP Status**: `200 OK`

```json
{
    "state": "success",
    "content": {
        "llm": {
            "think": "<模型思考过程文本>",
            "response": "<医生回复文本>"
        },
        "tree": "<决策树名称 或 null>",
        "tree_node": "<节点ID 或 null>",
        "node_flag": "<布尔值>",
        "risk_detection": {
            "is_proxy": "<布尔值>",
            "reason": "<检测原因说明>",
            "source": "<检测来源>"
        }
    }
}
```

#### 响应字段说明

| 字段路径 | 类型 | 说明 |
|---------|------|------|
| `state` | string | 固定为 `"success"` |
| `content.llm.think` | string | 模型推理的思考过程（可用于调试，不建议展示给患者） |
| `content.llm.response` | string | **医生回复内容**（展示给患者的主要内容） |
| `content.tree` | string \| null | 当前决策树名称（需在下轮请求中回传） |
| `content.tree_node` | string \| null | 当前决策树节点ID（需在下轮请求中回传） |
| `content.node_flag` | boolean | 节点标志（需在下轮请求中回传） |
| `content.risk_detection` | object | 代问诊检测结果 |
| `content.risk_detection.is_proxy` | boolean | 是否为代问诊。`true` = 他人代为问诊，`false` = 患者本人 |
| `content.risk_detection.reason` | string | 检测判定的原因说明 |
| `content.risk_detection.source` | string | 检测来源。`"alg"` = 规则引擎判定，`"llm"` = 大模型语义判定，`"error"` = 检测出错 |

---

### 5.2 错误响应

#### HTTP 400 — 请求参数错误

```json
{
    "state": "error",
    "content": "错误原因描述"
}
```

常见错误原因：
- `"请求体必须是有效的JSON格式"` — 请求体不是合法 JSON
- `patient_say` 和 `history` 同时为空 — **注意：此错误的返回格式与其它错误不同**：
  ```json
  {"error": "患者发言和对话历史至少有一个不为空"}
  ```
- `"不支持的科室\"xxx\"。当前支持的科室有：心内科, 肝脏外科, ..."` — 传入了不支持的科室
- `"数据格式错误/Data format error."` — `history` 格式不符合交替规则

#### HTTP 500 — 服务内部错误

```json
{
    "state": "error",
    "content": "处理请求时发生错误: <具体异常信息>"
}
```

---

## 6. 多轮对话状态管理

调用方需要在每轮对话中维护以下状态，并在下一次请求中回传：

```
┌──────────────────────────────────────────────────────────┐
│                      调用方维护的状态                       │
├──────────────────────────────────────────────────────────┤
│  1. history     — 累积的对话历史数组                        │
│  2. tree        — 上轮响应中 content.tree 的值              │
│  3. tree_node   — 上轮响应中 content.tree_node 的值         │
│  4. node_flag   — 上轮响应中 content.node_flag 的值         │
└──────────────────────────────────────────────────────────┘
```

### 典型多轮对话流程

**第 1 轮（首次问诊）**:
```
请求: patient_say="我最近胸口疼", history=[], tree=null, tree_node=null, node_flag=false
响应: content.tree="胸痛问诊", content.tree_node="1", content.node_flag=false
      content.llm.response="请问疼痛是什么时候开始的？"
```

**第 2 轮（回传状态 + 追加历史）**:
```
请求: patient_say="大概三天前开始的",
      history=[
          {"role":"patient","content":"我最近胸口疼"},
          {"role":"doctor","content":"请问疼痛是什么时候开始的？"}
      ],
      tree="胸痛问诊", tree_node="1", node_flag=false
响应: content.tree="胸痛问诊", content.tree_node="2", content.node_flag=false
      content.llm.response="疼痛发作时有没有伴随出汗、气促等症状？"
```

**关键规则**:
- `history` 由调用方自行维护和累积。每轮对话后，将本轮的 `patient_say` 和响应中的 `content.llm.response` 分别以 `{"role": "patient", "content": "..."}` 和 `{"role": "doctor", "content": "..."}` 的格式追加到 `history` 数组末尾。
- `tree`、`tree_node`、`node_flag` 直接取上一次响应的对应字段值回传，**不要自行修改**。

---

## 7. 完整调用示例

### 示例 1：首次问诊（无档案）

**请求**:
```json
{
    "patient_say": "有，选择仁济医院就诊记录，上次就诊记录：仁济东院，2025-06-13，东血管外科门诊，下肢深静脉血栓形成，无过敏史，有，有冠心病史，周三开始心脏这边背部有点疼痛，主要集中在下午，无，周五在地段医院做了个心电图，无异常，建议我来仁济做心脏造影，是否有必要",
    "history": [],
    "department": "心内科"
}
```

### 示例 2：携带就医档案的首次问诊

**请求**:
```json
{
    "patient_say": "主任你好，我有高脂血症，由于吃他汀肝损，现在在用依洛优单抗，目前总胆固醇维持在4.3，低密度在2.3左右，请问目前的低密度水平达标吗？",
    "history": [],
    "department": "心内科",
    "report": {
        "patientSex": "1",
        "patientAge": "48",
        "assayMD": "# [2025-07-28 07:59] 检验科 肝功能(新3)+血脂4项+肾功能\n\n| 项目 | 结果 | 参考范围 | 提示 |\n|------|------|---------|------|\n| 总胆红素 | 25.8μmol/L | 5.1-22.2μmol/L | H |\n| γ-谷氨酰转肽酶 | 174U/L | 7-45U/L | H |\n| 谷氨酸脱氢酶 | 17.6U/L | 0-8.0U/L | H |\n| 总胆固醇 | 4.62mmol/L | 3.1-5.7mmol/L | |\n| 低密度脂蛋白胆固醇 | 2.62mmol/L | 0-3.4mmol/L | |\n\n# [2025-07-28 07:54] 检验科 PSA、FPSA组合①\n\n| 项目 | 结果 | 参考范围 | 提示 |\n|------|------|---------|------|\n| 总前列腺特异性抗原(t-PSA) | 4.180ng/ml | 0-4.0ng/ml | H |",
        "examineMD": "# [2025-07-28 08:42] 彩超检查报告\n\n检查部位: 肾脏, 肝，胆，胰，脾（需空腹）\n检查结论: 胆囊壁稍毛糙\n胰腺回声增高\n肝脏,脾脏未见明显异常\n双侧肾脏未见明显异常"
    }
}
```

### 示例 3：多轮对话（回传决策树状态）

**请求**:
```json
{
    "patient_say": "叶老师你好，麻烦你帮我看一下这个光子冠脉CTA的报告有什么问题吗？",
    "history": [],
    "tree": "有冠脉CT检查",
    "tree_node": "4",
    "node_flag": true,
    "department": "心内科",
    "report": {
        "patientSex": "1",
        "patientAge": "41",
        "assayMD": "# [2025-08-07 06:33] 检验科 心梗标志物（快速）\n\n| 项目 | 结果 | 参考范围 | 提示 |\n|------|------|---------|------|\n| 肌钙蛋白I | 0.00ng/ml | 0-0.04ng/ml | |\n| 肌红蛋白 | 23.00ng/ml | 0-70ng/ml | |\n| 肌酸激酶同工酶 | 1.3ng/ml | 0-4.3ng/ml | |",
        "examineMD": "# [2025-08-08 12:08] CT报告\n\n检查部位: 光子冠脉CTA\n检查结论: 左前降支中段可疑轻度非钙化斑块，请结合临床、必要时完善DSA检查。\n\n# [2025-08-07 12:39] 动态心电报告\n\n检查部位: 心脏\n检查结论: 全程基础心律为窦性心律；平均心率62bpm；最快心率120bpm；最慢心率43bpm。"
    }
}
```

### 示例 4：携带过往医嘱

**请求**:
```json
{
    "patient_say": "我之前吃的药还需要继续吃吗？",
    "history": [
        {"role": "patient", "content": "医生您好，我胃不太舒服"},
        {"role": "doctor", "content": "请问之前有看过医生吗？做过什么检查？"}
    ],
    "department": "消化科",
    "medical_orders": [
        {
            "medicalDate": "2025-06-30",
            "deptName": "消化科-网医",
            "diagnose": "胃炎",
            "differTime": "六个月内",
            "recommendDiagnose": [
                {"diagnoseName": "高血压病"},
                {"diagnoseName": "2型糖尿病"}
            ],
            "prescriptionDrugList": [
                {
                    "drugName": "(甲)泮托拉唑钠肠溶片",
                    "drugFrequency": "QD",
                    "drugDose": "40",
                    "drugDoseUnit": "mg",
                    "usage": "口服"
                }
            ]
        }
    ]
}
```

### 成功响应示例

```json
{
    "state": "success",
    "content": {
        "llm": {
            "think": "患者48岁男性，有高脂血症病史，因他汀类药物引起肝损而改用PCSK9抑制剂...",
            "response": "您好！根据您提供的信息，您目前低密度脂蛋白在2.3左右。对于有高脂血症且合并他汀不耐受的患者，建议将低密度脂蛋白控制在1.8mmol/L以下..."
        },
        "tree": "血脂管理",
        "tree_node": "2",
        "node_flag": false,
        "risk_detection": {
            "is_proxy": false,
            "reason": "对话内容与档案信息一致，判断为本人问诊",
            "source": "alg"
        }
    }
}
```
