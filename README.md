# NVIDIA Model Router

智能路由 NVIDIA NIM 免费模型，并发探针竞速 + 粘性会话 + 思维链归一化。

## 特性

- **并发探针竞速**：所有模型并发探测（1 token，无思考），谁先响应就用谁
- **API Key 轮询**：多个 key 轮换分配，分散压力
- **粘性会话**：成功一次后记住该模型，下次跳过探针直接请求（TTL 可配）
- **思维链归一化**：统一不同模型的 thinking 输出格式（normalize/strip/passthrough）
- **OpenAI 兼容**：`/v1/chat/completions` 接口，直接替换 OpenAI SDK base_url

## 快速开始

```bash
# 1. 克隆
git clone https://github.com/lq-259/nvidia-router.git
cd nvidia-router

# 2. 配置
cp .env.example .env
# 编辑 .env 填入 NVIDIA_API_KEYS

# 3. 启动
docker compose up -d
```

## 配置说明

| 环境变量 | 说明 | 默认值 |
|---|---|---|
| `NVIDIA_API_KEYS` | API Key，逗号分隔，轮询使用 | 必填 |
| `NVIDIA_MODELS` | 模型列表，逗号分隔 | 6 个免费模型 |
| `REQUEST_TIMEOUT` | 探针超时（秒） | 2 |
| `FULL_REQUEST_TIMEOUT` | 完整请求超时（秒） | 30 |
| `STICKY_TTL` | 粘性会话有效期（秒） | 300 |
| `THINKING_MODE` | normalize / strip / passthrough | normalize |
| `AUTH_API_KEY` | 本服务鉴权 Key，留空不鉴权 | 空 |
| `PORT` | 服务端口 | 8000 |

## 思维链模式

不同模型处理思考内容的方式不同（DeepSeek 用独立 `reasoning_content`，Kimi 用 `reasoning`，GLM 可能内联 ` response` 标签）。通过 `THINKING_MODE` 统一输出：

| 模式 | 行为 |
|---|---|
| `normalize`（默认） | 内联思考提取到 `reasoning_content`，已有该字段的保持不变 |
| `strip` | 完全移除思考内容，只返回 `content` |
| `passthrough` | 不做任何处理，保留原始输出 |

## API

### 对话

```bash
POST /v1/chat/completions
Content-Type: application/json
Authorization: Bearer <AUTH_API_KEY>  # 如果设置了

{
  "model": "auto",
  "session_id": "user-abc",
  "messages": [{"role": "user", "content": "hello"}],
  "stream": false
}
```

### 流式

```bash
POST /v1/chat/completions
{
  "model": "auto",
  "session_id": "user-abc",
  "messages": [{"role": "user", "content": "hello"}],
  "stream": true
}
```

### 健康检查

```bash
GET /health
GET /v1/models
```

### 粘性会话管理

```bash
GET  /sessions/{session_id}     # 查看
DELETE /sessions/{session_id}   # 清除
```

## OpenAI SDK 使用

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-auth-key",          # AUTH_API_KEY
    base_url="http://localhost:8000/v1",
)

response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "hello"}],
    extra_body={"session_id": "user-abc"},
)
print(response.choices[0].message.content)
```

## 路由逻辑

```
请求 → 粘性会话命中？ → 直接请求上次成功的模型
       ↓ 失败/无
       并发探针所有模型（max_tokens=1，无思考）→ 谁先响应 → 发完整请求
       ↓ 失败
       按探针速度顺序 fallback → 下一个
       ↓ 成功
       更新粘性缓存 → 归一化 thinking → 返回
```