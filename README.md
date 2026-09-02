# sub2api

把 AI 编码代理的**订阅额度**包装成统一的 **HTTP API**。首个渠道是 [AdaL](https://docs.sylph.ai/)（headless CLI / Python SDK 两种接入方式），通过渠道适配器抽象，新渠道（Claude Code、Codex CLI 等）可以零改动服务端地快速接入。

## 架构

```text
            HTTP 客户端
                │
   POST /v1/chat        POST /v1/chat/stream (SSE)
                │
┌───────────────────────────────────────────────┐
│  server/app.py —— 只认识归一化契约             │
│  公共逻辑: 会话管理 · 错误映射 · 聚合 · SSE     │
└───────────────┬───────────────────────────────┘
                │ ChatRequest ↓ / Event 流 ↑
┌───────────────┴───────────────────────────────┐
│  core/channel.BaseChannel（抽象基类）          │
│  公共管线: 懒启动 · 异常→TurnFailed 归一化      │
└───┬───────────────┬───────────────┬──────────────┬───────────┐
    │               │               │              │           │
 echo           adal-cli        adal-sdk      adal-backend  adal-cloud
(参考实现)   (子进程+NDJSON)  (Python SDK)  (本地后端HTTP) (远程代理·无需adal)
```

**`adal-cloud`（推荐）**：直接调用 AdaL 托管代理 `api.adal.sylph.ai/proxy/*`，
不在本机启动任何 `adal` 进程，订阅额度由云端代理计费。只需一次性登录拿
Clerk JWT（`adal` 登录或内置设备码 OAuth 流），之后 Pi / Claude Code 等客户端
指向 sub2api 的 OpenAI 兼容端点即可。

**设计原则**：共性下沉到 `core`（请求/事件契约、会话存储、错误分类、聚合、SSE 编码、生命周期与错误归一化管线）；渠道只实现差异部分（运行时探测、参数构造、原始事件 → 归一化事件的翻译）。

## 快速开始

```bash
# 安装依赖（任选其一）
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # Windows
python -m venv .venv && .venv/bin/pip install -e ".[dev]"       # macOS/Linux

# 启动（默认 echo 渠道，无需任何外部依赖）
sub2api --port 8080
# 或: python -m sub2api --port 8080

# 切换到 adal-cloud 渠道（推荐，无需安装 adal CLI；首次会触发设备码登录）
SUB2API_CHANNEL=adal-cloud python -m sub2api
# 若已用 `adal` 登录过，直接复用 ~/.adal/adal_oauth_creds.json；
# 或显式注入 token: SUB2API_AUTH_TOKEN=<jwt> SUB2API_CHANNEL=adal-cloud python -m sub2api

# 切换到 AdaL 渠道（需先安装并登录 AdaL CLI：adal）
SUB2API_CHANNEL=adal-cli python -m sub2api
```

## OpenAI 兼容接口（可接入 cliproxyapi 等聚合器）

`sub2api` 同时暴露标准 OpenAI Chat Completions 协议，任何支持自定义 `base_url` 的客户端/网关都能直接把它当 OpenAI 上游使用：

| 端点 | 说明 |
|---|---|
| `POST /v1/chat/completions` | 兼容 `messages` / `model` / `stream`；`stream: true` 时输出 `chat.completion.chunk` SSE，以 `data: [DONE]` 结束 |
| `POST /v1/responses` | OpenAI Responses API（`adal-cloud` 透传）：`input` / `model` / `stream` / `reasoning` / `background` 原样透传，SSE 事件 `response.created` → `response.completed` |
| `GET /v1/responses/{id}` | 获取/轮询已创建的 response 对象（后台推理模式） |
| `DELETE /v1/responses/{id}` | 删除已存储的 response 对象 |
| `GET /v1/models` | OpenAI 格式模型列表（来自当前渠道的 `models` 声明） |

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"hi"}],"stream":false}'
```

映射规则：

- `messages` 摊平成单轮 prompt：单条 user 消息取原文；多角色历史转为 `[SYSTEM]/[USER]/[ASSISTANT]` 标记文本（渠道按一次性请求消费）。
- 思考增量（`thought.delta`）映射为 DeepSeek 风格的 `delta.reasoning_content`；工具事件不透传。
- 流中失败：终帧附 `error` 对象后正常收尾 `[DONE]`；非流失败返回 OpenAI 错误形状 `{"error":{"message","type","code"}}`。
- 该路由的权限模式由 `SUB2API_OPENAI_PERMISSION_MODE` 控制（默认 `yolo`——headless 调用方无法批准工具确认）。**公网部署务必配合 `SUB2API_API_KEY` 与 `SUB2API_ENABLED_TOOLS` 白名单收敛风险。**

### 原生透传（`adal-cloud` 渠道）

当激活 `adal-cloud` 时，sub2api 把云端代理的原生端点直接暴露出来，**跳过归一化事件层**——工具调用、`usage` 计量、`stop_reason`、多轮 `messages`、thinking 签名全部原样透传，零损耗：

| 端点 | 协议 | 透传到 |
|---|---|---|
| `POST /v1/messages` | Anthropic Messages API | `api.adal.sylph.ai/proxy/v1/messages`（X-Target-URL=api.anthropic.com） |
| `POST /v1/chat/completions` | OpenAI Chat Completions | `api.adal.sylph.ai/proxy/v1/chat/completions`（X-Target-URL 按 model 推断，默认 OpenAI） |
| `POST /v1/responses` | OpenAI Responses API | `api.adal.sylph.ai/proxy/v1/responses`（X-Target-URL 按 model 推断，默认 OpenAI） |
| `GET /v1/responses/{id}` | OpenAI Responses API | 轮询/获取已创建的 response 对象 |
| `DELETE /v1/responses/{id}` | OpenAI Responses API | 删除已存储的 response 对象 |
| `GET /v1/usage` | sub2api 自有 | `adal.sylph.ai/api/subscription/user/{id}`（订阅额度查询，见下文 cc-switch 集成） |

CLIProxyAPI（同时支持 OpenAI 和 Anthropic 上游）可直接把 sub2api 配为上游，无需 SSE 解析。Claude Code 也可直连 `/v1/messages`：

```bash
# Anthropic 原生（Claude Code）
curl http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-5","max_tokens":1024,"messages":[{"role":"user","content":"hi"}]}'
```

curl http://127.0.0.1:48080/v1/messages -H "Authorization: Bearer sk-sub2api-secret" -H "Content-Type: application/json" -d '{"model":"claude-sonnet-5","max_tokens":1024,"messages":[{"role":"user","content":"hi"}]}'

```bash
# OpenAI Responses API（现代端点，支持 reasoning/background/stream）
curl http://127.0.0.1:8080/v1/responses \
  -H "Authorization: Bearer sk-sub2api-secret" \
  -H "Content-Type: application/json" \
  -d '{"model":"openai-gpt-5.6-sol","input":"Plan a 3-day Tokyo itinerary.","stream":false}'

# 流式：SSE 事件 response.created → response.output_text.delta → response.completed
curl http://127.0.0.1:8080/v1/responses \
  -H "Authorization: Bearer sk-sub2api-secret" \
  -H "Content-Type: application/json" \
  -d '{"model":"openai-gpt-5.6-sol","input":"count 1 2 3","stream":true}'

# 轮询已创建的 response（后台推理模式）
curl http://127.0.0.1:8080/v1/responses/resp_abc123 \
  -H "Authorization: Bearer sk-sub2api-secret"
```

**注**：归一化事件层（`text.delta`/`thought.delta`/`tool.*`）仅对 `echo`/`adal-cli`/`adal-sdk`/`adal-backend` 等需要翻译的渠道生效；`adal-cloud` 走透传路径时不经过该层。

### 订阅额度查询（cc-switch 集成）

`adal-cloud` 渠道额外暴露 `GET /v1/usage`，直接回报 AdaL 订阅的**真实剩余额度**（美元计价的 credits），供 cc-switch 等客户端展示"用量"：

```bash
curl http://127.0.0.1:48080/v1/usage -H "Authorization: Bearer sk-sub2api-secret"
```

```json
{
  "isValid": true,
  "planName": "Pro",
  "total": 80.0,
  "used": 0.625384,
  "remaining": 79.374616,
  "unit": "USD",
  "extra": "trialing · resets 2026-09-09T09:24:29Z",
  "sub2api": {
    "channel": "adal-cloud", "pool_enabled": false,
    "accounts": 1, "accounts_resolved": 1, "accounts_parked": 0,
    "queried_at": "2026-09-02T12:29:43Z",
    "detail": [{ "ok": true, "email": "…", "tier": "pro", "status": "trialing", "…": "…" }]
  }
}
```

机制与语义：

- 账号身份取自各 JWT 的 `sub`（Clerk user id），随后两次**匿名** GET 拿到订阅记录——**token 过期或账号被封依然能查额度**。
- `total` 取订阅记录的 `monthly_credits`（**不是**套餐目录里的额定值：试用/改价账号两者会不同）；套餐目录只用于把 `tier` 翻成 `planName`。
- 多账号池模式下，所有**解析成功**的账号额度求和（含已 park 的账号），"不可用"信号通过 `isValid` / `invalidMessage` 表达，而非把数字清零；`planName` 以 `Pro x2` 形式标注同套餐数量，`extra` 附带 park 原因。
- 聚合结果缓存 30s、套餐目录缓存 600s，账号查询并发上限 8——定时轮询不会放大成上游请求风暴。`?refresh=1` 强制绕过缓存。
- 额度查询**永不抛错**：上游异常降级为 `isValid: false` + `invalidMessage`，不会让端点 500。
- 别名：`GET /usage`、`GET /v1/v1/usage`；受 `SUB2API_API_KEY` 保护（未带 key 返回 401）。非 `adal-cloud` 渠道返回 501 `channel_not_supported`。

cc-switch「用量查询 → 自定义脚本」直接粘贴（`127.0.0.1` 属 loopback，免 HTTPS 校验；同源校验因两者同为 sub2api 地址而通过）：

```javascript
({
  request: {
    url: "{{baseUrl}}/v1/usage",
    method: "GET",
    headers: { "Authorization": "Bearer {{apiKey}}", "User-Agent": "cc-switch/1.0" }
  },
  extractor: function (r) {
    return {
      isValid: r.isValid, invalidMessage: r.invalidMessage,
      planName: r.planName, used: r.used, total: r.total,
      remaining: r.remaining, unit: r.unit, extra: r.extra
    };
  }
})
```

### 多账号池（`adal-cloud` 渠道）

当拥有多个 AdaL 订阅账号时，可以配置账号池实现**高并发调度**与**资源共享**。账号池支持：

- **轮询（round-robin）/ 最少连接（least-connections）**两种调度策略
- 每账号独立**并发上限**，总并发由所有账号上限之和决定
- **健康追踪**：连续失败达阈值后自动冷却该账号，冷却期满自动恢复
- **故障降级**：所有账号都在冷却时 fail-open，选择最早恢复的账号

#### 配置方式

通过环境变量 `SUB2API_ACCOUNTS` 传入 JSON，或放置配置文件 `~/.adal/accounts.json`：

```json
{
  "strategy": "round-robin",
  "max_failures": 3,
  "cooldown_seconds": 60,
  "accounts": [
    {"token": "<jwt-1>", "session_id": "sub2api-acct1", "max_concurrent": 4},
    {"token": "<jwt-2>", "session_id": "sub2api-acct2", "max_concurrent": 4},
    {"token": "<jwt-3>", "session_id": "sub2api-acct3", "max_concurrent": 2}
  ]
}
```

| 字段 | 默认 | 说明 |
|---|---|---|
| `strategy` | `round-robin` | 调度策略：`round-robin` 或 `least-connections` |
| `max_failures` | `3` | 连续失败多少次后冷却该账号 |
| `cooldown_seconds` | `60` | 冷却时长（秒），期满自动恢复 |
| `accounts[].token` | — | 账号的 Clerk JWT |
| `accounts[].session_id` | 自动生成 | 代理会话 ID，留空则自动生成 |
| `accounts[].max_concurrent` | `4` | 该账号最大并发请求数 |

```bash
# 环境变量方式
SUB2API_ACCOUNTS='{"strategy":"round-robin","accounts":[{"token":"jwt-a"},{"token":"jwt-b"}]}' \
  SUB2API_CHANNEL=adal-cloud python -m sub2api

# 或文件方式
echo '{"accounts":[{"token":"jwt-a"},{"token":"jwt-b"}]}' > ~/.adal/accounts.json
SUB2API_CHANNEL=adal-cloud python -m sub2api
```

未配置账号池时，自动退化为单账号模式（使用 `SUB2API_AUTH_TOKEN` 或 `~/.adal/adal_oauth_creds.json`）。账号池文件变化会在后续模型、聊天或健康请求前安全重载；池状态可通过 `/healthz` 的 `channel.pool` 字段查看，其中 `healthy_accounts`、`available_capacity` 和 `models_available` 反映当前可用性。

### Prompt Cache 自动注入（`adal-cloud` 渠道）

sub2api 自动为透传请求注入 **prompt cache** 标记，利用上游提供商（Anthropic/OpenAI）的 prefix caching 能力节省额度。缓存由上游提供商管理（TTL 通常 5 分钟~1 小时），sub2api 仅负责注入缓存标记。

| 提供商 | 注入内容 | 机制 |
|---|---|---|
| Anthropic | `cache_control: {type:"ephemeral"}` 注入到 `system` 最后一块 | 上游对 system prompt 做前缀缓存，命中时 `cache_read_input_tokens > 0` |
| OpenAI Responses | `prompt_cache_key: "sub2api"` 注入到请求体 | 上游按 key 缓存前缀，命中时 `cached_tokens > 0` |

**行为**：
- 仅当客户端**未发送** `cache_control` / `prompt_cache_key` 时才注入（幂等，不影响 Claude Code 等原生支持缓存的客户端）
- 字符串格式的 `system` 会被转换为 `[{"type":"text","text":..., "cache_control":{"type":"ephemeral"}}]`
- 缓存命中需要 system prompt ≥ 1024 tokens（Anthropic）或 input ≥ 1024 tokens（OpenAI）

`/healthz` 的 `channel.prompt_cache` 字段可查看注入状态。


### CLIProxyAPI 接入详细指南

[CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI) 同时支持 OpenAI 与 Anthropic 两种上游协议，可以把 sub2api 当作一个自建上游接入，再对外统一暴露 OpenAI / Claude / Gemini 兼容端点。`adal-cloud` 渠道下 sub2api 原样透传上游响应，CLIProxyAPI 负责按客户端格式分发，**sub2api 不解析 SSE**。

#### 前置：启动 sub2api

```bash
# 1) 准备 AdaL 凭证（二选一）
#    a) 已有 JWT：写入 ~/.adal/adal_oauth_creds.json，或导出 SUB2API_AUTH_TOKEN=<jwt>
#    b) 首次授权：启动后访问日志里的 verification_url 输入 user_code
export SUB2API_CHANNEL=adal-cloud
export SUB2API_PORT=8080
export SUB2API_API_KEY=sk-sub2api-secret      # 对外鉴权密钥；CLIProxyAPI 侧需配同样的值
python -m sub2api

$env:SUB2API_PORT=48080;$env:SUB2API_CHANNEL="adal-cloud";$env:SUB2API_API_KEY="sk-sub2api-secret"; python -m sub2api

# → Uvicorn running on http://127.0.0.1:8080
```

验证 sub2api 可用：

```bash
curl http://127.0.0.1:8080/v1/models -H "Authorization: Bearer sk-sub2api-secret"
curl http://127.0.0.1:8080/healthz
```

#### 不经 CLIProxyAPI：直接连接

`adal-cloud` 的 `/v1/chat/completions` 和 `/v1/messages` 本身就是原生 OpenAI / Anthropic 端点，任何兼容客户端可直接连 sub2api，**无需 CLIProxyAPI**。下面三种场景只有需要多凭证轮转 / 多上游聚合时才加 CLIProxyAPI。

**OpenAI 客户端直连**（curl / 任意 OpenAI SDK / Codex / OpenCode 等）：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-sub2api-secret" \
  -H "Content-Type: application/json" \
  -d '{"model":"openai-gpt-5.6-terra","messages":[{"role":"user","content":"hi"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-sub2api-secret")
r = client.chat.completions.create(
    model="openai-gpt-5.6-terra",
    messages=[{"role": "user", "content": "hi"}],
)
# 流式
stream = client.chat.completions.create(
    model="openai-gpt-5.6-terra", messages=[{"role":"user","content":"count 1 2 3"}], stream=True)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

**Anthropic 客户端直连**（Claude Code / Anthropic SDK）：

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_API_KEY=sk-sub2api-secret
claude   # Claude Code 直连 sub2api /v1/messages
```

直连时 `model` 既可填 `/v1/models` 返回的目录 id（如 `openai-gpt-5.6-terra`、`anthropic-claude-sonnet-5`），也可填上游原生 id（如 `gpt-5.6-terra`、`claude-sonnet-5`）——sub2api 自动把目录 id 重写为上游 `model_id`，其余字段原样透传。

#### 经 CLIProxyAPI：多凭证聚合

仅当需要以下能力时才在 sub2api 前加一层 CLIProxyAPI：

- 多上游凭证轮转 / 负载均衡 / 故障切换
- 对外统一暴露 OpenAI + Claude + Gemini 多种协议
- 对客户端隐藏 sub2api 的模型命名（用 `alias` 改名）

下面三种方案按客户端协议选择。

#### 方案一：OpenAI 协议接入（`openai-compatibility`）

把 sub2api 的 `/v1/chat/completions` 配为 OpenAI 兼容上游。CLIProxyAPI 收到客户端请求后以 OpenAI Chat Completions 格式转发给 sub2api，sub2api 再透传到云端代理。

```yaml
# CLIProxyAPI config.yaml
host: ""
port: 8317
api-keys:
  - "client-key-1"          # 客户端访问 CLIProxyAPI 用的 key

openai-compatibility:
  - name: "sub2api-adal"
    base-url: "http://127.0.0.1:8080/v1"      # sub2api 地址
    api-keys:
      - "sk-sub2api-secret"                    # 与 SUB2API_API_KEY 一致；未设置则留空字符串
    models:
      # name = sub2api /v1/models 返回的目录 id；alias = 对客户端暴露的名字
      # 完整列表见 GET /v1/models；下面列出全部 33 个模型
      - name: "anthropic-claude-sonnet-5"
        alias: "claude-sonnet-5"
      - name: "anthropic-claude-sonnet-4-6"
        alias: "claude-sonnet-4-6"
      - name: "anthropic-claude-opus-5"
        alias: "claude-opus-5"
      - name: "anthropic-claude-opus-4-6"
        alias: "claude-opus-4-6"
      - name: "openai-gpt-5.6-terra"
        alias: "gpt-5.6-terra"
      - name: "openai-gpt-5.6-luna"
        alias: "gpt-5.6-luna"
      - name: "openai-gpt-5.6-sol"
        alias: "gpt-5.6-sol"
      - name: "google-gemini-3.7-flash"
        alias: "gemini-3.7-flash"
      - name: "google-gemini-3.1-pro-preview"
        alias: "gemini-3.1-pro-preview"
      - name: "google-gemini-3.6-flash"
        alias: "gemini-3.6-flash"
      - name: "google-gemini-3-flash-preview"
        alias: "gemini-3-flash-preview"
      - name: "zai-glm-5.3-flash"
        alias: "glm-5.3-flash"
      - name: "zai-glm-5.3"
        alias: "glm-5.3"
      - name: "zai-glm-5.2"
        alias: "glm-5.2"
      - name: "zai-glm-5.1"
        alias: "glm-5.1"
      - name: "deepseek-deepseek-v4-flash"
        alias: "deepseek-v4-flash"
      - name: "deepseek-deepseek-v4-flash-vision-exp"
        alias: "deepseek-v4-flash-vision-exp"
      - name: "deepseek-deepseek-v4-pro"
        alias: "deepseek-v4-pro"
      - name: "kimi-kimi-k3"
        alias: "kimi-k3"
      - name: "kimi-kimi-k2.7-code"
        alias: "kimi-k2.7-code"
      - name: "minimax-MiniMax-M2.7"
        alias: "MiniMax-M2.7"
      - name: "minimax-MiniMax-M3"
        alias: "MiniMax-M3"
      - name: "xai-grok-4.6"
        alias: "grok-4.6"
      - name: "xai-grok-4.5"
        alias: "grok-4.5"
      - name: "qwen-qwen3.8-flash"
        alias: "qwen3.8-flash"
      - name: "qwen-qwen3.8-max"
        alias: "qwen3.8-max"
      - name: "qwen-qwen3.7-max"
        alias: "qwen3.7-max"
      - name: "qwen-qwen3.7-plus"
        alias: "qwen3.7-plus"
      - name: "meta-muse-spark-1.2"
        alias: "muse-spark-1.2"
      - name: "meta-muse-spark-1.1"
        alias: "muse-spark-1.1"
      - name: "chatgpt_web-gpt-5.6-sol"
        alias: "chatgpt-web-gpt-5.6-sol"
      - name: "chatgpt_web-gpt-5.6-terra"
        alias: "chatgpt-web-gpt-5.6-terra"
      - name: "chatgpt_web-gpt-5.6-luna"
        alias: "chatgpt-web-gpt-5.6-luna"
```

客户端调用（OpenAI SDK / curl）：

```bash
curl http://127.0.0.1:8317/v1/chat/completions \
  -H "Authorization: Bearer client-key-1" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-5","messages":[{"role":"user","content":"hi"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8317/v1", api_key="client-key-1")
r = client.chat.completions.create(
    model="claude-sonnet-5",
    messages=[{"role": "user", "content": "hi"}],
)
```

流式同样可用——`stream: true` 时 sub2api 原样透传 `chat.completion.chunk` SSE：

```python
stream = client.chat.completions.create(
    model="claude-sonnet-5",
    messages=[{"role": "user", "content": "count 1 2 3"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

#### 方案二：Anthropic 协议接入（`claude-api-key`）

把 sub2api 的 `/v1/messages` 配为 Anthropic 兼容上游。Claude Code 等原生 Anthropic 客户端经 CLIProxyAPI → sub2api → 云端代理，thinking 签名、`usage`、`tool_use` 全部原样保留。

```yaml
# CLIProxyAPI config.yaml
host: ""
port: 8317
api-keys:
  - "client-key-1"

claude-api-key:
  - api-key: "sk-sub2api-secret"               # 与 SUB2API_API_KEY 一致
    base-url: "http://127.0.0.1:8080"           # sub2api 根地址（不含 /v1）
    models:
      - name: "anthropic-claude-sonnet-5"       # sub2api 实际模型名
        alias: "claude-sonnet-5"                # 对客户端暴露的名字
      - name: "anthropic-claude-opus-5"
        alias: "claude-opus-5"
      - name: "anthropic-claude-sonnet-4-6"
        alias: "claude-sonnet-4-6"
    # sub2api 不做 Claude Code 伪装，关掉 cloak 避免改写请求
    cloak:
      mode: "never"
```

Claude Code 直连 CLIProxyAPI：

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8317
export ANTHROPIC_API_KEY=client-key-1
claude
```

或直连 sub2api（不经 CLIProxyAPI）：

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_API_KEY=sk-sub2api-secret
claude
```

#### 方案三：双协议同时接入

在同一个 CLIProxyAPI 实例里同时配 `openai-compatibility` 和 `claude-api-key`，指向同一个 sub2api。客户端按需选 OpenAI 或 Anthropic 协议接入，CLIProxyAPI 自动路由。

```yaml
api-keys:
  - "client-key-1"

openai-compatibility:
  - name: "sub2api-adal-openai"
    base-url: "http://127.0.0.1:8080/v1"
    api-keys: ["sk-sub2api-secret"]
    models:
      - name: "anthropic-claude-sonnet-5"
        alias: "claude-sonnet-5"
      - name: "openai-gpt-5.6-terra"
        alias: "gpt-5.6-terra"

claude-api-key:
  - api-key: "sk-sub2api-secret"
    base-url: "http://127.0.0.1:8080"
    cloak:
      mode: "never"
    models:
      - name: "anthropic-claude-sonnet-5"
        alias: "claude-sonnet-5"
```

#### 可用模型清单

`GET /v1/models` 返回 33 个目录模型（`<provider>-<model>` 格式）。配置 CLIProxyAPI 时 `name` 字段填这些 id，`alias` 自定义对客户端暴露的名字：

| Provider | 模型（sub2api id） |
|---|---|
| anthropic | `anthropic-claude-sonnet-5`, `anthropic-claude-sonnet-4-6`, `anthropic-claude-opus-5`, `anthropic-claude-opus-4-6` |
| openai | `openai-gpt-5.6-terra`, `openai-gpt-5.6-luna`, `openai-gpt-5.6-sol` |
| google | `google-gemini-3.7-flash`, `google-gemini-3.1-pro-preview`, `google-gemini-3.6-flash`, `google-gemini-3-flash-preview` |
| zai | `zai-glm-5.3-flash`, `zai-glm-5.3`, `zai-glm-5.2`, `zai-glm-5.1` |
| deepseek | `deepseek-deepseek-v4-flash`, `deepseek-deepseek-v4-flash-vision-exp`, `deepseek-deepseek-v4-pro` |
| kimi | `kimi-kimi-k3`, `kimi-kimi-k2.7-code` |
| minimax | `minimax-MiniMax-M2.7`, `minimax-MiniMax-M3` |
| xai | `xai-grok-4.6`, `xai-grok-4.5` |
| qwen | `qwen-qwen3.8-flash`, `qwen-qwen3.8-max`, `qwen-qwen3.7-max`, `qwen-qwen3.7-plus` |
| meta | `meta-muse-spark-1.2`, `meta-muse-spark-1.1` |
| chatgpt_web | `chatgpt_web-gpt-5.6-sol`, `chatgpt_web-gpt-5.6-terra`, `chatgpt_web-gpt-5.6-luna` |

#### 字段对照与排错

| CLIProxyAPI 字段 | 值 | 说明 |
|---|---|---|
| `base-url`（openai-compatibility） | `http://127.0.0.1:8080/v1` | sub2api 地址 + `/v1` |
| `base-url`（claude-api-key） | `http://127.0.0.1:8080` | sub2api 根地址，CLIProxyAPI 自动追加 `/v1/messages` |
| `api-keys` / `api-key` | 与 `SUB2API_API_KEY` 一致 | sub2api 未设 `SUB2API_API_KEY` 时填空字符串 |
| `models[].name` | sub2api `/v1/models` 里的 `id` | 必须完全匹配 |
| `models[].alias` | 自定义 | 客户端请求时用的名字 |
| `cloak.mode`（claude-api-key） | `"never"` | sub2api 透传不做伪装，关掉避免改写 |

常见问题：

- **401 invalid api key**：CLIProxyAPI 的 `api-keys`/`api-key` 与 sub2api 的 `SUB2API_API_KEY` 不一致。
- **模型不在列表**：CLIProxyAPI `models[].name` 拼写与 `/v1/models` 返回的 `id` 不符；用 `curl /v1/models` 核对。
- **`auth_unavailable: no auth available`**：CLIProxyAPI 的 `openai-compatibility.models[]` 里**没有配客户端请求的 model**。CLIProxyAPI 按客户端发的 `model` 匹配 `models[].name` 或 `alias`，匹配不到就报此错。解决：把所有要用的模型都加到 `models[]`（见上方完整 33 个模型配置），或确保客户端请求的 model 名与 `alias` 一致。
- **`/v1/messages` 走了 OpenAI 上游**：`claude-api-key` 的 `base-url` 不要带 `/v1`，CLIProxyAPI 自己追加 `/v1/messages`。
- **404 `{"detail":"Not Found"}`**：`base-url` 多带了 `/v1` 导致路径变成 `/v1/v1/messages`。`claude-api-key` 填根地址 `http://127.0.0.1:8080`（不带 `/v1`）。若无法改 CLIProxyAPI 配置，sub2api 也兼容 `/v1/v1/messages` 和 `/v1/v1/chat/completions` 别名。
- **400 `max_tokens` not supported**：OpenAI 新模型（gpt-5.6-*）只认 `max_completion_tokens`。sub2api 已自动把 `max_tokens` 重写为 `max_completion_tokens`（仅 OpenAI 模型），CLIProxyAPI 侧无需改动。
- **流式中断**：确认 sub2api 进程存活（`curl /healthz`），且 CLIProxyAPI 与 sub2api 同机或网络可达。

### API

| 端点 | 说明 |
|---|---|
| `POST /v1/chat` | 同步对话，返回 `{answer, session_id, channel, model}` |
| `POST /v1/chat/stream` | SSE 流式，逐帧输出归一化事件 |
| `GET /v1/channels` | 已注册渠道列表 + 当前渠道健康状态 |
| `GET /v1/sessions/{id}` | 会话详情（轮数、native id、模型） |
| `GET /v1/usage` | 订阅额度（`adal-cloud`）：`isValid`/`planName`/`used`/`total`/`remaining`/`unit`，`?refresh=1` 绕过缓存 |
| `GET /healthz` | 存活探针 |

请求体：

```json
{
  "prompt": "fix the failing test",
  "session_id": null,             // 传回上次响应的 id 即可续接多轮
  "model": "claude-sonnet-4-6",
  "permission_mode": "default",   // default | acceptEdits | yolo
  "enabled_tools": ["Read", "Search"],
  "workspace": "/path/to/project"
}
```

SSE 帧类型：`session.started` → `thought.delta` / `text.delta` / `message.completed` / `tool.started` / `tool.completed` → 终态 `turn.completed`（含渠道 native session id 与模型）或 `turn.failed`（含统一错误码）。

### 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `SUB2API_CHANNEL` | `echo` | 激活的渠道名 |
| `SUB2API_HOST` / `SUB2API_PORT` | `127.0.0.1` / `8080` | 监听地址 |
| `SUB2API_WORKSPACE` | `.` | 渠道默认工作目录 |
| `SUB2API_ACCOUNTS` | — | JSON，多账号池配置（详见「多账号池」章节）；也可用 `~/.adal/accounts.json` |
| `SUB2API_AUTH_TOKEN` | — | 显式 JWT（如 AdaL 的 `access_token`），CI 无浏览器时用 |
| `SUB2API_API_KEY` | — | 设置后 `/v1/*` 全部要求 `Authorization: Bearer <key>`（`/healthz` 除外） |
| `SUB2API_OPENAI_PERMISSION_MODE` | `yolo` | OpenAI 兼容路由使用的权限模式 |
| `SUB2API_ENABLED_TOOLS` | — | 部署级工具白名单（逗号分隔），请求未显式指定时生效 |
| `SUB2API_RUNTIME_PATH` | — | 渠道运行时路径（如 adal 可执行文件） |
| `SUB2API_CHANNEL_OPTIONS` | — | JSON，透传给渠道的额外选项 |

## 接入一个新渠道

只需一个文件 + 一行注册，**不改任何服务端代码**：

1. 新建 `sub2api/channels/my_channel.py`，继承 `BaseChannel` 并实现 `_chat()`：
2. 在 `sub2api/channels/__init__.py` 里 import 它；
3. 用 `SUB2API_CHANNEL=my_channel` 启动即生效。

```python
from typing import AsyncIterator, ClassVar
from ..core.channel import BaseChannel
from ..core.registry import register
from ..core.types import ChatRequest, Event, MessageCompleted, TurnCompleted


@register
class MyChannel(BaseChannel):
    name: ClassVar[str] = "my_channel"          # 全局唯一
    display_name: ClassVar[str] = "My Agent"
    models: ClassVar[tuple[str, ...]] = ("model-a",)

    def runtime_available(self) -> bool:        # 健康检查（可选）
        return True

    async def _chat(self, request: ChatRequest) -> AsyncIterator[Event]:
        # 把上游协议翻译成归一化事件；续接用 request.native_session_id
        yield MessageCompleted(text="answer")
        yield TurnCompleted(session_id="<上游会话id>", model="model-a")
```

约定：

- **只产出归一化事件**，以恰好一个终态事件（`TurnCompleted`/`TurnFailed`）结束；异常随便抛——基类管线统一转成 `turn.failed`。
- **会话双轨**：客户端只见服务端 `session_id`；渠道续接所需的上游 id 放进 `TurnCompleted.session_id`，服务端自动记录并在下轮以 `request.native_session_id` 还给你。
- 文本要么用 `TextDelta` 流式、要么用 `MessageCompleted` 整段，同一轮不要混用两者表达同一段文字。
- 参考 `channels/echo.py`（最小模板）与 `channels/adal_cli.py`（子进程 + NDJSON 解析 + 参数构造，均为纯函数可直接单测）。

## 测试

```bash
pytest -q          # 65 个用例：契约、注册表、会话、聚合、两个 AdaL 渠道的解析/参数/子进程端到端、HTTP 全表面、OpenAI 兼容层
```

无需安装 AdaL 即可跑全部测试——AdaL 渠道用假运行时（临时启动器脚本）做子进程级验证，SDK 渠道测纯映射函数。

## 许可

MIT
