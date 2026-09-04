# sub2api

把 AI 编码代理的**订阅额度**包装成统一的 **HTTP API**。首个渠道是 [AdaL](https://docs.sylph.ai/)（云端代理 / headless CLI / Python SDK 三种接入方式），通过渠道适配器抽象，新渠道（Claude Code、Codex CLI 等）可以零改动服务端地快速接入。除协议翻译外，sub2api 还负责上游兼容层（净化 Claude Code 等真实客户端的请求，把"一次成功后连续 500"变成稳定可用）、账号池调度、本地用量与成本计量，以及 `--web` 管理界面。

## 架构

```text
                              HTTP 客户端
                                   │
        ┌──────────────────────────┴───────────────────────────┐
        │ server/routes/*  —— 每个模块一个 router(ctx)          │
        │ anthropic · openai_compat · responses · models        │
        │ usage · chat · admin · web(--web)                     │
        └───────────┬──────────────────────────┬───────────────┘
                    │ 原生透传                 │ 归一化事件
        ┌───────────┴────────────┐   ┌─────────┴──────────────┐
        │ compat/ 兼容层         │   │ core/channel.BaseChannel│
        │ sanitize  请求净化     │   │ 懒启动 · 异常→TurnFailed│
        │ aggregate 流式聚合     │   └─────────┬──────────────┘
        │ errors    错误还原     │             │
        │ affinity  缓存亲和     │   echo · adal-cli · adal-sdk
        └───────────┬────────────┘        · adal-backend
                    │
        server/proxy.py  转发 · 401 重铸 · UsageMeter 计量
                    │
        api.adal.sylph.ai/proxy/*  →  Anthropic / OpenAI / zai / …
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

# 启动需要 API key —— 不设 key 的网关等于把订阅额度公开给任何能连上端口的人
SUB2API_API_KEY=sk-sub2api-secret sub2api --port 8080
# 或: SUB2API_API_KEY=sk-sub2api-secret python -m sub2api --port 8080
# 明确要开放时: SUB2API_ALLOW_ANONYMOUS=1 python -m sub2api

# adal-cloud 渠道（推荐，无需安装 adal CLI；首次会触发设备码登录）
SUB2API_API_KEY=sk-sub2api-secret python -m sub2api --channel adal-cloud
# 若已用 `adal` 登录过，直接复用 ~/.adal/adal_oauth_creds.json；
# 或显式注入 token: SUB2API_AUTH_TOKEN=<jwt>

# 全量开启：管理界面 + 计量库
SUB2API_API_KEY=sk-sub2api-secret python -m sub2api \
  --channel adal-cloud --port 8080 --web --db ./sub2api.sqlite3

# 本地 AdaL CLI 渠道（需先安装并登录 adal）
SUB2API_API_KEY=sk-sub2api-secret python -m sub2api --channel adal-cli
```

PowerShell：

```powershell
$env:SUB2API_API_KEY="sk-sub2api-secret"
python -m sub2api --channel adal-cloud --port 8080 --web
```

加账号最省事的方式是开着 `--web` 打开 `http://127.0.0.1:8080/admin`，在「一键导入账号」里**粘贴任意含 token 的内容**（裸 JWT、`~/.adal/adal_oauth_creds.json` 全文、`accounts.json` 条目都行），或点「开始设备码登录」——详见「[一键导入账号](#一键导入账号只输入一个值)」。

## OpenAI 兼容接口（可接入 cliproxyapi 等聚合器）

`sub2api` 同时暴露标准 OpenAI Chat Completions 协议，任何支持自定义 `base_url` 的客户端/网关都能直接把它当 OpenAI 上游使用：

| 端点 | 说明 |
|---|---|
| `POST /v1/chat/completions` | 兼容 `messages` / `model` / `stream`；`stream: true` 时输出 `chat.completion.chunk` SSE，以 `data: [DONE]` 结束 |
| `POST /v1/responses` | OpenAI Responses API（`adal-cloud` 透传）：`input` / `model` / `stream` / `reasoning` / `background` 原样透传，SSE 事件 `response.created` → `response.completed` |
| `GET /v1/responses/{id}` | 获取/轮询已创建的 response 对象（后台推理模式） |
| `DELETE /v1/responses/{id}` | 删除已存储的 response 对象 |
| `GET /v1/models` | OpenAI 格式模型列表。`adal-cloud` 下只列**已验证可达**的模型（当前 28 个），id 与官方原生 id 一致，不列出无法路由的条目 |

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-sub2api-secret" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"hi"}],"stream":false}'
```

映射规则：

- `messages` 摊平成单轮 prompt：单条 user 消息取原文；多角色历史转为 `[SYSTEM]/[USER]/[ASSISTANT]` 标记文本（渠道按一次性请求消费）。
- 思考增量（`thought.delta`）映射为 DeepSeek 风格的 `delta.reasoning_content`；工具事件不透传。
- 流中失败：终帧附 `error` 对象后正常收尾 `[DONE]`；非流失败返回 OpenAI 错误形状 `{"error":{"message","type","code"}}`。
- 该路由的权限模式由 `SUB2API_OPENAI_PERMISSION_MODE` 控制（默认 `yolo`——headless 调用方无法批准工具确认）。鉴权默认强制（无 `SUB2API_API_KEY` 时进程拒绝启动）；**公网部署另需 `SUB2API_ENABLED_TOOLS` 白名单收敛工具风险。**

### 原生透传（`adal-cloud` 渠道）

当激活 `adal-cloud` 时，sub2api 把云端代理的原生端点直接暴露出来，**跳过归一化事件层**——工具调用、`usage` 计量、`stop_reason`、多轮 `messages`、thinking 签名全部原样透传，零损耗：

| 端点 | 协议 | 透传到 |
|---|---|---|
| `POST /v1/messages` | Anthropic Messages API | `proxy/v1/messages`；`X-Target-URL` 按 model 的 provider 解析（`anthropic` / `zai` / `minimax` / `xai`） |
| `POST /v1/messages/count_tokens` | Anthropic Token Counting | 上游拒绝该路径，sub2api 用 `max_tokens: 1` 探针实现，回报 `{"input_tokens": N}` |
| `POST /v1/chat/completions` | OpenAI Chat Completions | `proxy/v1/chat/completions`；`X-Target-URL` 按 model 推断，默认 OpenAI |
| `POST /v1/responses` | OpenAI Responses API | `proxy/v1/responses`；`X-Target-URL` 按 model 推断，默认 OpenAI |
| `GET /v1/responses/{id}` | OpenAI Responses API | 轮询/获取已创建的 response 对象 |
| `DELETE /v1/responses/{id}` | OpenAI Responses API | 删除已存储的 response 对象 |
| `GET /v1/usage` | sub2api 自有 | 云端订阅额度 + 本地 24h 计量（见下文 cc-switch 集成） |
| `GET /v1/health` | sub2api 自有 | 渠道详情：账号池内部状态、prompt cache、token 存在性（需鉴权） |

CLIProxyAPI（同时支持 OpenAI 和 Anthropic 上游）可直接把 sub2api 配为上游，无需 SSE 解析。Claude Code 也可直连 `/v1/messages`：

```bash
# Anthropic 原生（Claude Code / Anthropic SDK）
curl http://127.0.0.1:8080/v1/messages \
  -H "Authorization: Bearer sk-sub2api-secret" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-5","max_tokens":1024,"messages":[{"role":"user","content":"hi"}]}'

# Token 计数（Claude Code 会在每轮前调用）
curl http://127.0.0.1:8080/v1/messages/count_tokens \
  -H "Authorization: Bearer sk-sub2api-secret" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-5","messages":[{"role":"user","content":"hi"}]}'
# → {"input_tokens": 8}
```

```bash
# OpenAI Responses API（现代端点，支持 reasoning/background/stream）
curl http://127.0.0.1:8080/v1/responses \
  -H "Authorization: Bearer sk-sub2api-secret" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.6-sol","input":"Plan a 3-day Tokyo itinerary.","stream":false}'

# 流式：SSE 事件 response.created → response.output_text.delta → response.completed
curl http://127.0.0.1:8080/v1/responses \
  -H "Authorization: Bearer sk-sub2api-secret" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.6-sol","input":"count 1 2 3","stream":true}'

# 轮询已创建的 response（后台推理模式）
curl http://127.0.0.1:8080/v1/responses/resp_abc123 \
  -H "Authorization: Bearer sk-sub2api-secret"
```

**注**：归一化事件层（`text.delta`/`thought.delta`/`tool.*`）仅对 `echo`/`adal-cli`/`adal-sdk`/`adal-backend` 等需要翻译的渠道生效；`adal-cloud` 走透传路径时不经过该层。

### 上游兼容层（`adal-cloud` 渠道）

云端代理是一层**路径绑定的 SDK 包装**，对请求体的容忍度远低于官方 API。未经处理直连时
Claude Code 的典型表现是：**一次流式 200，随后一片 HTTP 500**。`sub2api/compat/` 专治这些
差异，每条规则都由真实上游拒绝报文反推得到（见 `compat/profiles.py` 的注释）。

**1. 请求净化（`compat/sanitize.py`）** —— 按协议分流，删除或改写上游 SDK 不接受的字段：

| 协议 | 处理 |
|---|---|
| Anthropic | 删除 `context_management`、`mcp_servers`、`betas`、`stream_options`、`n`、`seed`、`user`、`response_format`、`logit_bias`、`top_logprobs`、`system_prompt`、`max_completion_tokens`、`parallel_tool_calls`（上游报 `unexpected keyword argument`）；删除 `service_tier`、`container`、顶层 `cache_control`（上游报 `Extra inputs are not permitted`）；`metadata` 只保留 `user_id`（`metadata.session_id` 会被拒） |
| Anthropic（`claude-sonnet-5` / `claude-opus-5` / `claude-fable-5-1`） | 这些模型已弃用采样参数：删除 `temperature`/`top_p`/`top_k`；`thinking.type: enabled` 改写为 `adaptive`，且仅在客户端未给 `output_config` 时补 `output_config.effort: high` |
| Anthropic（4-6 家族） | `thinking` 处于 `enabled`/`adaptive` 且 `temperature != 1` 时把 `temperature` 置 1（上游 400：`temperature` may only be set to 1 when thinking is enabled or in adaptive mode） |
| Anthropic（工具） | 未知 `tools[].type` 降级为 `custom` 并补 `input_schema`；允许 `bash_20250124`、`memory_20250818`、`text_editor_20250728`、`tool_search_tool_bm25_20251119`、`tool_search_tool_regex_20251119` 原生透传，前三者额外校验规范工具名（`bash` / `memory` / `str_replace_based_edit_tool`） |
| OpenAI Chat | `max_tokens` → `max_completion_tokens`；删除 `top_p`/`frequency_penalty`/`presence_penalty`/`stop`/`logprobs`/`top_logprobs`；`temperature != 1` 时删除；无 `store` 时删除 `metadata`；`response_format: {"type":"json_object"}` 而 `messages` 里没有 `json` 字样时删除（上游会拒）；带 `tools` 时强制 `reasoning_effort: "none"`（否则 400）；`stream` 时注入 `stream_options.include_usage`（唯一的流式计量途径） |
| Responses | `max_tokens`/`max_completion_tokens` → `max_output_tokens`；`messages` → `input`；删除 `temperature`/`top_p`/`stop` |

**2. 非流式 = 上游流式 + 服务端聚合（`compat/aggregate.py`）** —— 上游对 `max_tokens > 21333`
的非流式 `/v1/messages` 直接硬 500，对请求体问题也只回不透明 500，而**同一请求的流式接口两者
都能正常处理**。因此 `/v1/messages` 的非流式请求一律强制上游 `stream: true`，服务端把 SSE 帧
重组成单个 JSON 信封回给客户端——客户端看到的仍是标准非流式响应，`usage`、`stop_reason`、
`tool_use`、thinking 签名全部保留。实测 `max_tokens: 32000` 非流式 200、正常 `end_turn`、无静默截断。

**3. 错误还原（`compat/errors.py`）** —— 上游把被拒请求伪装成 HTTP 200 里的 `event: error` 帧
（原样转发的话客户端看到的是"成功但空响应"）。sub2api 提取 `error.type` 并还原成官方 API 会给的
状态码：`invalid_request_error`→400、`authentication_error`→401、`permission_error`→403、
`not_found_error`→404、`rate_limit_error`→429、`overloaded_error`→529，无法归类→502。同时该
账号被标记为失败，不会污染池健康度。

**4. 模型不可达 = 原生 404** —— 每个 provider 只在**实测 200** 的路径上可达（如 `zai` 只有
`/v1/messages`，`qwen` 只有 OpenAI 两条），请求打到不支持的组合时按调用方的错误方言回 404，
而不是转发到默认 provider 换回一个不透明 500。provider 优先按**目录查表**解析（原生 id 与
目录 key 都能查到），查不到时才退回**最长前缀**匹配；`gpt-5.6-luna` 这类原生 id 不会被切成
`gpt` 这种不存在的 provider，而是明确返回"未知"，交给端点的原生 provider 兜底。

**5. 缓存亲和（`compat/affinity.py`）** —— 从请求的可缓存前缀算出稳定摘要
（`metadata.user_id` 优先），账号池据此把同一前缀调度回同一账号，命中上游 prompt cache。

### 订阅额度查询（cc-switch 集成）

`adal-cloud` 渠道额外暴露 `GET /v1/usage`，直接回报 AdaL 订阅**当前实际可花的额度**（美元计价的 credits），供 cc-switch 等客户端展示"用量"。试用期账号按**周限额**报数（见下），付费账号按月度额度：

```bash
curl http://127.0.0.1:8080/v1/usage -H "Authorization: Bearer sk-sub2api-secret"
```

```json
{
  "isValid": true,
  "planName": "Pro",
  "total": 20.0,
  "used": 19.828592,
  "remaining": 0.171408,
  "unit": "USD",
  "extra": "trialing · trial weekly cap · resets 2026-09-09T09:24:29Z",
  "sub2api": {
    "channel": "adal-cloud", "pool_enabled": false,
    "accounts": 1, "accounts_resolved": 1, "accounts_parked": 0,
    "queried_at": "2026-09-04T14:32:11Z",
    "requests": 17, "tokens": 21384, "cost_usd": 0.235578,
    "window": "24h", "rate_source": "calibrated",
    "detail": [{
      "ok": true, "email": "…", "tier": "pro", "status": "trialing",
      "limit_basis": "weekly-trial",
      "total": 20.0, "used": 19.828592, "remaining": 0.171408,
      "monthly_total": 80.0, "monthly_used": 19.828592,
      "monthly_remaining": 60.171408
    }]
  }
}
```

机制与语义：

- 账号身份取自各 JWT 的 `sub`（Clerk user id），随后两次 GET 拿到订阅记录——**token 过期或账号被封依然能查额度**（身份看 `sub` 声明，不验签名新鲜度）。
- 查询**以被查账号自己的 token 认证**：`/api/user/by-clerk-id/` 允许匿名，但带上*别人*的 `Authorization` 会返回 403 `Cannot query other users`。这就是多账号池必须逐账号带 token 的原因——共享 client 上残留的旧 header 会把整池额度打成 0。
- **试用账号按周限额报数**。`status: trialing` / `is_trialing: true` 的订阅不允许花掉月度额度：实测同一账号同一周期，订阅接口报 `monthly_credits 80.0 / credits_remaining 60.171408`，而 `/api/credits/balance` 报 `weekly_usage {spent 19.828592, limit 20.0, remaining 0.171408}`——花掉的钱一样，天花板是 20。真正拦请求的是周限额，所以 `total`/`used`/`remaining` 报周限额，月度值保留在 `monthly_*` 里备查，`limit_basis` 标明依据（`weekly-trial` / `monthly`），`extra` 追加 `trial weekly cap`。付费账号不受影响，也**不会**多打这次余额查询。
- 试用账号的余额查询失败或没有周限额（`limit` 为 0/缺失）时**退回月度值**，绝不凭空造一个上限或清零。
- 非试用账号的 `total` 取订阅记录的 `monthly_credits`（**不是**套餐目录里的额定值：试用/改价账号两者会不同）；套餐目录只用于把 `tier` 翻成 `planName`。
- 多账号池模式下，所有**解析成功**的账号额度求和（含已 park 的账号），每个账号按**自己**的依据计入——试用账号计周限额、付费账号计月度额度，`extra` 用 `1/2 on trial weekly cap` 标注比例。"不可用"信号通过 `isValid` / `invalidMessage` 表达，而非把数字清零；`planName` 以 `Pro x2` 形式标注同套餐数量，`extra` 附带 park 原因。
- 聚合结果缓存 30s、套餐目录缓存 600s，账号查询并发上限 8——定时轮询不会放大成上游请求风暴。`?refresh=1` 强制绕过缓存。
- 额度查询**永不抛错**：上游异常降级为 `isValid: false` + `invalidMessage`，不会让端点 500。
- `sub2api.requests` / `tokens` / `cost_usd` 是**本地计量**的滚动 24h 汇总（见「计量与成本」）。未开启计量库时这几个键**不出现**，避免把"没测量"显示成"花了 0"。
- 合并写在 payload 的副本上：渠道会缓存云端额度 30s，原地改会把逐请求明细累积进缓存。
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

- **缓存亲和优先**：同一可缓存前缀回到同一账号（上游 prompt cache 按账号而非会话生效，缓存读比新输入便宜约 10 倍）；绑定超过上游最长缓存 TTL 后失效
- **轮询（round-robin）/ 最少连接（least-connections）**两种兜底调度策略
- 每账号独立**并发上限**，总并发由所有账号上限之和决定
- **健康追踪**：连续失败达阈值后冷却该账号，重复冷却按 `2^n` 指数退避，成功一次即清零
- **额度耗尽感知**：额度同步发现 `is_usage_limited` 的账号直接不参与调度（能认证但每个请求都会因欠额被拒）
- **故障降级**：所有账号都在冷却时 fail-open，选择最早恢复的账号

#### 配置方式

> 不想手写 JSON、也不知道去哪里找 token？开 `--web` 用「[一键导入账号](#一键导入账号只输入一个值)」——粘贴任意含 token 的内容，或走设备码登录，一个输入框搞定，其余字段全部自动推导。

按优先级解析：环境变量 `SUB2API_ACCOUNTS`（JSON）→ SQLite 账号库（`--web` 界面写入的就是这里，只读探测、不会创建库）→ 配置文件 `~/.adal/accounts.json`：

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
| `accounts[].cookies` | — | Clerk cookies，用于 token 过期后自动重铸 bearer |

```bash
# 环境变量方式
SUB2API_API_KEY=sk-sub2api-secret \
  SUB2API_ACCOUNTS='{"strategy":"round-robin","accounts":[{"token":"jwt-a"},{"token":"jwt-b"}]}' \
  python -m sub2api --channel adal-cloud

# 或文件方式
echo '{"accounts":[{"token":"jwt-a"},{"token":"jwt-b"}]}' > ~/.adal/accounts.json
SUB2API_API_KEY=sk-sub2api-secret python -m sub2api --channel adal-cloud
```

未配置账号池时，自动退化为单账号模式（使用 `SUB2API_AUTH_TOKEN` 或 `~/.adal/adal_oauth_creds.json`）。账号池文件变化会在后续模型、聊天或健康请求前安全重载。池状态在**已鉴权**的 `GET /v1/health` 的 `channel.pool` 字段下查看（`healthy_accounts`、`available_capacity`、`models_available`、每账号的 `usage_limited` / `cooldown_strikes` / `dead_reason`）；`/healthz` 只做存活探针，不暴露这些内部信息。

### Prompt Cache 自动注入（`adal-cloud` 渠道）

sub2api 自动为透传请求注入 **prompt cache** 标记，利用上游提供商（Anthropic/OpenAI）的 prefix caching 能力节省额度。缓存由上游提供商管理（TTL 通常 5 分钟~1 小时），sub2api 仅负责注入缓存标记。

| 提供商 | 注入内容 | 机制 |
|---|---|---|
| Anthropic | `cache_control: {type:"ephemeral"}` 注入到 `system` 最后一块 | 上游对 system prompt 做前缀缓存，命中时 `cache_read_input_tokens > 0` |
| OpenAI Responses | `prompt_cache_key: "sub2api"` 注入到请求体 | 上游按 key 缓存前缀，命中时 `cached_tokens > 0` |

**行为**：
- 仅当客户端**未发送** `cache_control` / `prompt_cache_key` 时才注入（幂等，不影响 Claude Code 等原生支持缓存的客户端）
- 字符串格式的 `system` 会被转换为 `[{"type":"text","text":..., "cache_control":{"type":"ephemeral"}}]`
- 缓存命中需要 system prompt ≥ 1024 tokens（Anthropic）或 input ≥ 1024 tokens（OpenAI）；注入本身无长度阈值

`GET /v1/health` 的 `channel.prompt_cache` 字段可查看注入状态。命中需要 system prompt ≥ 1024 tokens（实测 2522 tokens 的提示词：首次 `cache_creation_input_tokens: 2522`，二次 `cache_read_input_tokens: 2522`）。

### 管理界面（`--web`）

`--web`（或 `SUB2API_WEB=1`）在**同一端口**挂载 `/admin` 管理界面——单页 HTML，无外部依赖、无构建步骤。

```bash
SUB2API_API_KEY=sk-sub2api-secret python -m sub2api \
  --channel adal-cloud --port 8080 --web --db ./sub2api.sqlite3
# → http://127.0.0.1:8080/admin
```

| 端点 | 说明 |
|---|---|
| `GET /admin` | 管理页面本身（**不鉴权**：页面不含任何数据，key 在浏览器里输入并存于 `sessionStorage`） |
| `GET /admin/api/accounts` | 账号列表，`token` 只回长度+尾 4 位，`cookies` 只回条数 |
| `POST /admin/api/accounts` | 新增/更新账号（要求 `session_id` + `token`） |
| `POST /admin/api/accounts/paste` | **一键导入**：`{"text": "<任意含 token 的内容>"}`，其余字段全部自动推导 |
| `PATCH /admin/api/accounts/{sid}` | 局部更新（如只改 `max_concurrent`，不会擦掉 token） |
| `DELETE /admin/api/accounts/{sid}` | 删除账号 |
| `POST /admin/api/accounts/import` | 导入 `accounts.json` 格式的批量账号 |
| `GET /admin/api/accounts/export` | 导出，与导入格式**逐字节往返一致** |
| `GET /admin/api/accounts/{sid}/quota` | 查该账号的实时额度 |
| `POST /admin/api/device/start` | 发起设备码登录，返回 `flow_id` / `verification_url` / `user_code`（**不返回 `device_code`**） |
| `POST /admin/api/device/claim` | 轮询一个 `flow_id`；授权完成即自动建账号行 |
| `POST /admin/api/reload` | 重载账号池，无需重启 |

所有 `/admin/api/*` 都要求 `SUB2API_API_KEY`（`x-api-key` 或 `Authorization: Bearer` 均可）。不带 `--web` 时整个 router 不注册——`/admin` 返回 404，而不是"存在但被守卫"。

#### 一键导入账号（只输入一个值）

页面顶部的「一键导入账号」区块只有**一个输入框**，两条路径都不需要填 `session_id`、`cookies`、`email`、`max_concurrent`：

| 方式 | 你要做的 | sub2api 自动做的 |
|---|---|---|
| **粘贴导入** | 把任意含 token 的内容粘进输入框（裸 JWT、整个 `~/.adal/adal_oauth_creds.json`、`accounts.json` 条目、甚至带 `Authorization: Bearer …` 的 curl 命令），点「粘贴导入」 | 按 JWT 形状定位 token（三段 base64url、`eyJ` 开头，且 payload 必须带 Clerk `sub`）→ 生成 `session_id` → 注册会话 → 查 email → 写库 → 重载池 |
| **设备码登录** | 点「开始设备码登录」，在弹出的 AdaL 页面输入显示的 9 位验证码（如 `BHLE-VCXT`） | `POST /api/auth/device/initiate` → 前端每 2.5s 轮询 `claim` → 拿到 token 后同上全套 |

行为细节（均为实测）：

- `session_id` 由服务端生成，格式 `sub2api-pool-<12 位十六进制>`（与 `adal-registrar` 一致）；远端按 `(user, session_id)` upsert，重复无害。
- **重复导入同一账号会更新原行，不会新增一行**：身份取 JWT 的 `sub`（Clerk user id），并保留你手动调过的 `max_concurrent`。
- `device_code` **绝不下发到浏览器**——它等价于一个可换取 bearer 的凭据，只留在服务端内存里（`app.state.device_flows`，单 worker 前提），`expires_in` 实测 600s，过期的 flow 再 claim 返回 410。
- 设备码的 `verification_url`（`/auth/device-verify?adal_surface=cli`）实测会 **302 到 `/sign-in`**，只提供"邮箱+密码"或 Google 登录。也就是说设备码路径要求你**这台浏览器已经登录过 AdaL**（或知道账号密码）；否则请走粘贴导入——这也是推荐路径。
- 会话注册失败（如上游 403）时**仍然保存 token**，但行标记为 `status=unknown` / `reason=session_unregistered`，并在响应里回 `detail`——不会假装账号可用。
- 这两条路径写出的行**没有 cookies**，因此无法通过 Clerk 重新铸造 token。这是可接受的：AdaL 边缘校验的是 Clerk 会话（JWT 的 `sid`），不是 `exp`——实测一个已过期 2 天的设备码 token 依然能跑通 `/proxy/v1/messages`、`/api/client-sessions/start` 与 `/api/credits/balance`。想要可续期的行，就粘贴带 `cookies` 的 `accounts.json` 条目，粘贴导入会一并收下。
- 导入成功立即 `channel.refresh()`，无需再点「reload pool」；刷新失败也不会丢弃已落库的账号（响应里 `pool_reloaded: false`）。
- 导入后的额度查询**以新账号自己的 token 认证**：`/api/user/by-clerk-id/` 对携带别人 bearer 的请求返回 403 `Cannot query other users`，因此单账号→池的切换会先清掉共享 client 上的旧 `Authorization`。这也解释了为什么行内「quota」按钮（只认 bearer）一直是对的，而 `/v1/usage` 曾显示 0。

「accounts」区块里原来的手填表单收进了折叠的「手动添加 / 更新（需自备 token）」，只在你确实要指定 `session_id` 时才用得上。

### 计量与成本

指定 `--db`（或 `SUB2API_DB`）后，每个透传请求写一行 `usage_events`，费率按实测校准，
`GET /v1/usage` 汇总滚动 24h。计量库在 lifespan 启动时打开——构造 app 对象（跑测试、`--help`）
不会创建数据库文件；打不开就降级为不计量，不会拒绝服务。

| 表 | 内容 |
|---|---|
| `usage_events` | `ts, session_id, model, provider, path, stream, status, latency_ms, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens, cost_usd, rate_estimated` |
| `credit_snapshots` | 额度同步快照，主键 `(session_id, ts)`：`total, monthly_allocation, monthly_used, weekly_spent, weekly_limit, weekly_remaining, is_usage_limited, limit_reason, resets_at` |
| `accounts` | 账号池（`--web` 界面读写的就是这张表） |
| `pool_settings` | 池级设置的 `k`/`v` 键值表（`strategy` / `max_failures` / `cooldown_seconds`） |

费率（USD / 1M tokens）由**差分真实额度**测得，已含各模型促销折扣，不从目录倍率反推：

| 模型 | input | output | cache write | cache read |
|---|---|---|---|---|
| `claude-sonnet-4-6` / `claude-sonnet-5` | 3.00 | 15.00 | 3.75 | 0.30 |
| `claude-opus-4-6` / `claude-opus-5` | 5.00 | 25.00 | 6.25 | 0.50 |
| `claude-fable-5-1` | 7.50 | 37.50 | 9.375 | 0.75 |
| `gpt-5.6-terra` | 2.00 | 12.00 | 2.00 | 0.20 |
| `gpt-5.6-luna` | 0.20 | 1.20 | 0.20 | 0.02 |
| `gpt-5.6-sol` | 4.00 | 24.00 | 4.00 | 0.40 |

两种计费惯例都实测过，按协议分别套用：

- **Anthropic**：`input_tokens` **不含**缓存读和缓存写，四类 token 各按自己的费率计
  → `in*input + out*output + cw*cache_write + cr*cache_read`
- **OpenAI（chat / responses）**：`prompt_tokens` **含**缓存读
  → `max(in-cr,0)*input + cr*cache_read + out*output`

表中没有的模型退化为该 provider 的中档费率，并在该行标记 `rate_estimated = 1`。
额度同步任务每 600s 拉一次各账号余额（**启动即拉一次**，这样"额度耗尽"在第一个请求前就已知，
而不是靠烧掉一个请求发现），写入 `credit_snapshots` 并更新池内 `usage_limited`。

### 命令行

```bash
python -m sub2api --help
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--host` | `SUB2API_HOST` 或 `127.0.0.1` | 监听地址 |
| `--port` | `SUB2API_PORT` 或 `8080` | 监听端口 |
| `--channel` | `SUB2API_CHANNEL` 或 `echo` | 激活渠道 |
| `--web` | 关 | 挂载 `/admin` 管理界面 |
| `--db PATH` | `SUB2API_DB` 或 `~/.adal/sub2api.sqlite3` | SQLite 路径；同时导出 `SUB2API_DB`，让计量库与账号来源始终是同一个文件 |

**默认拒绝匿名启动**：没有 `SUB2API_API_KEY` 时进程直接退出（exit 2）并提示——
一个不设 key 的网关等于把订阅额度（开了 `--web` 还包括凭证）交给任何能连上端口的人。
确实要开放时显式设 `SUB2API_ALLOW_ANONYMOUS=1`。该检查只在启动监听时生效，构造 app 对象不受影响。


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
  -d '{"model":"gpt-5.6-terra","messages":[{"role":"user","content":"hi"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-sub2api-secret")
r = client.chat.completions.create(
    model="gpt-5.6-terra",
    messages=[{"role": "user", "content": "hi"}],
)
# 流式
stream = client.chat.completions.create(
    model="gpt-5.6-terra", messages=[{"role":"user","content":"count 1 2 3"}], stream=True)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

**Anthropic 客户端直连**（Claude Code / Anthropic SDK）：

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_API_KEY=sk-sub2api-secret
claude   # Claude Code 直连 sub2api /v1/messages
```

直连时 `model` 直接填官方原生 id（如 `gpt-5.6-terra`、`claude-sonnet-5`）——`/v1/models` 返回的就是这些原生 id，与官方 API 完全一致，为官方端点写的客户端配置无需任何改写。AdaL 目录 key（如 `openai-gpt-5.6-terra`）作为别名继续兼容，只是不再出现在 `/v1/models` 里；`chatgpt_web-*` 三个模型不再对外暴露（与 `openai-*` 同主机同模型，且是唯一的 id 冲突来源），但显式填 `chatgpt_web-gpt-5.6-luna` 这类 key 仍可路由。

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
      - "sk-sub2api-secret"                    # 与 SUB2API_API_KEY 一致（该值必填，sub2api 默认拒绝匿名启动）
    models:
      # name = sub2api /v1/models 返回的 id，已与官方原生 id 一致，
      # 因此不再需要 alias 改名；仅当想对客户端换个名字时才加 alias。
      # 完整列表见 GET /v1/models；下面是全部 28 个可达模型
      - name: "claude-sonnet-5"
      - name: "claude-sonnet-4-6"
      - name: "claude-opus-5"
      - name: "claude-opus-4-6"
      - name: "claude-fable-5-1"
      - name: "gpt-5.6-terra"
      - name: "gpt-5.6-luna"
      - name: "gpt-5.6-sol"
      - name: "glm-5.3-flash"
      - name: "glm-5.3"
      - name: "glm-5.2"
      - name: "glm-5.1"
      - name: "deepseek-v4-flash"
      - name: "deepseek-v4-flash-vision-exp"
      - name: "deepseek-v4-pro"
      - name: "kimi-k3"
      - name: "kimi-k2.7-code"
      - name: "MiniMax-M2.7"
      - name: "MiniMax-M3"
      - name: "grok-4.6"
      - name: "grok-4.5"
      - name: "qwen3.8-flash"
      - name: "qwen3.8-max"
      - name: "qwen3.7-max"
      - name: "qwen3.7-plus"
      - name: "muse-spark-1.3"
      - name: "muse-spark-1.2"
      - name: "muse-spark-1.1"
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
      - name: "claude-sonnet-5"                 # 与官方原生 id 一致，无需 alias
      - name: "claude-opus-5"
      - name: "claude-sonnet-4-6"
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
      - name: "claude-sonnet-5"
      - name: "gpt-5.6-terra"

claude-api-key:
  - api-key: "sk-sub2api-secret"
    base-url: "http://127.0.0.1:8080"
    cloak:
      mode: "never"
    models:
      - name: "claude-sonnet-5"
```

#### 可用模型清单

`GET /v1/models` 返回 28 个**已验证可达**的模型，id 与官方原生 id 完全一致（如 `claude-sonnet-5`、`gpt-5.6-terra`）。配置 CLIProxyAPI 时 `name` 字段填这些 id，`alias` 仅在想换名时才需要。"可达路径"列是该 provider 实测 200 的接口，请求打到其他组合会得到原生 404：

| Provider | X-Target-URL | 可达路径 | 模型 id |
|---|---|---|---|
| anthropic | `api.anthropic.com` | `/v1/messages` | `claude-sonnet-5`, `claude-sonnet-4-6`, `claude-opus-5`, `claude-opus-4-6`, `claude-fable-5-1` |
| openai | `api.openai.com` | `/v1/chat/completions`, `/v1/responses` | `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.6-sol` |
| zai | `api.z.ai/api/anthropic` | `/v1/messages` | `glm-5.3-flash`, `glm-5.3`, `glm-5.2`, `glm-5.1` |
| deepseek | `api.deepseek.com` | `/v1/chat/completions`, `/v1/responses` | `deepseek-v4-flash`, `deepseek-v4-flash-vision-exp`, `deepseek-v4-pro` |
| kimi | `api.moonshot.ai` | `/v1/chat/completions`, `/v1/responses` | `kimi-k3`, `kimi-k2.7-code` |
| minimax | `api.minimax.io/anthropic` | `/v1/messages` | `MiniMax-M2.7`, `MiniMax-M3` |
| xai | `api.x.ai` | `/v1/messages`, `/v1/chat/completions`, `/v1/responses` | `grok-4.6`, `grok-4.5` |
| qwen | `dashscope-intl.aliyuncs.com/compatible-mode` | `/v1/chat/completions`, `/v1/responses` | `qwen3.8-flash`, `qwen3.8-max`, `qwen3.7-max`, `qwen3.7-plus` |
| meta | `api.meta.ai` | `/v1/chat/completions`, `/v1/responses` | `muse-spark-1.3`, `muse-spark-1.2`, `muse-spark-1.1` |

**目录 key 仍兼容**：AdaL 目录里每个条目的 key 是 `<provider>-<model_id>`（如 `openai-gpt-5.6-terra`）。这些 key 作为别名继续被接受并重写为原生 id，只是不再出现在 `/v1/models` 里。

**`chatgpt_web` 已不再暴露**：它把 `openai` 的三个模型（`gpt-5.6-terra/luna/sol`）以第二套 key 重新导出，目标主机与路径都与 `openai` 相同，是原生 id 的唯一冲突来源；改暴露原生 id 后必须择一，保留走官方 API key 的 `openai`。显式填 `chatgpt_web-gpt-5.6-luna` 这类完整 key 仍可路由。

**暂不支持**：目录里的 5 个 `google-gemini-*` 不在上表内，也不会出现在 `/v1/models`。它们的
OpenAI 兼容层在云端代理上是死路（每种参数组合都回 403 `model_not_allowed`），只有原生
`generativelanguage.googleapis.com` 的 `:generateContent` / `:streamGenerateContent?alt=sse` /
`:countTokens` 可用（实测 200），接入需要一层 Anthropic/OpenAI ⇄ Gemini 协议翻译，尚未实现。
与其列出去必然 500，不如直接不列——请求这些 model 会得到原生 404。

#### 字段对照与排错

| CLIProxyAPI 字段 | 值 | 说明 |
|---|---|---|
| `base-url`（openai-compatibility） | `http://127.0.0.1:8080/v1` | sub2api 地址 + `/v1` |
| `base-url`（claude-api-key） | `http://127.0.0.1:8080` | sub2api 根地址，CLIProxyAPI 自动追加 `/v1/messages` |
| `api-keys` / `api-key` | 与 `SUB2API_API_KEY` 一致 | 必填：sub2api 默认拒绝无 key 启动（只有显式 `SUB2API_ALLOW_ANONYMOUS=1` 时才可留空） |
| `models[].name` | sub2api `/v1/models` 里的 `id` | 必须完全匹配 |
| `models[].alias` | 自定义 | 客户端请求时用的名字 |
| `cloak.mode`（claude-api-key） | `"never"` | sub2api 透传不做伪装，关掉避免改写 |

常见问题：

- **401 invalid api key**：CLIProxyAPI 的 `api-keys`/`api-key` 与 sub2api 的 `SUB2API_API_KEY` 不一致。
- **模型不在列表**：CLIProxyAPI `models[].name` 拼写与 `/v1/models` 返回的 `id` 不符；用 `curl /v1/models` 核对。
- **`auth_unavailable: no auth available`**：CLIProxyAPI 的 `openai-compatibility.models[]` 里**没有配客户端请求的 model**。CLIProxyAPI 按客户端发的 `model` 匹配 `models[].name` 或 `alias`，匹配不到就报此错。解决：把所有要用的模型都加到 `models[]`（见上方完整 28 个模型配置），或确保客户端请求的 model 名与 `alias` 一致。
- **`/v1/messages` 走了 OpenAI 上游**：`claude-api-key` 的 `base-url` 不要带 `/v1`，CLIProxyAPI 自己追加 `/v1/messages`。
- **404 `{"detail":"Not Found"}`**：`base-url` 多带了 `/v1` 导致路径变成 `/v1/v1/messages`。`claude-api-key` 填根地址 `http://127.0.0.1:8080`（不带 `/v1`）。若无法改 CLIProxyAPI 配置，sub2api 也兼容 `/v1/v1/messages` 和 `/v1/v1/chat/completions` 别名。
- **400 `max_tokens` not supported**：OpenAI 新模型（gpt-5.6-*）只认 `max_completion_tokens`，Responses API 只认 `max_output_tokens`。sub2api 已自动按协议改写，CLIProxyAPI 侧无需改动。
- **Claude Code 一次成功后连续 500**：这是未经净化直连云端代理的症状（`context_management` 等字段触发 `unexpected keyword argument`）。sub2api 的兼容层已处理，若仍出现请确认请求确实经过 sub2api 而非直连 `api.adal.sylph.ai`。
- **非流式 `/v1/messages` 在大 `max_tokens` 下 500**：上游对 `max_tokens > 21333` 的非流式请求硬 500。sub2api 改走"上游流式 + 服务端聚合"，客户端无需改动。
- **某个 model 返回 404 `model not found`**：该 provider 不在这条路径上提供服务（如 `zai` 只有 `/v1/messages`），或请求的是 `google-*`。用 `curl /v1/models` 核对可达清单。
- **流式中断**：确认 sub2api 进程存活（`curl /healthz`），且 CLIProxyAPI 与 sub2api 同机或网络可达。

### API

| 端点 | 鉴权 | 说明 |
|---|---|---|
| `POST /v1/messages` | ✔ | Anthropic Messages（`adal-cloud` 透传） |
| `POST /v1/messages/count_tokens` | ✔ | Anthropic Token Counting，回 `{"input_tokens": N}` |
| `POST /v1/chat/completions` | ✔ | OpenAI Chat Completions |
| `POST /v1/responses` | ✔ | OpenAI Responses API |
| `GET` / `DELETE /v1/responses/{id}` | ✔ | 获取/删除已创建的 response 对象 |
| `GET /v1/models` | ✔ | 可达模型列表（OpenAI 格式） |
| `POST /v1/chat` | ✔ | 归一化同步对话，返回 `{answer, session_id, channel, model}` |
| `POST /v1/chat/stream` | ✔ | 归一化 SSE 流式，逐帧输出事件 |
| `GET /v1/channels` | ✔ | 已注册渠道列表 + 当前渠道健康状态 |
| `GET /v1/sessions/{id}` | ✔ | 会话详情（轮数、native id、模型） |
| `GET /v1/usage` | ✔ | 订阅额度 + 本地 24h 计量，`?refresh=1` 绕过缓存 |
| `GET /v1/health` | ✔ | 渠道详情：账号池内部、prompt cache、token 存在性 |
| `GET /healthz` | ✘ | 存活探针，只回 `{"status","version"}` |
| `/admin`, `/admin/api/*` | 页面✘ / API✔ | 管理界面（仅 `--web`，见「管理界面」） |

鉴权接受两种写法（真实客户端两派都有）：Anthropic SDK 与 Claude Code 发 `x-api-key`，
OpenAI 兼容客户端发 `Authorization: Bearer`。比较用常量时间，避免 `!=` 通过时序泄露前缀。

兼容别名（给 base-url 里多带了 `/v1` 或省掉 `/v1` 的客户端）：
`/v1/v1/messages`、`/v1/v1/chat/completions`、`/v1/v1/responses`、`/v1/v1/usage`、
`/v1/completions`、`/models`、`/usage`。

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
| `SUB2API_API_KEY` | — | 全部 `/v1/*`、`/admin/api/*` 要求 `x-api-key` 或 `Authorization: Bearer`（`/healthz` 与 `/admin` 页面除外）。**未设置时进程拒绝启动** |
| `SUB2API_ALLOW_ANONYMOUS` | — | 设为 `1` 才允许无 key 启动（明确放弃鉴权） |
| `SUB2API_WEB` | — | 设为 `1` 挂载 `/admin` 管理界面，等价于 `--web` |
| `SUB2API_DB` | `~/.adal/sub2api.sqlite3` | SQLite 路径：计量库 + 账号库，等价于 `--db` |
| `SUB2API_PROXY` | — | 出网 HTTP(S) 代理，作用于所有对上游的 `httpx` 连接 |
| `SUB2API_ACCOUNTS` | — | JSON，多账号池配置（详见「多账号池」章节）；也可用 SQLite 或 `~/.adal/accounts.json` |
| `SUB2API_AUTH_TOKEN` | — | 显式 JWT（如 AdaL 的 `access_token`），CI 无浏览器时用 |
| `SUB2API_WORKSPACE` | `.` | 渠道默认工作目录 |
| `SUB2API_OPENAI_PERMISSION_MODE` | `yolo` | OpenAI 兼容路由（归一化路径）使用的权限模式 |
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
pytest -q          # 464 个用例：契约、注册表、会话、聚合、兼容层净化/聚合/错误还原、路由表（含原生 id 暴露）、费率、计量库、账号池、四个 AdaL 渠道的解析/参数/子进程端到端、HTTP 全表面、OpenAI 兼容层、订阅额度（含试用周限额）、管理界面（含一键导入/设备码流）、CLI
```

无需安装 AdaL 即可跑全部测试——AdaL 渠道用假运行时（临时启动器脚本）做子进程级验证，SDK 渠道测纯映射函数。

## 许可

MIT
