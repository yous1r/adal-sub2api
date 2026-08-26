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
└───┬───────────────┬───────────────┬───────────┘
    │               │               │
 echo           adal-cli        adal-sdk
(参考实现)   (子进程+NDJSON)  (Python SDK)
```

**设计原则**：共性下沉到 `core`（请求/事件契约、会话存储、错误分类、聚合、SSE 编码、生命周期与错误归一化管线）；渠道只实现差异部分（运行时探测、参数构造、原始事件 → 归一化事件的翻译）。

## 快速开始

```bash
# 安装依赖（任选其一）
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # Windows
python -m venv .venv && .venv/bin/pip install -e ".[dev]"       # macOS/Linux

# 启动（默认 echo 渠道，无需任何外部依赖）
sub2api --port 8080
# 或: python -m sub2api --port 8080

# 切换到 AdaL 渠道（需先安装并登录 AdaL CLI：adal）
SUB2API_CHANNEL=adal-cli python -m sub2api
```

## OpenAI 兼容接口（可接入 cliproxyapi 等聚合器）

`sub2api` 同时暴露标准 OpenAI Chat Completions 协议，任何支持自定义 `base_url` 的客户端/网关都能直接把它当 OpenAI 上游使用：

| 端点 | 说明 |
|---|---|
| `POST /v1/chat/completions` | 兼容 `messages` / `model` / `stream`；`stream: true` 时输出 `chat.completion.chunk` SSE，以 `data: [DONE]` 结束 |
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

### cliproxyapi 接入示例

cliproxyapi 侧把 sub2api 配为一个 OpenAI 兼容上游即可：

```yaml
openai-compatibility:
  - name: sub2api-adal
    base-url: http://127.0.0.1:8080/v1
    api-keys: ["<SUB2API_API_KEY>"]   # 未设置 SUB2API_API_KEY 时留空
    models:
      - name: "claude-sonnet-4-6"
```

任何 OpenAI SDK 同样直连：

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
client.chat.completions.create(model="claude-sonnet-4-6",
                               messages=[{"role": "user", "content": "hi"}])
```

### API

| 端点 | 说明 |
|---|---|
| `POST /v1/chat` | 同步对话，返回 `{answer, session_id, channel, model}` |
| `POST /v1/chat/stream` | SSE 流式，逐帧输出归一化事件 |
| `GET /v1/channels` | 已注册渠道列表 + 当前渠道健康状态 |
| `GET /v1/sessions/{id}` | 会话详情（轮数、native id、模型） |
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
