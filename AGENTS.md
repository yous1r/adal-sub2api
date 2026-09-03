# adal-sub2api — 安全任务技能路由注册（手工注册）

本文件把外部技能包 **reverse-skill**（安全任务技能路由：逆向工程 / 渗透测试 / 安全分析）
注册到本项目，打通 AI 客户端集成的最后一环。reverse-skill 核心保持客户端中立、不写任何
客户端全局配置；本文件即项目级注入点，对 Claude Code / Codex CLI / Cursor / Cline /
Windsurf / ZCode 等所有读取 `AGENTS.md` 的客户端等效生效。

- 技能包根目录（`<SKILL_ROOT>`）：`D:\Packages\reverse-skill`
- 本项目即**分析项目**：路由与 case 产物默认落在**本项目** `work\<case>\`
  （脚本按调用时 CWD 解析项目根，或显式传 `-ProjectRoot`）。
- case 产物目录 `work/` 已加入 `.gitignore`，证据不得提交进 git。

## 触发与路由

处理本项目任务时，命中安全/逆向关键词（APK、反编译、IDA、抓包、JS 逆向、加密参数、
渗透测试、漏洞利用、CTF、固件、EDR、malware、逆向工程 等，中英双语全表见
`<SKILL_ROOT>\RULES.md`）时，先路由再动手：

1. PRIMARY 路由（唯一事实源：`<SKILL_ROOT>\skills\config\routing.json`）：

   ```text
   powershell -NoProfile -ExecutionPolicy Bypass -File "D:\Packages\reverse-skill\skills\scripts\master-route.ps1" -Hint "<任务>"
   ```

   在本项目根目录执行（产物随调用方项目走），或显式追加
   `-ProjectRoot "D:\Packages\adal-sub2api"`。

2. PRIMARY 歧义时读 `<SKILL_ROOT>\skills\routing.md`（三轴咨询矩阵，不是第二路由器）。
3. 入口文件（按需读取，勿预载）：`<SKILL_ROOT>\skills\SKILL.md` → 对应子 skill 的 `SKILL.md`。
4. 路由未命中 → 不要硬套现有 skill；按 RULES.md 提议新增 skill（改 routing.json + benchmark）。

## 授权门禁（硬性）

对任何目标动手前，先初始化 case scope（产物落在本项目 `work\<case>\`）：

```text
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\Packages\reverse-skill\skills\scripts\case-init.ps1" -Hint "<任务>" [-CaseName <名称>] [-Preset offline-sample -Sample <样本路径>]
```

- `auth.status=granted` + 合法 `network_profile`，或显式授权的 offline sample 就绪，才可进入 ACT。
- 本地离线样本使用 `offline-sample` preset；仅提到目标 ≠ 已授权。
- `case-guard --force` / `-Force` **不得**绕过该硬门。
- 证据链：`<SKILL_ROOT>\skills\ops\evidence-finding-path.md`；角色：`skills\ops\role-map.md`；身份：`skills\ops\IDENTITY.md`。
- 报告交接前复核：`python "D:\Packages\reverse-skill\skills\case-review\scripts\review_case.py" work\<case> --verify-hashes --strict`，逐项解决 error。

## 工具

- 工具索引已生成（2026-08-31）：`<SKILL_ROOT>\skills\tool-index.md` / `tool-index.json`
  —— 工具可用性唯一事实源，**禁止猜路径**。
- 缺工具 → 按需 bootstrap（只装"需要且缺失"的，能力名必须来自 bootstrap-manifest.json），
  装完必须刷新索引：

  ```text
  powershell -NoProfile -ExecutionPolicy Bypass -File "D:\Packages\reverse-skill\skills\scripts\bootstrap-reverse.ps1" -Capability <名称>
  powershell -NoProfile -ExecutionPolicy Bypass -File "D:\Packages\reverse-skill\skills\scripts\refresh-tool-index.ps1"
  ```

- 同一工具自动安装失败 2 次 → 停止重试，输出完整手动安装步骤。
- 经验复用：进入任何 route 前先查 `<SKILL_ROOT>\skills\field-journal\_index.md`。

## 迁移

若 reverse-skill 包移动位置，同步更新本文件内全部 `<SKILL_ROOT>` 绝对路径。
