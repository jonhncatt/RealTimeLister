# RealTimeLister

会议实时翻译工具：

- 本地 ASR：`faster-whisper`
- 在线翻译：公司 LLM（例如 `gpt-5.1`）
- Web UI：麦克风选择、伪说话人标签、自定义翻译 Prompt

---

## 0. 推荐流程（按这个顺序）

1. 安装项目
2. 直接运行 CLI
3. 首次启动时让 CLI 自动检查/下载 ASR 模型
4. 如果要翻译，再补 `.env` 里的 OpenAI 配置
5. 用 `start` 或 `terminal` 开始使用

---

## 1. ASR 模型默认放在仓库内 `model/`

默认目录：

```text
RealTimeLister/model/faster-whisper-small
```

不需要先下载到别的临时目录，再手工 copy 到其他地方。CLI 首次运行会优先检查这个目录；如果没有模型，会直接询问是否下载到这里。

如果你在外网机器上手动下载，也建议直接下到仓库里的 `model/`：

```powershell
cd RealTimeLister
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_faster_whisper_model.py --model small --output-dir .\model\faster-whisper-small
```

如果公司机器完全离线，把整个仓库带过去，或者至少把这个目录带过去：

```text
RealTimeLister/model/faster-whisper-small
```

目录至少包含：

- `config.json`
- `model.bin`
- `tokenizer.json`

---

## 2. 安装项目

```powershell
cd RealTimeLister
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

说明：全流程都可以不激活虚拟环境，不依赖 `Activate.ps1`。

完成这一步之后，就可以直接用命令 `realtime` 启动。

---

## 3. 如果要翻译，再配置 `.env`

### 3.1 先复制模板（只选一个）

```powershell
copy env.offline-model.example .env
```

- `env.offline-model.example`：公司机器不能访问 HF（推荐）
- `env.online.example`：公司机器可访问 HF
- `env.hf-mirror.example`：走公司 HF 镜像

### 3.2 最小必填

```env
OPENAI_API_KEY=your_key
RT_TRANSLATION_MODEL=gpt-5.1
RT_SOURCE_LANGUAGE=auto
RT_TARGET_LANGUAGE=en
RT_AUDIO_INPUT_DEVICE=auto
```

如果你只想先验证本地 ASR，可以先不填 `OPENAI_API_KEY`。此时会以 `ASR only` 模式运行。

### 3.3 翻译 Prompt 模板（可选）

```env
RT_TRANSLATION_PROMPT_TEMPLATE=You are a professional meeting interpreter.\nTranslate from {source_language} to {target_language}.\nSpeaker context: {speaker_label}.\nKeep original meaning, names, and technical terms.\nDo not add explanations.{glossary_block}
```

可用占位符：

- `{source_language}`
- `{target_language}`
- `{speaker_label}`
- `{speaker_id}`
- `{glossary}`
- `{glossary_block}`

### 3.4 ASR 选择规则（统一口径）

1. `RT_ASR_MODEL_DIR` 有值：
固定目录模式（最高优先级），不走 HF 自动下载。

2. 否则，如果仓库里的默认目录存在完整模型，例如 `model/faster-whisper-small`：
自动按固定目录模式使用它。

3. 否则，`RT_ASR_HF_LOCAL_ONLY=true`：
离线缓存模式，只读缓存，不下载。

4. 否则：
模型名模式（`RT_ASR_MODEL_NAME`），先读缓存，缺失时再尝试下载。

---

## 4. 运行与用法

### 4.1 交互式 CLI（默认入口）

```powershell
realtime
```

首次运行时会进入 CLI，流程是：

1. 自动检查仓库内 `model/` 里有没有 ASR 模型
2. 如果没有，询问是否现在下载
3. 确认下载目录，默认就是仓库内 `model/faster-whisper-small`
4. 下载完成后，询问是否立即启动 Web UI

进入 CLI 后可用命令：

- `help`
- `status`
- `setup`
- `download`
- `start`
- `terminal`
- `quit`

### 4.2 直接启动 Web UI

```powershell
realtime --web
```

默认地址：`http://127.0.0.1:8080`

### 4.3 终端模式

```powershell
realtime --terminal
```

### 4.4 常用参数

```powershell
realtime --interactive --source-language auto --target-language en --asr-model small --input-device auto --translation-model gpt-5.1 --host 127.0.0.1 --port 8080
```

---

## 5. 关于 `RT_SOURCE_LANGUAGE`：为什么还要设置

Whisper 确实支持自动识别语言，所以现在支持：

- `RT_SOURCE_LANGUAGE=auto`：自动识别（默认推荐）
- `RT_SOURCE_LANGUAGE=zh` / `en`：固定语言

为什么仍然保留“可设置固定语言”？

- 固定语言通常更稳、更快一点
- 多语言/口音混合时，自动识别有时会抖动
- 会议如果已知基本只用一种语言，固定可减少误判

建议：

1. 先用 `auto`
2. 如果你们会议语言很固定且出现误检，再改成固定 `zh` 或 `en`

---

## 6. 页面上看哪里

启动后先看：

- `Ready Check`：直接告诉你模型、麦克风、翻译器是否就绪，以及下一步该做什么
- `ASR Strategy`：当前到底走固定目录/离线缓存/联网自动

日常交互：

- `Source Language` / `Target Language`：基础方向配置，停机时可直接改
- `Audio Input`：可用麦克风列表
- `Advanced Settings`：Prompt 模板等不常改的项都收在这里
- `History`：显示 `Speaker N` 伪说话人标签
