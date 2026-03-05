# RealTimeLister

会议实时翻译工具：

- 本地 ASR：`faster-whisper`
- 在线翻译：公司 LLM（例如 `gpt-5.1`）
- Web UI：麦克风选择、伪说话人标签、自定义翻译 Prompt

---

## 0. 推荐流程（按这个顺序）

1. 先准备 ASR 模型（外网机器）
2. 再安装项目（公司机器）
3. 启动前先配置 `.env`
4. 最后运行与日常使用

---

## 1. 先准备 ASR 模型（外网机器）

> 如果公司机器不能访问 Hugging Face，先在外网机器下载模型再拷贝到公司机器。

```powershell
cd RealTimeLister
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_faster_whisper_model.py --model small --output-dir C:/temp/faster-whisper-small
```

把下载目录拷到公司机器，例如：

```text
C:/models/faster-whisper-small
```

目录至少包含：

- `config.json`
- `model.bin`
- `tokenizer.json`

---

## 2. 再安装项目（公司机器）

```powershell
cd RealTimeLister
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

说明：全流程都可以不激活虚拟环境，不依赖 `Activate.ps1`。

---

## 3. 启动前先配置 `.env`

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

2. 否则，`RT_ASR_HF_LOCAL_ONLY=true`：
离线缓存模式，只读缓存，不下载。

3. 否则：
模型名模式（`RT_ASR_MODEL_NAME`），先读缓存，缺失时再尝试下载。

---

## 4. 最后运行与用法

### 4.1 Web UI（默认）

```powershell
.\.venv\Scripts\python.exe -m realtime_lister.main
```

默认地址：`http://127.0.0.1:8080`

### 4.2 终端模式

```powershell
.\.venv\Scripts\python.exe -m realtime_lister.main --terminal
```

### 4.3 常用参数

```powershell
.\.venv\Scripts\python.exe -m realtime_lister.main --source-language auto --target-language en --asr-model small --input-device auto --translation-model gpt-5.1 --host 127.0.0.1 --port 8080
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

- `ASR Strategy`：当前到底走固定目录/离线缓存/联网自动
- `ASR Readiness`：模型可用性，错误会直接写原因

日常交互：

- `Audio Input`：可用麦克风列表
- `Translation Prompt`：可改模板并点 `Save Prompt`
- `History`：显示 `Speaker N` 伪说话人标签

