# RealTimeLister

本项目用于会议实时翻译：

- 语音识别（ASR）在本地运行（`faster-whisper`）
- 翻译调用公司 LLM（例如 `gpt-5.1`，OpenAI 兼容接口）

这样即使公司没有 `transcribe` API，也可以实现实时字幕翻译。

## 1. Windows 安装

建议 Python 3.10 - 3.12。

```powershell
cd RealTimeLister
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
copy .env.example .env
```

## 2. 配置 `.env`

推荐不要直接从零手写，先按场景复制模板：

```powershell
copy env.offline-model.example .env
```

可用模板：

- `env.online.example`：公司机器能访问 Hugging Face，首次运行自动下载
- `env.offline-model.example`：公司机器不能访问 Hugging Face，使用提前拷贝好的本地模型
- `env.hf-mirror.example`：公司内部有 Hugging Face 镜像

如果你还是想手动改，至少配置：

```env
OPENAI_API_KEY=your_key
RT_TRANSLATION_MODEL=gpt-5.1
RT_SOURCE_LANGUAGE=zh
RT_TARGET_LANGUAGE=en
RT_ASR_MODEL_NAME=small
```

如果走公司网关，再配置：

```env
OFFICETOOL_OPENAI_BASE_URL=https://your-company-gateway/v1
OFFICETOOL_CA_CERT_PATH=C:/path/to/company-root-ca.pem
OFFICETOOL_USE_RESPONSES_API=false
```

说明：程序会优先按 `OFFICETOOL_USE_RESPONSES_API` 调用；如果网关返回 `405`，会自动切换 `responses/chat-completions` 方式重试。

## 2.1 ASR 模型规则

ASR 模型加载优先级如下：

1. 如果配置了 `RT_ASR_MODEL_DIR`，优先使用这个本地目录
2. 如果没有配置 `RT_ASR_MODEL_DIR`，则读取 `RT_ASR_MODEL_NAME`
3. 如果本地缓存里没有 `RT_ASR_MODEL_NAME` 对应模型，`faster-whisper` 会尝试从 Hugging Face 下载

注意：

- 下载完成后，程序不会自动把 `.env` 改写成 `RT_ASR_MODEL_DIR=...`
- 如果你后续仍然只配置 `RT_ASR_MODEL_NAME=small`，程序会继续按“型号名 + 本地缓存”方式工作
- 如果你希望完全固定使用某个本地目录，仍然应该手动配置 `RT_ASR_MODEL_DIR`
- 旧变量名仍兼容，例如 `RT_WHISPER_MODEL`、`RT_WHISPER_MODEL_PATH`

## 2.2 Hugging Face / 本地模型说明

默认情况下，`faster-whisper` 会按 `RT_ASR_MODEL_NAME` 从 Hugging Face 拉取模型。

如果公司网络不能访问 Hugging Face，优先用这两种方式：

1. 直接指定本地模型目录

```env
RT_ASR_MODEL_DIR=C:/models/faster-whisper-small
```

注意：这里必须是 `faster-whisper/CTranslate2` 转换后的模型目录，不是原始 Whisper 的 PyTorch 权重目录。

2. 只使用本地缓存，不允许联网下载

```env
RT_ASR_HF_CACHE_DIR=C:/hf-cache
RT_ASR_HF_LOCAL_ONLY=true
```

如果公司内部有 Hugging Face 镜像，也可以配置：

```env
RT_ASR_HF_ENDPOINT=https://your-hf-mirror
```

如果公司代理需要自签 CA，请配置：

```env
OFFICETOOL_CA_CERT_PATH=C:/path/to/company-root-ca.pem
```

这项配置现在会同时用于公司 LLM 网关和 Hugging Face 模型下载。

## 2.3 不能访问 Hugging Face 时，如何在外网机器下载

仓库里加了一个下载脚本：

[`scripts/download_faster_whisper_model.py`](/Users/zhoudali/Desktop/RealTimeLister/scripts/download_faster_whisper_model.py)

在一台能访问外网的机器上执行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_faster_whisper_model.py --model small --output-dir C:/temp/faster-whisper-small
```

如果你是在 macOS/Linux 外网机器上准备模型：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_faster_whisper_model.py --model small --output-dir ./build/faster-whisper-small
```

下载完成后，把整个目录拷到公司 Windows 机器，例如：

```text
C:/models/faster-whisper-small
```

然后公司机器的 `.env` 配成：

```env
RT_ASR_MODEL_DIR=C:/models/faster-whisper-small
RT_ASR_HF_LOCAL_ONLY=true
```

目录里至少应该有这些文件：

- `config.json`
- `model.bin`
- `tokenizer.json`

脚本会做这三个文件的完整性检查。

## 3. 运行

默认会启动本地网页字幕页，并自动打开浏览器。

```powershell
python -m realtime_lister.main
```

默认地址：

```text
http://127.0.0.1:8080
```

界面包含：

- 中央大字幕区：主显示译文，原文放在上方
- 状态条：显示当前运行状态
- Session 面板：显示当前 ASR 来源、beam size、翻译模型
- History 面板：显示最近识别记录

如果你还想保留旧的终端输出模式：

```powershell
python -m realtime_lister.main --terminal
```

可选参数：

```powershell
python -m realtime_lister.main --source-language zh --target-language en --asr-model small --translation-model gpt-5.1 --host 127.0.0.1 --port 8080
```

## 4. 精度与延迟建议

- 默认快速档：`RT_ASR_MODEL_NAME=small`
- 精度优先：`RT_ASR_MODEL_NAME=medium`
- 更快但更容易掉字：`RT_ASR_MODEL_NAME=tiny`
- CPU 机器建议：`RT_ASR_COMPUTE_TYPE=int8`
- 低延迟建议：`RT_ASR_BEAM_SIZE=1`
- 开场前准备术语表：`RT_GLOSSARY=术语A=Term A\n术语B=Term B`

## 5. 常见问题

1. 如果出现麦克风权限问题，请在 Windows 隐私设置中允许终端访问麦克风。
2. 如果翻译接口报 401/403，检查 `OPENAI_API_KEY` 与公司网关权限。
3. 如果延迟较高，先确认 `RT_ASR_MODEL_NAME=small` 且 `RT_ASR_BEAM_SIZE=1`。
4. 如果报 Hugging Face 下载失败，优先改用 `RT_ASR_MODEL_DIR` 或 `RT_ASR_HF_LOCAL_ONLY=true`。
5. 如果浏览器没有自动打开，手动访问 `http://127.0.0.1:8080`。
