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

至少配置：

```env
OPENAI_API_KEY=your_key
RT_TRANSLATION_MODEL=gpt-5.1
RT_SOURCE_LANGUAGE=zh
RT_TARGET_LANGUAGE=en
```

如果走公司网关，再配置：

```env
OFFICETOOL_OPENAI_BASE_URL=https://your-company-gateway/v1
OFFICETOOL_CA_CERT_PATH=C:/path/to/company-root-ca.pem
OFFICETOOL_USE_RESPONSES_API=false
```

说明：程序会优先按 `OFFICETOOL_USE_RESPONSES_API` 调用；如果网关返回 `405`，会自动切换 `responses/chat-completions` 方式重试。

## 3. 运行

```powershell
python -m realtime_lister.main
```

可选参数：

```powershell
python -m realtime_lister.main --source-language zh --target-language en --asr-model large-v3 --translation-model gpt-5.1
```

## 4. 精度与延迟建议

- 精度优先：`RT_WHISPER_MODEL=large-v3`
- 低延迟优先：`RT_WHISPER_MODEL=medium`
- CPU 机器建议：`RT_COMPUTE_TYPE=int8`
- 开场前准备术语表：`RT_GLOSSARY=术语A=Term A\n术语B=Term B`

## 5. 常见问题

1. 如果出现麦克风权限问题，请在 Windows 隐私设置中允许终端访问麦克风。
2. 如果翻译接口报 401/403，检查 `OPENAI_API_KEY` 与公司网关权限。
3. 如果延迟较高，先把 `large-v3` 改成 `medium`。
