# RealTimeLister

会议实时翻译工具：

- 语音识别（ASR）在本地跑（`faster-whisper`）
- 翻译调用公司 LLM（例如 `gpt-5.1`）

---

## 先看这段：为什么有时下载、有时不下载

程序只有三种 ASR 运行策略，且只会命中其中一种：

1. `Fixed Local Directory`  
配置了 `RT_ASR_MODEL_DIR`，永远优先读这个目录，不自动走 Hugging Face。

2. `Offline Cache Only`  
没有 `RT_ASR_MODEL_DIR`，但配置了 `RT_ASR_HF_LOCAL_ONLY=true`。  
只允许读本地缓存，不允许下载。缓存没模型就直接报错。

3. `Model Name + Auto Download`  
没有 `RT_ASR_MODEL_DIR`，且 `RT_ASR_HF_LOCAL_ONLY=false`（默认）。  
先读本地缓存，缓存没有才尝试下载。

`RT_ASR_MODEL_DIR` 的优先级最高，这是整个项目最关键的一条规则。

---

## 1. Windows 安装

建议 Python 3.10 - 3.12。

```powershell
cd RealTimeLister
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

安装后推荐命令：

```powershell
realtime-lister
```

`python -m realtime_lister.main` 也可以，这是标准 Python 模块运行方式；只是命令更长。

---

## 2. 先选一个模板（只选一个）

```powershell
copy env.offline-model.example .env
```

三个模板：

- `env.offline-model.example`：公司机器不能访问 HF（推荐公司内网）
- `env.online.example`：公司机器可访问 HF
- `env.hf-mirror.example`：走公司 HF 镜像

不要混用多个模板内容。先选一个，再改字段。

---

## 3. `.env` 最小必填项

```env
OPENAI_API_KEY=your_key
RT_TRANSLATION_MODEL=gpt-5.1
RT_SOURCE_LANGUAGE=zh
RT_TARGET_LANGUAGE=en
RT_AUDIO_INPUT_DEVICE=auto
```

公司网关（可选）：

```env
OFFICETOOL_OPENAI_BASE_URL=https://your-company-gateway/v1
OFFICETOOL_CA_CERT_PATH=C:/path/to/company-root-ca.pem
OFFICETOOL_USE_RESPONSES_API=false
```

---

## 4. ASR 配置规则（统一口径）

### 4.1 固定目录模式（最稳）

```env
RT_ASR_MODEL_DIR=C:/models/faster-whisper-small
RT_ASR_HF_LOCAL_ONLY=true
```

行为：

- 启动只读 `RT_ASR_MODEL_DIR`
- 不会自动下载
- 目录不存在/缺文件就直接报错

### 4.2 离线缓存模式

```env
RT_ASR_MODEL_NAME=small
RT_ASR_HF_LOCAL_ONLY=true
RT_ASR_HF_CACHE_DIR=C:/hf-cache
```

行为：

- 只读缓存
- 缓存无模型就直接报错

### 4.3 联网自动模式

```env
RT_ASR_MODEL_NAME=small
RT_ASR_HF_LOCAL_ONLY=false
```

行为：

- 先读缓存
- 缓存没有时才下载

注意：

- 下载成功后不会自动回填 `RT_ASR_MODEL_DIR`
- 之后还是“模型名 + 缓存”路径

---

## 5. 公司不能访问 HF 时，怎么准备模型

在能访问外网的机器上：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_faster_whisper_model.py --model small --output-dir C:/temp/faster-whisper-small
```

把目录拷到公司机器，例如：

```text
C:/models/faster-whisper-small
```

公司机器 `.env`：

```env
RT_ASR_MODEL_DIR=C:/models/faster-whisper-small
RT_ASR_HF_LOCAL_ONLY=true
```

目录至少要有：

- `config.json`
- `model.bin`
- `tokenizer.json`

---

## 6. 运行

```powershell
realtime-lister
```

默认地址：`http://127.0.0.1:8080`

终端模式：

```powershell
realtime-lister --terminal
```

可选参数示例：

```powershell
realtime-lister --source-language zh --target-language en --asr-model small --input-device auto --translation-model gpt-5.1 --host 127.0.0.1 --port 8080
```

---

## 7. 页面上你应该看什么

启动后先看两个位置：

- `ASR Strategy`：当前是固定目录/离线缓存/联网自动哪一种
- `ASR Readiness`：当前模型是否可用；若不可用会给出具体原因

如果 `ASR Readiness` 是错误，`Start` 会被禁用或直接返回错误，不会静默卡住。

---

## 8. 常见问题

1. 没有麦克风  
页面会显示输入设备错误，`Start` 会禁用。可通过 `RT_AUDIO_INPUT_DEVICE` 或 `--input-device` 指定设备。

2. 页面没提示模型问题  
现在请看 `ASR Readiness` 卡片，它会直接显示目录缺失、离线缓存缺失等原因。

3. 延迟高  
先用：
`RT_ASR_MODEL_NAME=small`
`RT_ASR_BEAM_SIZE=1`
`RT_ASR_COMPUTE_TYPE=int8`

4. 翻译接口 401/403  
检查 `OPENAI_API_KEY`、网关地址和权限。
