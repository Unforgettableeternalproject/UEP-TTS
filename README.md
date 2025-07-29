# UEP-TTS

一個專門用於訓練和測試TTS (Text-to-Speech) 模型的專案，基於OuteTTS框架開發，將來會整合到U.E.P's Core主專案中。

## 🎯 專案目標

- 訓練高品質的TTS模型
- 測試不同語音合成配置
- 為U.E.P's Core提供語音合成能力

## 📁 專案結構

```
UEP-TTS/
├── data/                    # 訓練資料集
│   ├── dataset.json
│   ├── metadata.csv
│   └── wavs/               # 音頻檔案
├── models/                 # 預訓練和自訓練模型
│   └── Llama-OuteTTS-1.0-1B/
├── generation/             # 語音生成和測試
│   ├── Example.py
│   └── Enhanced_Example.py
├── src/                    # 核心程式碼
│   ├── train_outetts.py
│   ├── prepare_data.py
│   ├── evaluate_audio.py
│   └── generate_preview.py
├── outputs/                # 訓練輸出
│   ├── checkpoints/
│   └── samples/
├── training_configs/       # 訓練配置檔案
└── speaker/               # 說話者配置
```

## 🚀 快速開始

### 環境設置

1. 啟動虛擬環境：
```bash
# Windows
env\Scripts\activate

# Linux/Mac
source env/bin/activate
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

### 測試基礎模型

運行示例腳本來測試OuteTTS基礎模型：

```bash
cd generation
python Example.py
```

### 訓練自定義模型

```bash
# 準備訓練資料
python src/prepare_data.py

# 開始訓練
python src/train_outetts.py
```

## 📋 功能特色

- ✅ 支援OuteTTS模型架構
- ✅ 本地模型載入和測試
- 🔄 自定義資料集訓練 (開發中)
- 🔄 模型微調和優化 (開發中)
- ⏳ 與U.E.P's Core整合 (規劃中)

## 🛠️ 開發狀態

- **當前階段**: 模型訓練和優化
- **下一步**: 整合到U.E.P's Core
- **測試狀態**: 基礎模型載入 ✅

## 📝 使用說明

### 生成語音

```python
import outetts
import os

# 載入模型
interface = outetts.Interface(
    config=outetts.ModelConfig(
        model_path="models/Llama-OuteTTS-1.0-1B",
        tokenizer_path="models/Llama-OuteTTS-1.0-1B",
        interface_version=outetts.InterfaceVersion.V3,
        backend=outetts.Backend.HF
    )
)

# 生成語音
speaker = interface.load_default_speaker("EN-FEMALE-1-NEUTRAL")
output = interface.generate(
    outetts.GenerationConfig(
        text="Hello, this is UEP-TTS!",
        speaker=speaker
    )
)
output.save("output.wav")
```

## 🔗 相關專案

- **U.E.P's Core**: 主要專案 (即將整合)
- **OuteTTS**: 基礎TTS框架

## 📄 授權

此專案為U.E.P's Core的子專案，遵循相同的授權協議。

## 🤝 貢獻

目前專案處於開發階段，歡迎提出建議和改進意見。

---

*此專案是U.E.P's Core生態系統的一部分*
