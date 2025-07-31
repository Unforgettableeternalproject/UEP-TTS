# scripts/train_outetts.py
import os
import torch
from datasets import load_dataset, Audio
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer
from outetts import Interface, GenerationConfig, Backend, GenerationType, SamplerConfig
import outetts

# 設定本地模型路徑
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, "models", "Llama-OuteTTS-1.0-1B")

# ========== 續接訓練設定 ==========
CONTINUE_TRAINING = False  # 設為 True 來續接之前的訓練
ADAPTER_PATH = os.path.join(base_dir, "outputs", "lora_model")  # 之前訓練的adapter路徑
NEW_OUTPUT_DIR = os.path.join(base_dir, "outputs", "lora_model_v2")  # 新的輸出目錄

print(f"Loading model from local path: {model_dir}")
if CONTINUE_TRAINING and os.path.exists(ADAPTER_PATH):
    print(f"Continuing training from adapter: {ADAPTER_PATH}")
else:
    print("Starting fresh training")

# 1. 載入和處理 dataset
print("Loading and processing dataset...")

# 先創建正確格式的數據集
import json

# 讀取原始JSON數組
with open(os.path.join(base_dir, "data", "dataset.json"), "r", encoding="utf-8-sig") as f:
    original_data = json.load(f)

# 轉換為正確格式並修正路徑
processed_data = []
for item in original_data:
    # 修正音頻路徑
    audio_path = item["audio"].replace("...", base_dir.replace("\\", "/"))
    audio_path = os.path.normpath(audio_path)
    
    processed_item = {
        "audio": audio_path,
        "text": item["transcript"]  # 改為 'text' 欄位，更符合標準
    }
    processed_data.append(processed_item)

# 創建臨時的JSONL格式文件
temp_dataset_path = os.path.join(base_dir, "data", "temp_dataset.jsonl")
with open(temp_dataset_path, "w", encoding="utf-8") as f:
    for item in processed_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Processed {len(processed_data)} samples")

# 載入處理後的數據集
ds = load_dataset("json", data_files=temp_dataset_path, split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=24000))

# 2. 從本地載入 Tokenizer & Base Model 與量化設置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16  # 使用 torch.float16 而不是字串
)

# 從本地載入
tokenizer = AutoTokenizer.from_pretrained(
    model_dir, 
    trust_remote_code=True,
    local_files_only=True  # 確保只使用本地文件
)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,  # 確保只使用本地文件
    torch_dtype=torch.float16
)

# 確保模型有 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. LoRA Adapter 設定和載入
if CONTINUE_TRAINING and os.path.exists(ADAPTER_PATH):
    print("Loading existing adapter for continued training...")
    # 載入已訓練的adapter
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    # 獲取當前的LoRA配置
    lora_config = model.peft_config['default']
    print(f"Loaded adapter with r={lora_config.r}, alpha={lora_config.lora_alpha}")
else:
    print("Creating new LoRA adapter...")
    # 創建新的LoRA配置
    lora_config = LoraConfig(
        r=32, # 增加 r 值提升效果
        lora_alpha=32,  # 增加 alpha 值提升效果
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 增加更多目標模組
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

# 4. 訓練參數設定 - 使用 TrainingArguments
training_args = TrainingArguments(
    output_dir=os.path.join(base_dir, "outputs", "lora_model"),  # 使用絕對路徑
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # 增加有效 batch size
    num_train_epochs=10,
    save_steps=500,
    logging_steps=100,
    warmup_steps=100,
    dataloader_num_workers=0,  # Windows 建議設為 0
    fp16=True,  # 啟用混合精度
    report_to=None,  # 關閉 wandb 等報告
    save_total_limit=2,  # 只保留最新的 2 個 checkpoint
    remove_unused_columns=False,  # 保留音頻欄位
    max_steps=-1,  # 使用 epochs 而不是 steps
)

# 數據預處理 - 為SFTTrainer準備文本格式
def format_dataset(examples):
    """將數據格式化為SFTTrainer期望的格式"""
    texts = examples['text']
    # 為OuteTTS格式化文本
    formatted_texts = []
    for text in texts:
        # 簡單的格式化，實際可能需要更複雜的處理
        formatted_text = f"<|text|>{text}<|audio|>"
        formatted_texts.append(formatted_text)
    
    return {"text": formatted_texts}

# 格式化數據集
print("Formatting dataset for training...")
formatted_ds = ds.map(
    format_dataset, 
    batched=True, 
    remove_columns=[col for col in ds.column_names if col != 'text'],  # 只保留audio和處理後的text
    desc="Formatting"
)

# 5. 建立 Trainer 並訓練
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_ds,  # 使用格式化的數據集
    args=training_args,
    processing_class=tokenizer,  # 使用 processing_class 而不是 tokenizer
    peft_config=lora_config,  # 添加 peft_config
)

print("Starting training...")
trainer.train()

# 儲存模型
print("Saving model...")
output_path = os.path.join(base_dir, "outputs", "lora_model")
os.makedirs(output_path, exist_ok=True)

try:
    # 使用trainer的save_model方法
    trainer.save_model(output_path)
    print(f"✅ Trainer model saved to: {output_path}")
except Exception as e:
    print(f"❌ Trainer save failed: {e}")

try:
    # 單獨保存adapter權重
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(output_path)
        print(f"✅ PEFT adapter saved to: {output_path}")
except Exception as e:
    print(f"❌ PEFT save failed: {e}")

try:
    # 保存tokenizer
    tokenizer.save_pretrained(output_path)
    print(f"✅ Tokenizer saved to: {output_path}")
except Exception as e:
    print(f"❌ Tokenizer save failed: {e}")

# 檢查保存的文件
saved_files = [f for f in os.listdir(output_path) if f.endswith(('.bin', '.safetensors', '.json'))]
print(f"📁 Saved files: {saved_files}")

print(f"Training completed! Check: {output_path}")

# 6. 測試生成（可選）
try:
    print("Testing generation with trained adapter...")
    
    # 確保輸出目錄存在
    os.makedirs("../outputs/samples", exist_ok=True)
    
    # 使用本地模型和訓練好的 adapter
    interface = Interface(
        config=outetts.ModelConfig(
            model_path=model_dir,
            tokenizer_path=model_dir,
            interface_version=outetts.InterfaceVersion.V3,
            backend=outetts.Backend.HF,
            # adapter_path="outputs/lora_model"  # 如果 OuteTTS 支援 adapter 載入
        )
    )
    
    # 載入說話者
    if os.path.exists("../speaker/uep_speaker.json"):
        speaker = interface.load_speaker("../speaker/uep_speaker.json")
    else:
        # 使用預設說話者
        speaker = interface.load_default_speaker("EN-FEMALE-1-NEUTRAL")
    
    # 生成測試語音
    output = interface.generate(
        config=GenerationConfig(
        text="Oh my gosh, stories are just the best! They whisk you away to different worlds and introduce you to amazing characters. The emotions, the adventures... I just love getting lost in them!",
        generation_type=GenerationType.CHUNKED,
        speaker=speaker,
        sampler_config=SamplerConfig(
            temperature=0.4,
            repetition_penalty=1.1,
            # 重要 Sampling 設定，OuteTTS‑1.0 要限制在 64-token recent window 才能避免破音 :contentReference[oaicite:1]{index=1}
            repetition_range=64,
            top_k=40,
            top_p=0.9,
            min_p=0.05
        ),
    )
    )
    
    output.save("../outputs/samples/test.wav")
    print("Sample output saved to outputs/samples/test.wav")
    
except Exception as e:
    print(f"Generation test failed: {e}")
    print("Training completed successfully, but generation test skipped.")
