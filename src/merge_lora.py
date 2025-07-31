"""
合併LoRA adapter到base model的腳本
將訓練好的LoRA權重合併到原始模型中，創建一個新的完整模型
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import shutil

def merge_lora_to_base():
    """將LoRA adapter合併到base model"""
    
    # 路徑設定
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_model_path = os.path.join(base_dir, "models", "Llama-OuteTTS-1.0-1B")
    lora_path = os.path.join(base_dir, "outputs", "lora_model")
    merged_model_path = os.path.join(base_dir, "models", "Llama-OuteTTS-1.0-1B-Finetuned")
    
    print(f"Base model: {base_model_path}")
    print(f"LoRA adapter: {lora_path}")
    print(f"Output path: {merged_model_path}")
    
    # 檢查路徑
    if not os.path.exists(lora_path):
        print("❌ LoRA模型不存在")
        return False
    
    try:
        print("🔄 載入base model...")
        # 載入base model
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, 
            trust_remote_code=True,
            local_files_only=True
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        
        print("🔄 載入LoRA adapter...")
        # 載入LoRA模型
        model = PeftModel.from_pretrained(base_model, lora_path)
        
        print("🔄 合併LoRA權重...")
        # 合併LoRA權重到base model
        merged_model = model.merge_and_unload()
        
        print("💾 保存合併後的模型...")
        # 創建輸出目錄
        os.makedirs(merged_model_path, exist_ok=True)
        
        # 保存合併後的模型
        merged_model.save_pretrained(merged_model_path)
        tokenizer.save_pretrained(merged_model_path)
        
        # 複製配置文件
        config_files = ['config.json', 'generation_config.json', 'special_tokens_map.json']
        for config_file in config_files:
            src = os.path.join(base_model_path, config_file)
            dst = os.path.join(merged_model_path, config_file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"  ✅ 複製: {config_file}")
        
        print(f"🎉 合併完成！新模型保存在: {merged_model_path}")
        
        # 檢查保存的文件
        saved_files = os.listdir(merged_model_path)
        print(f"📁 保存的文件: {saved_files}")
        
        return True
        
    except Exception as e:
        print(f"❌ 合併過程中發生錯誤: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("LoRA到Base Model合併工具")
    print("=" * 50)
    
    success = merge_lora_to_base()
    
    if success:
        print("\n✨ 成功！現在你可以:")
        print("1. 在OuteTTS中使用合併後的模型")
        print("2. 修改generation腳本使用新的模型路徑")
        print("3. 比較原始模型和微調模型的效果")
    else:
        print("\n💡 如果合併失敗，可以考慮:")
        print("1. 檢查LoRA模型是否正確保存")
        print("2. 確認PEFT版本相容性")
        print("3. 直接使用當前的base model")
