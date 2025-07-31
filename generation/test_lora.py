#!/usr/bin/env python3
"""
LoRA模型測試腳本
用於測試訓練好的LoRA adapter的TTS效果
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import outetts

def test_lora_model():
    """測試LoRA模型的語音生成效果"""
    
    # 設定路徑
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "models", "Llama-OuteTTS-1.0-1B")
    lora_path = os.path.join(base_dir, "outputs", "lora_model")
    
    print(f"🔍 檢查LoRA模型路徑: {lora_path}")
    
    # 檢查LoRA模型是否存在
    if not os.path.exists(lora_path):
        print(f"❌ LoRA模型不存在於: {lora_path}")
        return False
    
    # 列出LoRA模型檔案
    lora_files = [f for f in os.listdir(lora_path) if f.endswith(('.bin', '.safetensors', '.json'))]
    print(f"📁 找到LoRA檔案: {lora_files}")
    
    if not lora_files:
        print("❌ LoRA資料夾是空的，沒有找到模型檔案")
        return False
    
    try:
        print("🚀 載入基礎模型...")
        # 載入基礎模型和tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, 
            trust_remote_code=True,
            local_files_only=True
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        
        print("🔧 載入LoRA adapter...")
        # 載入LoRA adapter
        model = PeftModel.from_pretrained(base_model, lora_path)
        
        print("✅ LoRA模型載入成功！")
        print(f"📊 LoRA配置: r={model.peft_config['default'].r}, alpha={model.peft_config['default'].lora_alpha}")
        
        # 合併adapter到基礎模型 (可選)
        print("🔀 合併LoRA weights...")
        model = model.merge_and_unload()
        
        print("💾 暫時保存合併後的模型...")
        temp_model_path = os.path.join(base_dir, "outputs", "temp_merged_model")
        os.makedirs(temp_model_path, exist_ok=True)
        model.save_pretrained(temp_model_path)
        tokenizer.save_pretrained(temp_model_path)
        
        return temp_model_path
        
    except Exception as e:
        print(f"❌ 載入LoRA模型失敗: {e}")
        return False

def test_generation(model_path):
    """使用合併後的模型進行語音生成測試"""
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        print("🎤 初始化OuteTTS interface...")
        
        # 使用合併後的模型
        interface = outetts.Interface(
            config=outetts.ModelConfig(
                model_path=model_path,
                tokenizer_path=model_path,
                interface_version=outetts.InterfaceVersion.V3,
                backend=outetts.Backend.HF,
            )
        )
        
        print("🗣️ 載入說話者...")
        # 嘗試載入自定義說話者，如果沒有則使用預設
        speaker_path = os.path.join(base_dir, "speaker", "uep_speaker.json")
        if os.path.exists(speaker_path):
            speaker = interface.load_speaker(speaker_path)
            print("✅ 載入自定義說話者")
        else:
            speaker = interface.load_default_speaker("EN-FEMALE-1-NEUTRAL")
            print("✅ 使用預設說話者")
        
        # 測試用的文本
        test_texts = [
            "Hello! This is a test of our fine-tuned TTS model.",
            "The weather is beautiful today, isn't it?",
            "I hope this training has improved the voice quality.",
            "這是一個測試語句，用來檢驗中英文混合的效果。",
            "Wow! I see. Okay! Hmm. Alright!"  # 使用訓練數據中的短語
        ]
        
        print("🎵 開始生成測試語音...")
        os.makedirs(os.path.join(base_dir, "outputs", "samples"), exist_ok=True)
        
        for i, text in enumerate(test_texts):
            print(f"  🔊 生成第 {i+1}/{len(test_texts)} 個樣本: {text[:30]}...")
            
            try:
                output = interface.generate(
                    outetts.GenerationConfig(
                        text=text,
                        speaker=speaker,
                        temperature=0.4,
                        repetition_penalty=1.1,
                        top_k=40,
                        top_p=0.9
                    )
                )
                
                output_file = os.path.join(base_dir, "outputs", "samples", f"lora_test_{i+1}.wav")
                output.save(output_file)
                print(f"    ✅ 已保存: {output_file}")
                
            except Exception as e:
                print(f"    ❌ 生成失敗: {e}")
        
        print("🎉 測試完成！請檢查 outputs/samples/ 資料夾中的音頻檔案")
        return True
        
    except Exception as e:
        print(f"❌ 語音生成測試失敗: {e}")
        return False

def compare_with_base_model():
    """與基礎模型進行對比測試"""
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "models", "Llama-OuteTTS-1.0-1B")
    
    try:
        print("🔄 載入原始基礎模型進行對比...")
        
        interface = outetts.Interface(
            config=outetts.ModelConfig(
                model_path=model_dir,
                tokenizer_path=model_dir,
                interface_version=outetts.InterfaceVersion.V3,
                backend=outetts.Backend.HF,
            )
        )
        
        speaker = interface.load_default_speaker("EN-FEMALE-1-NEUTRAL")
        
        # 使用同一個測試文本
        test_text = "Hello! This is a comparison test between the base model and our fine-tuned model."
        
        output = interface.generate(
            outetts.GenerationConfig(
                text=test_text,
                speaker=speaker,
                temperature=0.4,
                repetition_penalty=1.1,
                top_k=40,
                top_p=0.9
            )
        )
        
        output_file = os.path.join(base_dir, "outputs", "samples", "base_model_comparison.wav")
        output.save(output_file)
        print(f"✅ 基礎模型對比樣本已保存: {output_file}")
        
    except Exception as e:
        print(f"❌ 基礎模型測試失敗: {e}")

def cleanup_temp_files():
    """清理臨時檔案"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_path = os.path.join(base_dir, "outputs", "temp_merged_model")
    
    if os.path.exists(temp_path):
        import shutil
        shutil.rmtree(temp_path)
        print("🧹 已清理臨時檔案")

def main():
    """主函數"""
    print("=" * 60)
    print("🧪 OuteTTS LoRA 模型測試工具")
    print("=" * 60)
    
    # 1. 測試LoRA模型載入
    merged_model_path = test_lora_model()
    if not merged_model_path:
        print("❌ 無法載入LoRA模型，請檢查模型檔案")
        return
    
    # 2. 進行語音生成測試
    print("\n" + "-" * 40)
    success = test_generation(merged_model_path)
    
    # 3. 與基礎模型對比 (可選)
    print("\n" + "-" * 40)
    compare_with_base_model()
    
    # 4. 清理臨時檔案
    cleanup_temp_files()
    
    if success:
        print("\n🎉 測試完成！")
        print("📁 請檢查 outputs/samples/ 資料夾中的生成音頻")
        print("💡 建議：")
        print("   - 比較 lora_test_*.wav 與 base_model_comparison.wav")
        print("   - 評估音質、自然度和準確性")
        print("   - 如果效果滿意，可以直接使用")
        print("   - 如果需要改進，可以考慮續接訓練")
    else:
        print("\n❌ 測試過程中出現問題，請檢查錯誤信息")

if __name__ == "__main__":
    main()
