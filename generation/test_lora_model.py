import outetts
import os
from outetts import GenerationConfig, SamplerConfig, ModelConfig, Interface, Backend, GenerationType

def test_lora_model():
    """測試訓練好的LoRA模型效果"""
    
    # 使用絕對路徑
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "models", "Llama-OuteTTS-1.0-1B")
    lora_path = os.path.join(base_dir, "outputs", "lora_model")
    
    print(f"Base model: {model_dir}")
    print(f"LoRA adapter: {lora_path}")
    
    # 檢查LoRA模型是否存在
    if not os.path.exists(lora_path):
        print("❌ LoRA模型不存在，請先完成訓練")
        return
    
    lora_files = [f for f in os.listdir(lora_path) if f.endswith(('.bin', '.safetensors', '.json'))]
    if not lora_files:
        print("❌ LoRA目錄為空，請檢查模型文件")
        return
    
    print(f"✅ 找到LoRA文件: {lora_files}")
    
    try:
        # 載入帶有LoRA的模型
        print("🔄 載入LoRA微調模型...")
        
        # 注意：這裡可能需要根據OuteTTS的實際API調整
        # 如果OuteTTS不直接支持LoRA載入，我們可能需要其他方法
        interface = outetts.Interface(
            config=outetts.ModelConfig(
                model_path=model_dir,
                tokenizer_path=model_dir,
                interface_version=outetts.InterfaceVersion.V3,
                backend=outetts.Backend.HF,
                # adapter_path=lora_path,  # 如果支援的話
                n_gpu_layers=99
            )
        )
        
        print("✅ 模型載入成功")
        
        # 測試文本列表（包含你訓練數據中的類型）
        test_texts = [
            "Hello there, how are you doing?",
            "Wow! That's amazing!",
            "I see what you mean.",
            "Okay, let's try this.",
            "Hmm, that's interesting.",
            "Alright, let's continue!",
            "這是一段中文測試語句。",
            "Testing the trained voice model with a longer sentence to see how well it performs.",
            "The quick brown fox jumps over the lazy dog."
        ]
        
        # 載入或創建說話者
        speaker_path = os.path.join(base_dir, "speaker", "uep_speaker.json")
        if os.path.exists(speaker_path):
            print(f"📂 載入說話者: {speaker_path}")
            speaker = interface.load_speaker(speaker_path)
        else:
            print("📂 使用預設說話者")
            speaker = interface.load_default_speaker("EN-FEMALE-1-NEUTRAL")
        
        # 確保輸出目錄存在
        output_dir = os.path.join(base_dir, "outputs", "lora_test")
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成測試語音
        print("🎙️ 開始生成測試語音...")
        
        for i, text in enumerate(test_texts):
            try:
                print(f"  生成 {i+1}/{len(test_texts)}: {text[:50]}...")
                
                output = interface.generate(
                    config=GenerationConfig(
                        text=text,
                        generation_type=GenerationType.CHUNKED,
                        speaker=speaker,
                        sampler_config=SamplerConfig(
                            temperature=0.8,
                            repetition_penalty=1.1,
                            repetition_range=64,  # 重要：限制在64-token避免破音
                            top_k=40,
                            top_p=0.9,
                            min_p=0.05
                        ),
                    )
                )
                
                # 保存音頻
                output_file = os.path.join(output_dir, f"lora_test_{i+1:02d}.wav")
                output.save(output_file)
                print(f"    ✅ 保存: {output_file}")
                
            except Exception as e:
                print(f"    ❌ 生成失敗 {i+1}: {e}")
        
        print(f"\n🎉 測試完成！音頻文件保存在: {output_dir}")
        print("\n📋 評估建議:")
        print("  1. 聽取生成的音頻，評估音質和自然度")
        print("  2. 比較不同文本長度的效果")
        print("  3. 檢查是否有破音或不自然的地方")
        print("  4. 與原始模型對比效果差異")
        
        # 生成對比測試（原始模型）
        print("\n🔄 生成原始模型對比...")
        interface_original = outetts.Interface(
            config=outetts.ModelConfig(
                model_path=model_dir,
                tokenizer_path=model_dir,
                interface_version=outetts.InterfaceVersion.V3,
                backend=outetts.Backend.HF,
                n_gpu_layers=99
            )
        )
        
        # 生成一個對比樣本
        compare_text = "Hello, this is a comparison between the original and fine-tuned model."
        original_output = interface_original.generate(
            config=GenerationConfig(
                text=compare_text,
                generation_type=GenerationType.CHUNKED,
                speaker=speaker,
                sampler_config=SamplerConfig(
                    temperature=0.4,
                    repetition_penalty=1.1,
                    repetition_range=64,
                    top_k=40,
                    top_p=0.9,
                    min_p=0.05
                ),
            )
        )
        
        original_file = os.path.join(output_dir, "original_comparison.wav")
        original_output.save(original_file)
        print(f"✅ 原始模型對比: {original_file}")
        
    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {e}")
        print("💡 可能的原因:")
        print("  - OuteTTS不支援直接載入LoRA adapter")
        print("  - 需要先將LoRA合併到base model")
        print("  - API參數不正確")

if __name__ == "__main__":
    test_lora_model()
