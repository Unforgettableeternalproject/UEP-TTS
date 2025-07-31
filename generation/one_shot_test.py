import outetts
import os
from outetts import GenerationConfig, SamplerConfig, ModelConfig, Interface, Backend, GenerationType

# 使用絕對路徑
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, "models", "Llama-OuteTTS-1.0-1B")

# 嘗試使用本地模式
interface = outetts.Interface(
    config=outetts.ModelConfig(
    model_path=model_dir,  # 只指定模型目錄，不包括具體的.safetensors文件
    tokenizer_path=model_dir,
    interface_version=outetts.InterfaceVersion.V3,
    backend=outetts.Backend.HF,
    n_gpu_layers=99
    )
)

# 建立 speaker profile（只需大約 10 秒左右的參考音檔）
speaker = interface.create_speaker("../data/my_reference.mp3")
interface.save_speaker(speaker, "../speaker/bernie_speaker.json")

# 或者之後重用：
# speaker = interface.load_speaker("../speaker/uep_speaker.json")

# 預覽 speaker 編碼：
# interface.decode_and_save_speaker(speaker, "../outputs/debug/debug_speaker.wav")

# 生成語音
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

# 輸出結果儲存
output.save("../outputs/samples/uep_test.wav")
print("Saved generated voice: uep_test.wav")
