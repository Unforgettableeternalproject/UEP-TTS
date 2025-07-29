import outetts
import os

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

speaker = interface.load_default_speaker("EN-FEMALE-1-NEUTRAL")
output = interface.generate(
    outetts.GenerationConfig(
        text="Hello there, how are you doing?",
        speaker=speaker,
    )
)
output.save("../outputs/samples/output.wav")
