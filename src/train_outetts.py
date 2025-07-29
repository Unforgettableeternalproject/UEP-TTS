from datasets import load_from_disk
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

tokenizer = AutoTokenizer.from_pretrained("OuteAI/Llama-OuteTTS-1.0-1B")
model = ...  # load_in_4bit QLoRA version
peft_config = LoraConfig(...)
model = get_peft_model(model, peft_config)
dataset = load_from_disk("data/hf_dataset")
sft = SFTConfig(...)
trainer = SFTTrainer(model=model, train_dataset=dataset, ...)
trainer.train()
trainer.save_pretrained("outputs/checkpoints/uep_lora")
