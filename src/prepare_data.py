from datasets import load_dataset, Audio

ds = load_dataset("json", data_files="data/dataset.json")
ds = ds.cast_column("audio", Audio(sampling_rate=24000))
ds.save_to_disk("data/hf_dataset")
