from outetts import Interface

interface = Interface.from_pretrained("outputs/checkpoints/uep_lora", ...)
speaker = interface.load_speaker("speaker/uep_speaker.json")
out = interface.generate(text="´ú¸Õ»y¥y", speaker=speaker, ...)
out.save("outputs/samples/test.wav")
