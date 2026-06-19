from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("retrain_fft/Llama-3.1-8B-SFT")
print(bool(tok.chat_template))
print(tok.chat_template[:80] if tok.chat_template else "NONE")
