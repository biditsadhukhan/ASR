from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Bidwill/whisper-medium-sanskrit-out-domain")

text = "नमस्ते संसार"
tokens = tokenizer(text, return_tensors="pt")  # Get PyTorch tensors
print(tokens)