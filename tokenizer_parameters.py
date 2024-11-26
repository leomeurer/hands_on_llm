from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Lista de textos de tamanhos diferentes
texts = ["I love proggraming python", "Learn is nice"]

for text in texts:
    print(len(text))

# Tokenizando SEM padding
tokens_no_padding = tokenizer(texts)
print(tokens_no_padding)

tokens_with_padding = tokenizer(texts, padding=True)
print(tokens_with_padding)



#####################

long_text = "Era uma vez um menino que sabia contar, ele contava: " * 100 #texto longo

# Tokenizando SEM truncation
tokens_no_truncation = tokenizer(long_text)
print(tokens_no_truncation)


# Tokenizando COM truncation
tokens_with_truncation = tokenizer(long_text, truncation=True)
print(tokens_with_truncation)

###################


texts = ["Eu amo programação", "Aprender é legal"]

# Tokenizando e convertendo para tensores PyTorch
tokens_pt = tokenizer(texts, padding=True, return_tensors="pt")
print("Tensores PyTorch:", tokens_pt)

# Tokenizando e convertendo para tensores TensorFlow
tokens_tf = tokenizer(texts, padding=True, return_tensors="tf")
print("Tensores TensorFlow:", tokens_tf)

