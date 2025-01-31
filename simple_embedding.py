from transformers import AutoModel, AutoTokenizer

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')

# Load a language model
model = AutoModel.from_pretrained('microsoft/deberta-v3-xsmall')

# Tokenizando uma frase
tokens = tokenizer("Hello world", return_tensors="pt")
print(tokens)

# Processes the tokens
output = model(**tokens)[0]
print(output)
print(output.shape)
