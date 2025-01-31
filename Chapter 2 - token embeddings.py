from transformers import AutoModel, AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

#Load a language model
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")



# Tokenize the input prompt
tokens = tokenizer("Hello world", return_tensors="pt")

# Process the tokens
output = model(**tokens)[0]


print("Shape:")
print(output.shape)

print("")
print("Vectors:")
for token in tokens['input_ids'][0]:
 print(tokenizer.decode(token))

