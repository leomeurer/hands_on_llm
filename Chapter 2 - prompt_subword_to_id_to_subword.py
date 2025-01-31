from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    attn_implementation="eager"
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

prompt = "Write an email apologizing Sarah for the late in last meeting. Explain why this happened.<|assistant|>"

# Tokenize the input prompt
input_ids = tokenizer(
    prompt,
    return_tensors="pt", #pt for PyTorch, or tf for TensorFlow
    padding=True,
    truncation=True).input_ids.to("cuda")

# Prompt tokenizado no formato de números
print(input_ids)

# Ler o significado de cada número gerado(id)
for id in input_ids[0]:
    print(tokenizer.decode(id))

generation_output = model.generate(
    input_ids=input_ids, # model não recebe o prompt, mas os input_ids
    max_new_tokens=100
)

print(generation_output)
for id in generation_output[0]:
    print(tokenizer.decode(id))


# Print the output
output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
print(output)