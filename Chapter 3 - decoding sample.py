import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",  # Indica onde o modelo será carregado
    torch_dtype="auto",
    trust_remote_code=True,
    # attn_implementation='flash_attention_2'
    attn_implementation='eager'  # Usado em substituição ao flash-attn
)

# Confirma onde o modelo está carregado
print("\nModelo está carregado em:", model.device)
print("Cuda está disponível?", torch.cuda.is_available())
prompt = "The capital of France is" #5 Tokens
print("Prompt: ", prompt)

# Tokenize the input prompt with PyTorch(pt) - CPU
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
print("Tonenized prompt", input_ids)

# Move o tensor da memória da CPU para a GPU (cuda)
input_ids = input_ids.to("cuda")
print("Tokenized id", input_ids, "\nShape: ", input_ids[0].shape)

# Get the output of the model before lm_head
model_output = model.model(input_ids)
print("Model Output before lm_head:", model_output[0], "\nShape:", model_output[0].shape)

# Get the output of the lm_head
lm_head_output = model.lm_head(model_output[0])
print("Model output after lm_head:", lm_head_output, "\nShape:", lm_head_output.shape)

token_id = lm_head_output[0,-1].argmax(-1)

print("Token escolhido: \nToken id:", token_id, "\nToken:", tokenizer.decode(token_id))

