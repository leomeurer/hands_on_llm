# Para funcionar esse o modelo local foi necessário instalar o Pytorch
# fiz a instalação do gerenciador de pacotes Anaconda
# Criei uma environment, atraves da linha de comando:
# conda create --name hands_on_llm python=3.12 -y
# Ativei a env que foi criada:
# conda activate hands_on_llm
# fiz a instalação dos pacotes do torch seguindo o exemplo da documentação https://pytorch.org/get-started/locally/#windows-verification
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Foi necessário instalar o accelerate:
# pip install accelerate>=0.26.0

# Depois disso foi só fazer a execução do código abaixo:

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    attn_implementation="eager"
)
# The tokenizer is in charge of splitting the input text into tokens before
# feeding it to the generative model.
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

from transformers import pipeline

# Create a pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False
)

# The prompt
messages = [
    {"role": "user", "content": "Conte uma piada sobre galinhas"}
]

# Generate output
output = generator(messages)
print(output[0]["generated_text"])