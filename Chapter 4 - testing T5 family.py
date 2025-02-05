import time
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset
from transformers import pipeline


# Loud data https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes
# Train | Validation | Test
data = load_dataset("rotten_tomatoes")

# Load model Text-To-Text Transfer Transformer (T5 model) into pipeline
start_time = time.perf_counter()
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",  #32 bits | flan-t5-small(300MB)/base(1GB)/large(3GB)/xl(12GB)/xxl(44GB)
    device="cuda:0"
    # device="cuda:0" if torch.cuda.is_available() else "cpu"
)
end_time = time.perf_counter()
print(f"Download e carregamento concluído em {end_time - start_time:.2f} segundos.")

# This models need to be instructed through prompts
prompt = "Is this following sentence positive or negative? "

# Tranformar data para "'t5': '[Is this following...? 'descrição do filme']"
data = data.map(lambda example: {"t5": prompt + example["text"]})

# Store the predicted data from T5 model - Use the TEST dataset
y_pred = []
for output in tqdm(pipe(KeyDataset(data['test'], "t5")),
                   total=len(data['test'])):
    text = output[0]['generated_text']
    y_pred.append(0 if text == 'negative' else 1)


def evaluate_performance(y_true, y_pred):
    """Create and print the classification report"""
    performance = classification_report(
        y_true, y_pred,
        target_names=['Negative Review', "Positive Review"]
    )
    print(performance)


# Valida a performance do modelo T5
evaluate_performance(data['test']['label'], y_pred)
