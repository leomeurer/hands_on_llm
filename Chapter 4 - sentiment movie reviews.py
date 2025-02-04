import torch, numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset
from transformers import pipeline

# Loud data https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes
data = load_dataset("rotten_tomatoes")

# Path to our model
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Load model into pipeline
pipe = pipeline(
    model=model_path,
    tokenizer=model_path,
    return_all_scores=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Run interface
y_pred = []
for output in tqdm(pipe(KeyDataset(data['test'], 'text')), total=len(data['test'])):
    negative_score = output[0]['score']
    positive_score = output[2]['score']
    assignment = np.argmax([negative_score, positive_score])
    y_pred.append(assignment)

from sklearn.metrics import classification_report


def evaluate_performance(y_true, y_pred):
    """Create and print the classification report"""
    performance = classification_report(
        y_true, y_pred,
        target_names=['Negative Review', "Positive Review"]
    )
    print(performance)


evaluate_performance(data['test']['label'], y_pred)
