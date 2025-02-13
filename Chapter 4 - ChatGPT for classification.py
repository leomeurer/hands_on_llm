import os
from dotenv import load_dotenv

import openai
from datasets import load_dataset
from tqdm import tqdm


#pip install python-dotenv
print("DotEnv Loaded... ", load_dotenv())

## Para fins didáticos estou usando a API da GROQ com compatibilidade da OpenAI
# https://console.groq.com/docs/openai
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ['GROQ_API_KEY']
    )


def chatgpt_generation(prompt, document, model='qwen-2.5-32b'):
    """Generate an output based on a prompt and an input document"""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": prompt.replace("[DOCUMENT]", document)
        }
    ]
    print(messages)
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0
    )
    return chat_completion.choices[0].message.content


prompt = """Predict whether the folowing document is a positive or negative movie review:

[DOCUMENT]

If it is positive return 1 and if it is negative return 0. Do not give any other answer to the question.
"""

# Loud data https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes
data = load_dataset("rotten_tomatoes")

#tqdm mostra a barra de progresso
#para fins de estudo, pego somente as 10 primeiras ocorrências
predictions = [
    chatgpt_generation(prompt, doc) for doc in tqdm(data["test"]["text"][:10])
]
print(predictions)
# Extract predictions


#document = "unpretentious, charming, quirky, original"
#prompt = chatgpt_generation(prompt, document)
#print(prompt)