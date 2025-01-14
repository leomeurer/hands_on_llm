#https://huggingface.co/docs/transformers/tokenizer_summary
from transformers import AutoTokenizer, XLNetTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
original_sentence = "I have a new GPU!"
print("Original Sentence: ", original_sentence)

tokenized_sentence = tokenizer.tokenize(original_sentence)
print("Tokenized BERT Sentence: ", tokenized_sentence)

###########################################

tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
original_sentence = "Don´t you love Transformers? We sure do."
print("Original Sentence: ", original_sentence)

tokenized_sentence = tokenizer.tokenize(original_sentence)
print("Tokenized XLNET Sentence: ", tokenized_sentence)

