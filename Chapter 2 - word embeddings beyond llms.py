import gensim.downloader as api

#info = api.info() #show info about avaliable models/datasets
#print(info)

# Download embeddings (66MB, glove, trained on wikipedia, vector size: 50)
 # Other options include "word2vec-google-news-300"
 # More options at https://github.com/RaRe-Technologies/gensim-data
model = api.load("glove-wiki-gigaword-50")

print(model.most_similar([model["king"]], topn=11))
