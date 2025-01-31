import pandas as pd
from urllib import request

#Get the playlist dataset file
data = request.urlopen("https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt")
#Parse the playlist dataset file, skipping the two first lines that contain metadata
lines = data.read().decode('utf-8').split('\n')[2:]
playlists = [s.rstrip().split() for s in lines if len(s.split()) > 1] #Remove the playlist with only one song

#Load song metadata
songs_file = request.urlopen("https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt")
lines_song = songs_file.read().decode('utf-8').split('\n')
songs = [song.rstrip().split('\t') for song in lines_song]
songs_df = pd.DataFrame(data=songs, columns=['id', 'title', 'artist'])
songs_df = songs_df.set_index('id')

#print('Playlist #1: \n ', playlists[0], '\n')
#print('Playlist #2: \n ', playlists[1], '\n')


#Training the model
from gensim.models import Word2Vec

#Train our Word2Vec model
model = Word2Vec(playlists, vector_size=32, window=20, negative=50, min_count=1, workers=4)


#Ask model for similar songs
song_id = 2172
print(songs_df.iloc[song_id])

import numpy as np
def print_recommendations(song_id):
    similar_songs = np.array(model.wv.most_similar(positive=str(song_id), topn=5))[:,0]
    return songs_df.iloc[similar_songs]

#Extract recommendations
print(print_recommendations(song_id))

