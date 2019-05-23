from pythainlp import dict_word_tokenize
import marisa_trie
import os
import re
import sys
import numpy as np
from random import shuffle
from keras.models import Sequential, model_from_yaml
from keras.layers import Dense, LSTM, Bidirectional, GRU
from keras.utils import to_categorical

stop_word = [line.strip() for line in open("dict_stopword.txt", 'r', encoding='utf-8')]
stop_trie = marisa_trie.Trie(stop_word)

dict_word = [line.strip() for line in open("cc_dict.txt", 'r', encoding='utf-8')]
word_trie = marisa_trie.Trie(dict_word)

vocab_vec = {}
with open('cc_vector.txt', 'r', encoding='utf-8-sig') as target:
    while True:
        temp = target.readline()
        if not temp:
            break
        temp.replace("\n",'')
        data = temp.split()
        vocab_vec[data[0]] = [float(data[i]) for i in range(1,len(data))]

def token(sentence):
    sentence = sentence.replace('\n', ' ')
    sentence = re.sub(r'TOT|tot|ToT|WOW|wow|WoW|55+5|๕๕+๕', r'', sentence) # text emotion
    sentence = re.sub(r'[!]|[#]|[$]|[%]|[&]|[(]|[)]|[*]|[+]|[,]|[;]|[<]|[=]|[>]|[?][@]|[[]|[]]|[_]|[|]|[`]|[{]|[}]|[~]|["]', r' ', sentence)
    
    sentence_token = dict_word_tokenize(sentence, word_trie, engine='newmm')
    # sentence_token = word_tokenize(sentence, engine='newmm')

    point = 0
    while point < len(sentence_token):
        if sentence_token[point] in stop_trie:
            del sentence_token[point]
            continue
        elif sentence_token[point] == ' ':
            del sentence_token[point]
            continue
        else:
            point += 1
    return sentence_token

if __name__ == "__main__":
    word_vectorLenght = 300
    max_length = 35
    print("Read data...")
    test_data = []
    with open(sys.argv[1], "r", encoding='utf-8') as input_data:
        for sentence in input_data:
            sentence_token = token(sentence)
            if len(sentence_token) > max_length:
                raise IndexError("sentence more than model lenght 37 but {}".format(len(sentence_token)))
            test_data.append(sentence_token)
        input_data.close()
    
    print("\nWord embedding...")
    word_vectors = np.zeros((len(test_data),max_length,300))
    sample_count = 0
    for s in test_data:
        word_count = 0
        for w in s:
            try:
                word_vectors[sample_count,max_length-word_count-1,] = vocab_vec[w]
                word_count = word_count+1
            except:
                pass
        sample_count = sample_count+1
    print(word_vectors.shape)

    print("\nLoading model...")
    yaml_file = open("intence.yaml", 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)
    model.load_weights("intence.h5") # load weights into new model
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    print("\nLoaded model from disk")

    print("\nPrediction...")
    result = model.predict(word_vectors)
    result = result.tolist()
    with open("result.txt", "w", encoding='utf-8') as target:
        for i in range(len(result)):
            if result[i].index(max(result[i])) == 0:
                target.write("Q\n")
            elif result[i].index(max(result[i])) == 1:
                target.write("I\n")
            else:
                target.write("G\n")
        target.close()        