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
from sklearn.metrics import confusion_matrix

dir_path = os.getcwd()
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
    num_label = 9
    print("Read data...")
    test_label = []
    test_data = []
    label = 0
    for data_file in [f for f in os.listdir(dir_path) if re.match("q", f)]:
        with open(data_file, "r", encoding='utf-8-sig') as input_data:
            for sentence in input_data:
                sentence.replace("\n", " ")
                sentence_token = token(sentence)
                if len(sentence_token) > max_length:
                    sentence_token = sentence_token[0:35]
                test_data.append(sentence_token)
                test_label.append(label)
            input_data.close()
        label += 1
    test_label = to_categorical(test_label, num_classes=num_label)
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
    yaml_file = open("classify.yaml", 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)
    model.load_weights("classify.h5") # load weights into new model
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    print("\nLoaded model from disk")

    print("\nEvaluate...")
    score, acc = model.evaluate(word_vectors, test_label, batch_size=1)

    print("score: {}".format(score))
    print("accuracy: {}".format(acc))

    print("\nConfusion matrix")
    y_test = model.predict(word_vectors)
    print(confusion_matrix(y_test.argmax(axis=1), test_label.argmax(axis=1)))