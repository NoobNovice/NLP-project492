import marisa_trie
import numpy as np
import os
import re
import random
from pythainlp import dict_word_tokenize
from keras.models import Sequential, model_from_yaml, Model
from keras.layers import Dense, LSTM, Bidirectional, GRU, Input, Dropout
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
    
    point = 0
    while point < len(sentence_token):
        if sentence_token[point] in stop_trie:
            del sentence_token[point]
            continue
        else:
            point += 1
    return sentence_token

if __name__ == "__main__":
    num_class = 9
    dir_path = os.getcwd()
    max_length = 35
    train_data = []
    label_data = []
    label = 0
    for intence_file in [f for f in os.listdir(dir_path) if re.match("set", f)]:
        f = open(intence_file, "r", encoding='utf-8-sig')
        print("file: {}".format(intence_file))
        print("label: {}".format(label))
        count = 1
        while(True):
            temp = f.readline()
            if not temp:
                break
            temp = temp.replace('\n', '')
            temp = temp.replace(' ', '')
            token_temp = token(temp)
            if len(token_temp) > max_length:
                raise ValueError("sentence out of range in row: {}".format(count))
            train_data.append(token_temp)
            label_data.append(label)
            count += 1
        label += 1
    if len(label_data) != len(train_data):
        raise ValueError("label and input must equal")

    for i in range(len(label_data)):
        temp_index = random.randint(0,len(label_data)-1)

        temp1 = train_data[i]
        train_data[i] = train_data[temp_index]
        train_data[temp_index] = temp1

        temp2 = label_data[i]
        label_data[i] = label_data[temp_index]
        label_data[temp_index] = temp2

    word_vectors = np.zeros((len(train_data),max_length,300))
    sample_count = 0
    for s in train_data:
        word_count = 0
        for w in s:
            try:
                word_vectors[sample_count, max_length-word_count-1,] = vocab_vec[w]
                word_count = word_count+1
            except:
                pass
        sample_count = sample_count+1
    label_data = to_categorical(label_data, num_classes=num_class)
    print(word_vectors.shape)

    inputLayer = Input(shape=(max_length,300,))
    rnn = LSTM(30, activation='sigmoid')(inputLayer)
    rnn = Dropout(0.5)(rnn)
    outputLayer = Dense(num_class, activation='softmax')(rnn)
    model = Model(inputs=inputLayer, outputs=outputLayer)

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    history = model.fit(word_vectors, label_data, epochs=50, batch_size=8, validation_split = 0.2)

    # save model weights
    model_yaml = model.to_yaml()
    with open("classify.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    model.save_weights("classify.h5")
    print("Saved model success")