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
from sklearn.metrics import confusion_matrix, classification_report

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

def pred2label(pred):
    out = []
    for pred_i in pred:
        for p in pred_i:
            out.append(np.argmax(p))
    return out

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
    train_data = []
    node_label = []
    print("Read data...")
    for dataset_file in [f for f in os.listdir(dir_path) if re.match("dataset", f)]:
        f = open(dataset_file, 'r', encoding='utf-8-sig')
        count = 1
        while(True):
            sentence = f.readline()
            if not sentence:
                break
            sentence = sentence.replace('\n','')
            token_temp = sentence.split('\t')
            if len(token_temp) > max_length:
                raise ValueError("sentence lenget must less than 35 but sentence lenght is {}"
                                .format(len(token_temp)))
            train_data.append(token_temp)

            temp = f.readline()
            temp = temp.replace('\n', '')
            char_label = temp.split('  ')
            
            print("line {}".format(count*3))
            print("lenght token: {}".format(len(token_temp)))
            print("lenght label: {}".format(len(char_label)))
            print(str(token_temp) + '\n')
            if len(char_label) != len(token_temp):
                raise ValueError("label not equal input")

            label = [6]*max_length
            point = max_length-1
            for i in char_label:
                if i == "EN":
                    label[point] = 0
                elif i == "EM":
                    label[point] = 1
                elif i == "ET":
                    label[point] = 2
                elif i == "EC":
                    label[point] = 3
                elif i == "EP":
                    label[point] = 4
                elif i == "EL":
                    label[point] = 5
                elif i == "OO":
                    label[point] = 6
                else:
                    raise KeyError("{} not has in case".format(i))
                point -= 1
            node_label.append(label)
            count += 1
            f.readline()
        f.close()

    if len(node_label) != len(train_data):
        raise ValueError("label and input must equal")

    word_vectors = np.zeros((len(train_data),max_length,300))
    sample_count = 0
    for s in train_data:
        word_count = 0
        for w in s:
            try:
                word_vectors[sample_count, max_length-word_count-1, :] = vocab_vec[w]
                word_count = word_count+1
            except:
                pass
        sample_count = sample_count+1
    node_label = to_categorical(node_label, num_classes=7)
    print(word_vectors.shape)


    print("\nLoading model...")
    yaml_file = open("nameEntity.yaml", 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)
    model.load_weights("nameEntity.h5") # load weights into new model
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    print("\nLoaded model from disk")

    print("\nEvaluate...")
    score, acc = model.evaluate(word_vectors, node_label, batch_size=1)

    print("\nscore: {}".format(score))
    print("accuracy: {}".format(acc))

    print("\nClassify report")
    y_test = model.predict(word_vectors)
    
    y_true = pred2label(node_label)
    y_pred = pred2label(y_test)

    x_true = [0,1,2,1]
    x_pred = [0,1,3,3]
    target_names = ["EN","EM","ET","EC","EP","EL","OO"]
    print(classification_report(y_true, y_pred, target_names=target_names))