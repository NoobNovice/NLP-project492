import marisa_trie
import numpy as np
import os
import re
from pythainlp import word_tokenize, dict_word_tokenize
import sys
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
    dir_path = os.getcwd()
    data = []
    print("Read data...")
    f = open(sys.argv[1], "r", encoding='utf-8-sig')
    count = 1
    while(True):
        print(count)
        temp = f.readline()
        if not temp:
            break
        temp = temp.replace('\n','')
        token_temp = token(temp)
        if len(token_temp) > 35:
            raise ValueError("sentence lenght must less 37 but sentence has {}".format(len(token_temp)))
        data.append(token_temp)
        count += 1
    
    print("\nWord embedding...")
    token_inModel = []
    word_vectors = np.zeros((len(data),35,300))
    sample_count = 0
    for s in data:
        word_count = 0
        temp = []
        for w in s:
            try:
                word_vectors[sample_count,35-word_count-1,:] = vocab_vec[w]
                temp.append(w)
                word_count = word_count+1
            except:
                pass
        sample_count = sample_count+1
        token_inModel.append(temp)
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

    print("\nPrediction...")
    result = model.predict(word_vectors)
    result = result.tolist()
    with open("Tdataset_" + sys.argv[1], "w", encoding='utf-8-sig') as target:
        for index in range(len(result)):
            target.write('\t'.join(token_inModel[index]) + '\n')
            bound = 35 - len(token_inModel[index])
            temp = []
            point = 34
            while point >= bound:
                max_index = result[index][point].index(max(result[index][point]))
                if max_index == 0:
                    temp.append("EN")
                elif max_index == 1:
                    temp.append("EM")
                elif max_index == 2:
                    temp.append("ET")
                elif max_index == 3:
                    temp.append("EC")
                elif max_index == 4:
                    temp.append("EP")
                elif max_index == 5:
                    temp.append("EL")
                else:
                    temp.append("OO")
                point -= 1
            target.write('  '.join(temp) + '\n\n')
        target.close()        