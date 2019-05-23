import numpy as np
import os
import re
import random
from keras.models import Sequential, model_from_yaml, Model
from keras.layers import Dense, LSTM, Bidirectional, GRU, Input, Dropout
from keras.utils import to_categorical

vocab_vec = {}
with open('cc_vector.txt', 'r', encoding='utf-8-sig') as target:
    while True:
        temp = target.readline()
        if not temp:
            break
        temp.replace("\n",'')
        data = temp.split()
        vocab_vec[data[0]] = [float(data[i]) for i in range(1,len(data))]

if __name__ == "__main__":
    dir_path = os.getcwd()
    max_length = 35
    train_data = []
    node_label = []
    # input กับ label อยู่ใน file เดียวกัน
    for dataset_file in [f for f in os.listdir(dir_path) if re.match("dataset", f)]:
        f = open(dataset_file, "r", encoding='utf-8-sig')
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
            
            print(dataset_file)
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

            f.readline()
            count += 1

    if len(node_label) != len(train_data):
        raise ValueError("label and input must equal")

    for i in range(len(node_label)):
        temp_index = random.randint(0,len(node_label)-1)

        temp1 = train_data[i]
        train_data[i] = train_data[temp_index]
        train_data[temp_index] = temp1

        temp2 = node_label[i]
        node_label[i] = node_label[temp_index]
        node_label[temp_index] = temp2
    
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

    inputLayer = Input(shape=(max_length,300,))
    rnn = Bidirectional(LSTM(30, return_sequences=True, activation='sigmoid'))(inputLayer)
    rnn = Dropout(0.5)(rnn)
    outputLayer = Dense(7, activation='softmax')(rnn)
    model = Model(inputs=inputLayer, outputs=outputLayer)

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    history = model.fit(word_vectors, node_label, epochs=300, batch_size=8, validation_split = 0.2)

    model_yaml = model.to_yaml()
    with open("nameEntity.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    model.save_weights("nameEntity.h5")
    print("Saved model success")