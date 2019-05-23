import marisa_trie
import numpy as np
import os
import re
from pythainlp import word_tokenize, dict_word_tokenize
import sys
stop_word = [line.strip() for line in open("dict_stopword.txt", 'r', encoding='utf-8')]
stop_trie = marisa_trie.Trie(stop_word)

dict_word = [line.strip() for line in open("cc_dict.txt", 'r', encoding='utf-8')]
word_trie = marisa_trie.Trie(dict_word)

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
    dir_path = os.getcwd()
    data = []
    f = open(sys.argv[1], "r", encoding='utf-8-sig')
    while(True):
        temp = f.readline()
        if not temp:
            break
        temp = temp.replace('\n','')
        token_temp = [':']
        token_temp = token_temp + token(temp)
        token_temp.append(':')
        if len(token_temp) > 37:
            raise ValueError("sentence lenght must less 37 but sentence has {}".format(token_temp))
        data.append(token_temp)
    
    test = open("test.txt", "w", encoding='utf-8')
    vocab = set([w for s in data for w in s])
    print(len(vocab))
    pretrained_word_vec_file = open('cc.th.300.vec', 'r',encoding = 'utf-8-sig')
    count = 0
    vocab_vec = {}
    for line in pretrained_word_vec_file:
        if count > 0:
            line = line.split()
            if(line[0] in vocab):
                vocab_vec[line[0]] = line[1:]
        count = count + 1
    print(len(vocab_vec))
    sample_count = 0
    for s in data:
        word_count = 0
        for w in s:
            try:
                www = vocab_vec[w]
                test.write(w + '\t')
                word_count = word_count+1
            except:
                pass
        test.write("\n")
        sample_count = sample_count+1
    test.close()