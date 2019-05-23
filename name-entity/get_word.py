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
    for sentence_file in [f for f in os.listdir(dir_path) if re.match("sentence", f)]:
        f = open(sentence_file, "r", encoding='utf-8-sig')
        while(True):
            temp = f.readline()
            if not temp:
                break
            temp = temp.replace('\n','')
            token_temp = token(temp)
            data.append(token_temp)
    
    print("\nSave word in to cc_dict")
    vocab = set([w for s in data for w in s])
    with open('cc_dict.txt', 'a', encoding='utf-8') as f:
        for w in vocab:
            f.write(w + '\n')
        f.close()