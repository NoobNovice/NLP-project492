import sys

if __name__ == "__main__":    
    data = []
    f = open(sys.argv[1], "r", encoding='utf-8')
    while(True):
        temp = f.readline()
        if not temp:
            break
        temp = temp.replace('\n','')
        data.append(temp)
    
    test = open("cc_dict.txt", "w", encoding='utf-8')
    vocab = set([w for w in data])
    pretrained_word_vec_file = open('cc.th.300.vec', 'r',encoding = 'utf-8-sig')
    for line in pretrained_word_vec_file:
        line = line.split()
        if line[0] in vocab:
            test.write(line[0] + '\n')
    test.close()