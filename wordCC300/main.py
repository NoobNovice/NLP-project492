import sys
if __name__ == "__main__":
    word = []
    with open(sys.argv[1], 'r', encoding='utf-8-sig') as f:
        while True:
            temp = f.readline()
            if not temp:
                break
            word.append(temp.replace('\n',''))
        f.close()
    
    with open('cc_vector.txt', 'w', encoding='utf-8-sig') as output:
        vocab = set([w for w in word])
        pretrained_word_vec_file = open('cc.th.300.vec', 'r',encoding = 'utf-8-sig')
        for line in pretrained_word_vec_file:
            line = line.split()
            if(line[0] in vocab):
                output.write(" ".join([str(d) for d in line]) + "\n")
        output.close()