if __name__ == "__main__":
    vocab_vec = {}
    with open('test.txt', 'r', encoding='utf-8-sig') as target:
        while True:
            temp = target.readline()
            if not temp:
                break
            temp.replace("\n",'')
            data = temp.split("\t")
            vocab_vec[data[0]] = [float(data[i]) for i in range(1,len(data))]
    print(len(vocab_vec))