from gensim.models import Word2Vec
import sys
import os
num_epochs = 50
vector_lenght = 0
window_size = 2
dir_path = os.getcwd()
model_name = "word_vector.model"
# python word2vec.py -v 50 
# check parametor
if len(sys.argv) >= 3 and len(sys.argv) < 9:
    for index, arg in enumerate(sys.argv):
        if arg in ['--vector', '-v'] and len(sys.argv) > index + 1:
            vector_lenght = int(sys.argv[index + 1])
            del sys.argv[index]
            del sys.argv[index]
            break
    for index, arg in enumerate(sys.argv):
        if arg in ['--epochs', '-e'] and len(sys.argv) > index + 1:
            num_epochs = int(sys.argv[index + 1])
            del sys.argv[index]
            del sys.argv[index]
            break
    for index, arg in enumerate(sys.argv):
        if arg in ['--window', '-w'] and len(sys.argv) > index + 1:
            window_size = int(sys.argv[index + 1])
            del sys.argv[index]
            del sys.argv[index]
            break

    for index, arg in enumerate(sys.argv):
        if arg in ['--output', '-o'] and len(sys.argv) > index + 1:
            model_name = int(sys.argv[index + 1])
            del sys.argv[index]
            del sys.argv[index]
            break
else:
    raise ValueError('Pease check number of arguments')

# read word sentence
sentence = []
for token_file in [f for f in os.listdir(dir_path+"/resources/dataset/token/") if f.endswith('.txt')]:
    try:
        text_file = open(dir_path + "/resources/dataset/token/" + token_file, "r", encoding='utf-8')
        while True:
            temp = text_file.readline()
            if not temp:
                break
            temp = temp.replace('\n','')
            temp = temp.split('\t')
            sentence.append(temp)
        text_file.close()
    except:
        raise IOError("Could not read file in {}/resources/dataset/token/{}".format(dir_path, token_file))

model = Word2Vec(sentence,
                 size=vector_lenght, #dimension of word vector
                 window=window_size, #context window size
                 min_count=3, #words that occur less than min_count will be ignored
                 sg=0) #use skip-gram model (cbow is used when sg = 0)
model.save(dir_path + "/model/" + model_name)