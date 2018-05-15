#https://pdfs.semanticscholar.org/52c5/ec55ff599ead59c8021ba2b5252e3289dee2.pdf cite this thing

import linear_classifier
import math
#from gensim.models import Word2Vec
#model = Word2Vec.load("new_model")
from copy import deepcopy
data_train = []  # Initialize an empty list of sentences
labels_train = []
import gensim
#model = gensim.models.KeyedVectors.load_word2vec_format('/work/cmanticuno/abuhman/cnn/cnn-text-classification-tf/data/input/word_embeddings/GoogleNews-vectors-negative300.bin', binary=True)
model = gensim.models.KeyedVectors.load_word2vec_format('/home/cmanticuno/pdghcc/cnn-text-classification-tf/data/input/word_embeddings/GoogleNews-vectors-negative300.bin', binary=True)

with open('/home/cmanticuno/pdghcc/cnn-text-classification-tf/data/spamham/spamham.spam', encoding='utf8') as spamfile:
    for key, line in enumerate(spamfile):
        data_train = data_train + [line]
        labels_train = labels_train + [-1]
		
with open('/home/cmanticuno/pdghcc/cnn-text-classification-tf/data/spamham/spamham.ham', encoding='utf8') as hamfile:
    for key, line in enumerate(hamfile):
        data_train = data_train + [line]
        labels_train = labels_train + [1]
		
max_document_length = max([len(x.split(" ")) for x in data_train])
print("MAX DOCUMENT LENGTH: " + str(max_document_length))
        
print(data_train[99])
print(labels_train[99])

def sentence_to_vectors(sentence, max_document_length):
    sentence_vectors = []
    for key, word in enumerate(sentence):
        if(len(sentence_vectors) < max_document_length):
            if(word.lower() in model):
                word_vector = deepcopy(model[word.lower()]).tolist()
            else:
                word_vector = [0.0]*300

            sentence_vectors = sentence_vectors + word_vector
        else:
            break
    word_vector = [0.0]*300
    while(len(sentence_vectors) < max_document_length):
        sentence_vectors = sentence_vectors + word_vector
    return sentence_vectors		

for key, sentence in enumerate(data_train):
    data_train[key] = sentence_to_vectors(sentence, max_document_length)
    
classifier = linear_classifier.train_classifier(data_train, labels_train)
linear_classifier.save_classifier(classifier, 'my_svm4')
classifier2 = linear_classifier.load_classifier('my_svm4')
classification = linear_classifier.get_classification(classifier, [data_train[99]])
print(classification)
