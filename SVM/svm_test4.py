import linear_classifier
import math
#from gensim.models import Word2Vec
#model = Word2Vec.load("300features_40minwords_10context")
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('/home/cmanticuno/pdghcc/cnn-text-classification-tf/data/input/word_embeddings/GoogleNews-vectors-negative300.bin', binary=True)
from copy import deepcopy
from sklearn.metrics import accuracy_score
import pandas as pd
import os

max_document_length = 6298

data_test = [] 
labels_test = []

with open('/home/cmanticuno/pdghcc/cnn-text-classification-tf/data/spamham/spamham.spam.eval', encoding='utf8') as spamfile:
    for key, line in enumerate(spamfile):
        data_test = data_test + [line]
        labels_test = labels_test + [-1]
		
with open('/home/cmanticuno/pdghcc/cnn-text-classification-tf/data/spamham/spamham.ham.eval', encoding='utf8') as hamfile:
    for key, line in enumerate(hamfile):
        data_test = data_test + [line]
        labels_test = labels_test + [1]

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
		
for key, sentence in enumerate(data_test):
    data_test[key] = sentence_to_vectors(sentence, max_document_length)
        
print('Data labels: {}'.format(data_test[10]))
print('Test labels: {}'.format(labels_test[10]))

classifier = linear_classifier.load_classifier('my_svm4')

labels_classify = labels_test
labels_classify = linear_classifier.get_classification(classifier, data_test)
	
accuracy = accuracy_score(labels_test, labels_classify)
print('Accuracy; {}'.format(accuracy))
