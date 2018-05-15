#comes from http://scikit-learn.org/stable/modules/svm.html
#http://scikit-learn.org/stable/modules/model_persistence.html

from sklearn import svm
from sklearn.externals import joblib
import numpy

def train_classifier(input_data, labels):
    X = input_data
    y = labels
    clf = svm.SVC()
    clf.fit(X, y)  
    return clf
	
def get_classification(clf, sentence_matrix):
    prediction = clf.predict(sentence_matrix)
    return prediction
	
temp_classifier = []	
	
def set_classifier(clf):
    temp_classifier = clf
	
def get_classification_numpy(clf, sentence_matrix):
    sentence_matrix = sentence_matrix.tolist()
    prediction = temp_classifier.predict(sentence_matrix)
    prediction = numpy.asarray(prediction)
    return prediction
	
def save_classifier(clf, file_name):
    joblib.dump(clf, file_name) 
	
def load_classifier(file_name):
    clf = joblib.load(file_name)
    return clf