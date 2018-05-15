import re
import numpy as np

def get_datasets_dbpedia(subset, limit):
    """
    Loads dbpedia data from files, split the data into words and generates labels.
    Returns split sentences and labels.
    """
    datasets = dict()
    data = []
    target = []
    target_int =[]
    target_names = []
    
    filename = 'data/dbpedia/'+subset+'.csv'
    last_label = ''
    i = 0
    # Load data from files
    with open(filename, 'r') as f:
        for line in f:
            label, header, text = line.split(',')
            #print('.',end='')

            if (i >= limit):
                if (label == last_label):
                    continue
                else:
                    i = 0    # reset i

            print('Entry : {}, label:{}, header: {}'.format(i,label, header))
            # remove blank spaces from text and insert into list 'data'
            data.append(text.strip())
            target.append(int(label)-1)
            if label not in target_names:
                target_names.append(label)
            last_label = label
            i += 1

    datasets['data'] = data
    datasets['target'] = target
    datasets['target_names'] = target_names
    return datasets

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_labels(datasets):
    """
    Load data and labels
    :param datasets:
    :return:
    """
    # Split by words
    x_text = datasets['data']
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    labels = []
    for i in range(len(x_text)):
        label = [0 for j in datasets['target_names']]
        #print('target={}, i={}'.format(datasets['target'], i))
        label[datasets['target'][i]] = 1
        labels.append(label)
    y = np.array(labels)
    return [x_text, y]

datasets = get_datasets_dbpedia('train', 100)
print('targets: {}'.format(datasets['target']))
print('target_namess: {}'.format(datasets['target_names']))
x_train, y = load_data_labels(datasets)
print('target_names(y) size: {}'.format(len(y)))

