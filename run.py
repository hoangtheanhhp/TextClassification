# -*- encoding: utf-8 -*-

import numpy as np
import re
from pyvi import ViTokenizer
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def stop_words(path):
    file = open(path, "r")
    stopwords = file.read().split('\n')
    return stopwords

def get_datasets_localdata(container_path=None, categories=None, load_content=True,
                           encoding='utf-16', shuffle=True, random_state=42):
    """
    Load text files with categories as subfolder names.
    Individual samples are assumed to be files stored a two levels folder structure.
    :param container_path: The path of the container
    :param categories: List of classes to choose, all classes are chosen by default (if empty or omitted)
    :param shuffle: shuffle the list or not
    :param random_state: seed integer to shuffle the dataset
    :return: data and labels of the dataset
    """
    datasets = load_files(container_path=container_path, categories=categories,
                          load_content=load_content, shuffle=shuffle, encoding=encoding,
                          random_state=random_state)
    return datasets

def get_data(path):
    doc_data = get_datasets_localdata(path)
    X, y = doc_data.data, doc_data.target
    sw = stop_words(r"stopwords.txt")
    documents = []
    for x in X:
        doc = ViTokenizer.tokenize(x)
        doc = gensim.utils.simple_preprocess(doc)
        doc = " ".join([word for word in doc if word.encode('utf-8') not in sw])
        documents.append(doc)
    return documents, y


# X_train, y_train = get_data(r"/home/anh/PycharmProjects/sub_train")
X_train, y_train = get_data(r"/home/sontc/dataset/text_classification/dataset1/train")

X_test, y_test = get_data(r"/home/sontc/dataset/text_classification/dataset1/test")
	
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X_train+y_test, y_train+y_test, test_size=0.3) 

tfidfconverter = TfidfVectorizer()
X_train = tfidfconverter.fit_transform(X_train).toarray()
X_test = tfidfconverter.fit_transform(X_test).toarray()

classifier = LogisticRegression(solver='lbfgs')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))



def count_words(docs):
    id = 0
    doc_info = []
    for doc in docs:
        id += 1
        temp = {'doc_id': id, 'doc_length': len(doc.split())}
        doc_info.append(temp)
    return doc_info


def create_freq_dict(docs):
    i = 0
    freq_dict_list = []
    for doc in docs:
        i += 1
        freq_dict = {}
        for word in doc:
            if word in freq_dict:
                freq_dict[word] += 1
            else:
                freq_dict[word] = 1
            temp = {'doc_id': i, 'freq_dict': freq_dict}
        freq_dict_list.append(temp)
    return freq_dict_list


def computeTF(doc_info, freq_dict_list):
    TF_scores = []
    for tempDict in freq_dict_list:
        id = tempDict['doc_id']
        for key in tempDict['freq_dict']:
            temp = {'doc_id': id,
                    'TF_score': tempDict['freq_dict'][key] / float(doc_info[id - 1]['doc_length']),
                    'key': key}
            TF_scores.append(temp)
    return TF_scores


def computeIDF(doc_info, freq_dict_list):
    import math
    IDF_scores = []
    for tempDict in freq_dict_list:
        for key in tempDict['freq_dict'].keys():
            count = sum([key in temp['freq_dict'] for temp in freq_dict_list])
            temp = {'doc_id': tempDict['doc_id'], 'IDF_score': math.log(1 + len(doc_info) / float(count)), 'key': key}
            IDF_scores.append(temp)
    return IDF_scores


def computeTFIDF(TF_scores, IDF_scores):
    TFIDF_scores = []
    for j in IDF_scores:
        for i in TF_scores:
            if i['key'] == j['key'] and i['doc_id'] == j['doc_id']:
                temp = {'doc_id': i['doc_id'],
                        'TFIDF_score': i['TF_score'] * j['IDF_score'],
                        'key': i['key']
                        }
            TFIDF_scores.append(temp)
    return TFIDF_scores

# doc_info = count_words(documents)
# freq_dict_list = create_freq_dict(documents)
# TF_scores = computeTF(doc_info, freq_dict_list)
# IDF_scores = computeIDF(doc_info, freq_dict_list)
# TFIDF_scores = computeTFIDF(TF_scores, IDF_scores)
# print TFIDF_scores
