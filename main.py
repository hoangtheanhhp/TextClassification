# -*- encoding: utf-8 -*-

import numpy as np
import re
from pyvi import ViTokenizer
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

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
        doc = re.sub(r'^https?:\/\/.*[\r\n]*', '', doc, flags=re.MULTILINE)
        doc = re.sub(" \d+", " ", doc)
        doc = gensim.utils.simple_preprocess(doc)
        doc = " ".join([word for word in doc if word.encode('utf-8') not in sw])
        documents.append(doc)
    return documents, y


X_test, y_test = get_data(r"/home/sontc/dataset/text_classification/dataset1/test")
X_train, y_train = get_data(r"/home/sontc/dataset/text_classification/dataset1/train")

tfidfconverter = TfidfVectorizer()
X_train = tfidfconverter.fit_transform(X_train)
X_test = tfidfconverter.transform(X_test)

# from sklearn.model_selection import train_test_split  
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

# classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', C=0.9)
classifier = LinearSVC(C=100)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))