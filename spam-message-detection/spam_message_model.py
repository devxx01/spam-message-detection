# Data representation
import pandas as pd
import numpy as np 
# #import textblob as textblob

# # NLP libraries
from textblob import TextBlob,Word
import nltk
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk import download as nltk_dl
nltk_dl('stopwords')
nltk_dl('punkt')
nltk_dl('wordnet')

# from wordcloud import WordCloud
# from matplotlib import pyplot as plt
# import seaborn as sns

# Lemmatization Library
#from textblob import TextBlob,Word

# Model Naive Bayes Library
import os
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# READ DATASET
data=pd.read_csv('dataset_sms_spam_v1.csv')
# data[0:1000]
# print('Banyak SMS yang ada: {}\n'.format(data.shape[0]))
# print('Banyak duplikat: {}\n'.format(data.duplicated().sum()))
# print('Informasi lain:')
# print(data.isnull().sum())
print(data.info())
print(data.sum())

# DATA PREPROCESSING

# Menghilangkan duplikat
data.drop_duplicates(inplace=True)

# Mengubah labeling yang salah
data = data.replace({'label' : 2}, 1)

# Mengubah textcase menjadi all lowercase
data['Teks']=data['Teks'].str.lower()

# Menghapus stopwords
stop = stopwords.words('indonesian')
stop += ["yg", "dr", "sm", "utk", "sd", "hub", "lg", "dgn", "tp", "udh", "nah", "aja", "dg", "gak", "sy", "hub", "nama1", "di", "ada", "dari", "dan", "ini", "ke", "anda", "aku", "saya", "yang", "mau", "ya", "untuk", "dengan", "atau", "kalau", "ga", "bisa", "nya", "sdh", "uinfo", "jg", "juga", "udah"]
data["Teks"] = data['Teks'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Menghapus karakter yang bukan alfanumerik maupun whitespace
data["Teks"] = data['Teks'].str.replace('[^\w\s]','')

# Lemmatisasi setiap kata di dataset
for i in range(len(data)):
    txt = data['Teks'].values[i]
    blb = TextBlob(txt)
    wrds = blb.words
    wrd_container = []
    for wrd in wrds:
        new_wrd = Word(wrd)
        lem_word = new_wrd.lemmatize()
        wrd_container.append(lem_word)
    wrd_line = ' '.join(wrd_container)
    data.Teks[i] = wrd_line

# Konversi dataset teks menjadi angka
data.dropna(inplace=True)

X = data.Teks
y = data.label

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25)

vec = TfidfVectorizer()
training_x = vec.fit_transform(X_train)
testing_x = vec.transform(X_test)

print('Bentuk training X: {}\n'.format(training_x.shape))
print('Bentuk training y: {}\n'.format(y_train.shape))

# Pembuatan model Naive Bayes
naive_model = MultinomialNB()

# fitting model ke dataset numerik
naive_model.fit(training_x,y_train)

# membuat prediksi memakai model
predictions = naive_model.predict(testing_x)

# akurasi dari model
print('Accuracy score: {}\n'.format(accuracy_score(y_test,predictions)))

# laporan klasifikasi
print('Precision score: {}\n'.format(precision_score(y_test,predictions, average='micro')))
print('Recall score: {}\n'.format(recall_score(y_test,predictions, average='micro')))

# Dump Model
# from joblib import dump, load
# dump(naive_model, 'naive_model.joblib')

import pickle
pickle.dump(naive_model, open('model.pkl','wb'))

import joblib
joblib.dump(vec, open('vectorizer.pickle','wb'))