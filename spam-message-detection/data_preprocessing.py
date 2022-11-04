# Data representation
import pandas as pd
# import numpy as np 
# #import textblob as textblob

# # NLP libraries
# from textblob import TextBlob,Word
# from nltk.corpus import stopwords
# from nltk import download as nltk_dl
# nltk_dl('stopwords')
# nltk_dl('punkt')
# nltk_dl('wordnet')

# from wordcloud import WordCloud
# from matplotlib import pyplot as plt
# import seaborn as sns

data=pd.read_csv('C:\Users\devin\OneDrive\Documents\project-ltka\dataset_sms_spam_v1.csv')
# data[0:1000]
# print('Banyak SMS yang ada: {}\n'.format(data.shape[0]))
# print('Banyak duplikat: {}\n'.format(data.duplicated().sum()))
# print('Informasi lain:')
# print(data.isnull().sum())
print(data.info())
print(data.sum())