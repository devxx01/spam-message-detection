# BUAT FRONTEND PAKE STREAMLIT DISINI
from sklearn.naive_bayes import MultinomialNB
naive_model = MultinomialNB()

from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()


sms_message = [input("masukan SMS anda:")]

test = vec.transform(sms_message)

if naive_model.predict(test) == [0]:
  print('Bukan pesan spam')
else:
  print("AWAS PESAN SPAM!!")