# # BUAT FRONTEND PAKE STREAMLIT DISINI
# from sklearn.naive_bayes import MultinomialNB
# naive_model = MultinomialNB()

# from sklearn.feature_extraction.text import TfidfVectorizer
# vec = TfidfVectorizer()


# sms_message = [input("masukan SMS anda:")]

# test = vec.transform(sms_message)

# if naive_model.predict(test) == [0]:
#   print('Bukan pesan spam')
# else:
#   print("AWAS PESAN SPAM!!")

# BUAT FRONTEND PAKE STREAMLIT DISINI
import streamlit as st
from PIL import Image
#from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

naive_model = joblib.load(open('model.pkl','rb'))
vec = joblib.load(open('vectorizer.pickle','rb'))

def main(title = "Indonesian Spam Message Classification".upper()):
    st.markdown("<h1 style='text-align: center; font-size: 45px; color: #4682B4;'>{}</h1>".format(title), 
    unsafe_allow_html=True)
    st.image("chat.jpg")
    info = ''
    
    with st.expander("Mengklasifikasikan suatu teks apakah masuk kategori spam atau tidak"):
        text_message = st.text_input("Masukan SMS anda:")
        text_message = [text_message]
        
        if st.button("Prediksi"):
            # vec.fit(text_message)
            test = vec.transform(text_message)
            if (naive_model.predict(test) == [0]):
                info = 'Bukan pesan spam'
            else:
                info = 'AWAS PESAN SPAM !!'
            st.success('{}'.format(info))
            
if __name__ == "__main__":
    main()