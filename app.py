import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
import neattext.functions as nfx
import tensorflow as tf

nltk.download('wordnet')
nltk.download('stopwords')


app = Flask(__name__)


model = tf.keras.models.load_model('USE_model')


def cleantext(text):
    text = nfx.remove_special_characters(text)
    text = nfx.remove_multiple_spaces(text)
    text = nfx.remove_bad_quotes(text)
    text = nfx.remove_currency_symbols(text)
    text = nfx.remove_dates(text)
    text = nfx.remove_numbers(text)
    text = nfx.remove_currencies(text)
    text = nfx.remove_punctuations(text)
    text = text.lower()
    return text


def remove_stopwords(text):
    return ' '.join([word for word in text.split() if not word in set(stopwords.words('english'))])


def lemmatize_word(text):
    wordnet = WordNetLemmatizer()
    return ' '.join([wordnet.lemmatize(word)for word in text.split()])


def predict_genre(value):
    if value == 0:
        return "Fantasy"
    if value == 1:
        return "Science Fiction"
    if value == 2:
        return "Crime Fiction"
    if value == 3:
        return "Historical Novel"
    if value == 4:
        return "Horror"
    if value == 5:
        return "Thriller"


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        mydict = request.form
        text = mydict['summary']
        text = cleantext(text)
        text = remove_stopwords(text)
        text = lemmatize_word(text)
        probs = model.predict([text])[0]
        preds = np.argmax(probs)
        prediction = predict_genre(preds)
        return render_template('index.html', genre=prediction, text=str(text)[:100], showresult=True)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
