from flask import Flask, render_template, redirect, url_for, request
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import joblib
import neattext.functions as nfx
import numpy as np

labels = ['Negative', 'Neutral', 'Positive']

model = load_model('./Models/twitter_sentiment_analyser.h5')
token = joblib.load('./Models/Tokenizer.joblib')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        text_without_stopwords = nfx.remove_stopwords(str(text))
        twxt_without_hashtags = nfx.remove_stopwords(str(text_without_stopwords))
        sequence = token.texts_to_sequences([text_without_stopwords])
        x_test = pad_sequences(sequence, maxlen=100)
        prediction = model.predict(x_test)
        label = labels[np.argmax(prediction)]
        return render_template('index.html', label=label)
    return render_template('index.html')