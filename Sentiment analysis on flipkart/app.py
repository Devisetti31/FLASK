import pickle
from flask import Flask,render_template,request

from sklearn.pipeline import Pipeline
import re

import nltk
from nltk.tokenize import word_tokenize as wt,sent_tokenize as st
from nltk.corpus import stopwords
from nltk import PorterStemmer,LancasterStemmer,SnowballStemmer
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.pipeline import Pipeline ,make_pipeline
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import FunctionTransformer,PowerTransformer,StandardScaler
import pandas as pd

import emoji

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

app = Flask(__name__)

#data = pd.read_csv(r'C:\Users\LAKSHMI NARASIMHARAO\innomatics\m_internship\task flipkart-reviews\reviews_badminton\data.csv')




def lowers(x):
    return x.str.lower()

def html(x):
    return x.apply(lambda x:re.sub("<.+?>"," ",x))

def url(x):
    return x.apply(lambda x:re.sub("https[s]?://.+? +"," ",x))

def unw(x):
    return x.apply(lambda x:re.sub("[]()*\-.:,@#$%&^!?/0-9']"," ",x))

def unw_1(x):
    return x.apply(lambda x:re.sub(r'[^\w\s]','',x))

def emoji_remove(x):
    x = x.apply(lambda x : emoji.demojize(x))
    return x


def lemma(x):
  list_stp = stopwords.words("english")  # list_stp contains group of stop words.
  wl=WordNetLemmatizer()

  def lemmatize_text(text):
        words = wt(text)
        lemmatized_words = [wl.lemmatize(word, pos="v") for word in words if word not in list_stp]
        return " ".join(lemmatized_words)

  return x.apply(lemmatize_text)

# create a pipeline for pre-processing the data
pre_pro_pi = Pipeline([("emojii remover",FunctionTransformer(emoji_remove)),
                       ("lower",FunctionTransformer(lowers)),
                       ("html",FunctionTransformer(html)),
                       ("url",FunctionTransformer(url)),
                       ("unw",FunctionTransformer(unw)),
                       ("unw2",FunctionTransformer(unw_1)),
                       ("advance",FunctionTransformer(lemma))])

text_to_bow = Pipeline([("pre-processing",pre_pro_pi),("countvectorizer",CountVectorizer()),('normalization',StandardScaler(with_mean=False))])

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods=['POST','GET'])
def result():
    review = request.form.get('review')

    pre = pickle.load(open(r'final_pre.pkl','rb'))

    model = pickle.load(open(r'model\model.pkl','rb') )

    if review is not None:
    
        query = pd.DataFrame([review],columns=["Reviews"])

        query = pre.transform(query)

        y_pred = model.predict(query)

        if y_pred == 0:
            y_pred = 'Negative'
        else:
            y_pred = 'Positive'
    else:
        y_pred = "There is an error occured from user"
    

    return render_template('output.html', y_pred = y_pred)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')