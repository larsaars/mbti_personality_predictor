"""
use the trained model to predict personality
"""
import pickle as pkl
import re

import numpy as np
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from numpy import ndarray
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC


# init important variables
from xgboost import XGBClassifier

personality_types = ['IE', 'NS', 'FT', 'JP']
models = []
useless_words = stopwords.words("english")
unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
                    'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
unique_type_list = [x.lower() for x in unique_type_list]

# load the models etc.
cntizer: CountVectorizer
tfizer: TfidfTransformer
lemmatizer: WordNetLemmatizer

with open('model/cntizer.pkl', 'rb') as f:
    cntizer = pkl.load(f)
with open('model/tfizer.pkl', 'rb') as f:
    tfizer = pkl.load(f)
with open('model/lemmatizer.pkl', 'rb') as f:
    lemmatizer = pkl.load(f)

for name in personality_types:
    with open(f'model/{name}.pkl', 'rb') as f:
        model: XGBClassifier = pkl.load(f)
        models.append(model)

# pre processing methods
# Splitting the MBTI personality into 4 letters and binarizing it

b_Pers = {'I': 0, 'E': 1, 'N': 0, 'S': 1, 'F': 0, 'T': 1, 'J': 0, 'P': 1}
b_Pers_list = [{0: 'I', 1: 'E'}, {0: 'N', 1: 'S'}, {0: 'F', 1: 'T'}, {0: 'J', 1: 'P'}]


def translate_personality(personality):
    # transform mbti to binary vector
    return [b_Pers[l] for l in personality]


# To show result output for personality prediction
def translate_back(personality):
    # transform binary vector to mbti personality
    s = ''
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s


def pre_process_text(data: list, remove_stop_words=True, remove_mbti_profiles=True) -> ndarray:
    list_posts = []

    for posts in data:
        # Remove url links
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)

        # Remove Non-words - keep only words
        temp = re.sub("[^a-zA-Z]", " ", temp)

        # Remove spaces > 1
        temp = re.sub(' +', ' ', temp).lower()

        # Remove multiple letter repeating words
        temp = re.sub(r'([a-z])\1{2,}[\s|\w]*', '', temp)

        # Remove stop words
        if remove_stop_words:
            temp = " ".join([lemmatizer.lemmatize(w) for w in temp.split(' ') if w not in useless_words])
        else:
            temp = " ".join([lemmatizer.lemmatize(w) for w in temp.split(' ')])

        # Remove MBTI personality words from posts
        if remove_mbti_profiles:
            for t in unique_type_list:
                temp = temp.replace(t, '')

        # the cleaned data temp is passed here
        list_posts.append(temp)

    return np.array(list_posts)


def predict(my_posts_s: str) -> str:
    my_posts = list(my_posts_s)

    my_posts = pre_process_text(my_posts, remove_stop_words=True, remove_mbti_profiles=True)

    my_X_cnt = cntizer.transform(my_posts)
    my_X_tfidf = tfizer.transform(my_X_cnt).toarray()

    results = []
    for model in models:
        results.append(model.predict(my_X_tfidf)[0])

    return translate_back(results)


if __name__ == '__main__':
    while True:
        print('>> ' + predict(input('<< ')))
