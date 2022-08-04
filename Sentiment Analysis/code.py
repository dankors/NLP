import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stopwords = stopwords.words("english")


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, review):
        return [self.wnl.lemmatize(t) for t in nltk.word_tokenize(review) if t not in stopwords]


pos_raw = np.loadtxt('/Users/danielkorsunsky/Desktop/McGill/Fall 2021/COMP 550/Assignments/A1/'
                     'Dataset/rt-polaritydata/rt-polarity.pos', dtype=str, delimiter='\n', encoding='latin-1')
neg_raw = np.loadtxt('/Users/danielkorsunsky/Desktop/McGill/Fall 2021/COMP 550/Assignments/A1/'
                     'Dataset/rt-polaritydata/rt-polarity.neg', dtype=str, delimiter='\n', encoding='latin-1')

reviews = np.concatenate([pos_raw, neg_raw])
target_sentiments = np.concatenate([np.ones(len(pos_raw), dtype=np.int8), np.zeros(len(neg_raw), dtype=np.int8)])
train_data, test_data, train_target, test_target = train_test_split(reviews, target_sentiments, test_size=0.20,
                                                                    random_state=550)

text_classifier = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', LogisticRegression(max_iter=200)),
])

# the first token_pattern is the default, the second includes punctuation
parameters = {
    'vect__tokenizer': [None, LemmaTokenizer()],
    'vect__token_pattern': [r"(?u)\b\w\w+\b", r"(?u)\b\w\w+\b|!|\?|\"|\'"],
    'vect__stop_words': [None, stopwords],
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
}

# GridSearchCV parameter search takes 1-2 minutes to run

gs_clf = GridSearchCV(text_classifier, param_grid=parameters, cv=5, n_jobs=-1)
gs_clf.fit(train_data, train_target)
print("Best parameter (CV score=%0.3f):" % gs_clf.best_score_)
print(gs_clf.best_params_)
predicted = gs_clf.predict(test_data)
print("Accuracy:", np.mean(predicted == test_target), "\n")
print("Classification Report:\n", classification_report(test_target, predicted, target_names=['positive', 'negative']))


# model with the best results in the gridsearch classifier

best_text_classifier = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=200)),
])

best_text_classifier.fit(train_data, train_target)
predicted = best_text_classifier.predict(test_data)
print("Accuracy:", np.mean(predicted == test_target), "\n")
print("Classification Report:\n", classification_report(test_target, predicted, target_names=['positive', 'negative']))