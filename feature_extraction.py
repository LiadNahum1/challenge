import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

USER_COUNT = 40
# WORDS_COUNT = TRAIN_SEGMENT_COUNT
WORDS_COUNT_PER_SEGMENT = 100
SEGMENT_COUNT = 150
TRAIN_SEGMENT_COUNT = 50

'''
def get_fifty_segments():
    users_lines = []
    for i in range(0, USER_COUNT):
        f = open("FraudedRawData/User" + str(i), "r")
        lines = ""
        for j in range(0, WORDS_COUNT):
            lines += f.readline()[:-1] + ' '
        users_lines.append(lines[:-1])
    return users_lines
'''


# returns an array with the number of occurrence of each word in each segment
def separate_user_to_segment(user_id):
    user_segments = []
    file_of_user = open("FraudedRawData/User" + str(user_id), "r")
    for i in range(0, SEGMENT_COUNT):
        lines = []
        for j in range(0, WORDS_COUNT_PER_SEGMENT):
            lines.append(file_of_user.readline()[:-1])
        segment_words_count = list(count_word_occurrence(lines).values())
        user_segments.append(segment_words_count)
    return user_segments


def build_word_dict():
    commends_dict = dict()
    file_of_commends = open("cmds", "r")
    for line in file_of_commends:
        commends_dict[line[:-1]] = 0
    return commends_dict


def count_word_occurrence(segment):
    d = build_word_dict()
    for word in segment:
        if word in d:
            d[word] = d[word] + 1
    return d


def build_train_set(user_id):
    train_set = separate_user_to_segment(user_id)[:TRAIN_SEGMENT_COUNT]
    train_labels = list(np.zeros(TRAIN_SEGMENT_COUNT))
    for other_user in range(0, USER_COUNT):
        if other_user != user_id:
            train_set.append(separate_user_to_segment(other_user)[0])
            train_labels.append(1)
    return train_set, train_labels


def build_test_set(user_id):
    return separate_user_to_segment(user_id)[TRAIN_SEGMENT_COUNT:SEGMENT_COUNT]


def train_model(user_id):
    train_set, train_labels = build_train_set(user_id)
    text_clf = RandomForestClassifier(n_estimators=100)
    commends = list(build_word_dict().keys())
    X_train = pd.DataFrame(data=train_set, columns=commends)
    text_clf.fit(X_train, train_labels)
    test_set = build_test_set(user_id)
    X_test = pd.DataFrame(data=test_set, columns=commends)
    predicted = text_clf.predict(X_test)
    print(predicted)
    predicted = pd.DataFrame(predicted)
    predicted = predicted.T
    predicted.to_csv("predicted.csv", mode='a')


def main():
    for i in range(0, 10):
        train_model(i)


main()
