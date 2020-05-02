import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

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
        else:
            d[word] = 1
    return d


def build_train_set():
    train_set = separate_user_to_segment(0)[:TRAIN_SEGMENT_COUNT]
    train_labels = list(np.zeros(TRAIN_SEGMENT_COUNT))
    for user_id in range(1, USER_COUNT):
        train_set.extend(separate_user_to_segment(user_id)[:TRAIN_SEGMENT_COUNT])
        train_labels.extend(list(np.ones(TRAIN_SEGMENT_COUNT)))
    return train_set, train_labels


def build_test_set():
    return separate_user_to_segment(0)[TRAIN_SEGMENT_COUNT:SEGMENT_COUNT]


def main():

    train_set , train_labels = build_train_set()
    text_clf = RandomForestClassifier(n_estimators=100)
    text_clf.fit(np.array(train_set), np.array(train_labels))
    test_set = build_test_set()
    predicted = text_clf.predict(np.array(train_set))
    print(predicted)


main()
