import collections
import os

import numpy as np
from sklearn import metrics, svm, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

USER_COUNT = 40
WORDS_COUNT_PER_SEGMENT = 100
SEGMENT_COUNT = 150
TRAIN_SEGMENT_COUNT = 50
feature_number = 1000
segments_of_other_users = 2
num_of_estimatiors = 80
# returns a dictinary of the words in dict with the occurance of them in specific segment
def count_word_occurrence(dict, segment):
    for word in segment:
        if word in dict:
            dict[word] = dict[word] + 1
    return dict

# feature 1 - commands in cmds
# returns a dictionary of all commands with zero
def build_word_dict_feature_1():
    commends_dict = dict()
    file_of_commends = open("cmds", "r")
    for line in file_of_commends:
        commends_dict[line[:-1]] = 0
    return commends_dict

# returns an array with the number of occurrence of each word in each segment of specific user
def separate_user_to_segment_feature_1(user_id):
    user_segments = []
    file_of_user = open("FraudedRawData/User" + str(user_id), "r")
    for i in range(0, SEGMENT_COUNT):
        lines = []
        for j in range(0, WORDS_COUNT_PER_SEGMENT):
            lines.append(file_of_user.readline()[:-1])
        dict = build_word_dict_feature_1()
        segment_words_count = list(count_word_occurrence(dict, lines).values())
        user_segments.append(segment_words_count)
    return user_segments


# feature 2 - 3grams of commands taking the best 1000 according to tidf
# returns 3grams commands tf-idf for each user
def tidf_n_grams():
    words_per_user = []
    for i in range(0, USER_COUNT):
        file_of_user = open("FraudedRawData/User" + str(i), "r")
        n_gram_str = ""
        n_gram_1 = file_of_user.readline()[:-1]
        n_gram_2 = file_of_user.readline()[:-1]
        for j in range(2, WORDS_COUNT_PER_SEGMENT * TRAIN_SEGMENT_COUNT):
            n_gram_3 = file_of_user.readline()[:-1]
            n_gram_str = n_gram_str + n_gram_1 + n_gram_2 + n_gram_3 + " "
            n_gram_1 = n_gram_2
            n_gram_2 = n_gram_3
        words_per_user.append(n_gram_str[:-1])
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(words_per_user)
    dense = X.todense()
    denselist = dense.tolist()
    return pd.DataFrame(denselist, columns=vectorizer.get_feature_names())


# taking top 1000 ngrams for specific user
def best_ngrams(user_id, n_grams_tidf):
    columns = []
    for i in range(0, USER_COUNT):
        columns.append(f'user_{i}')
    n_grams_tidf_T = pd.DataFrame(data=n_grams_tidf.T)
    n_grams_tidf_T.columns = columns
    best_ngrams = n_grams_tidf_T.nlargest(feature_number, f'user_{user_id}').index.to_list()
    return best_ngrams

# returns a dictionary of best ngrams with zero
def build_word_dict_feature_2(best_ngrams):
    ngrams_dict = dict()
    for ngram in best_ngrams:
        ngrams_dict[ngram] = 0
    return ngrams_dict

# returns an array with the number of occurrence of each word in each segment of specific user
def separate_user_to_segment_feature_2(user_id, n_grams_tidf):
    user_segments = []
    top_ngrams = best_ngrams(user_id, n_grams_tidf)
    file_of_user = open("FraudedRawData/User" + str(user_id), "r")
    for i in range(0, SEGMENT_COUNT):
        lines = []
        n_gram_1 = file_of_user.readline()[:-1]
        n_gram_2 = file_of_user.readline()[:-1]
        for j in range(2, WORDS_COUNT_PER_SEGMENT):
            n_gram_3 = file_of_user.readline()[:-1]
            lines.append(n_gram_1 + n_gram_2 + n_gram_3)
            n_gram_1 = n_gram_2
            n_gram_2 = n_gram_3
        dict = build_word_dict_feature_2(top_ngrams)
        segment_words_count = list(count_word_occurrence(dict, lines).values())
        user_segments.append(segment_words_count)
    return user_segments




def get_all_features_of_all_users(tidf_grams):
    all_features_of_all_users = []
    for user_id in range(0, USER_COUNT):
        features_1 = separate_user_to_segment_feature_1(user_id)
        features_2 = separate_user_to_segment_feature_2(user_id, tidf_grams)
        all_features = []
        for i in range(0, SEGMENT_COUNT):
            features_1[i].extend(features_2[i])
            all_features.append(features_1[i])
        all_features_of_all_users.append(all_features)
    return all_features_of_all_users

def build_train_set(user_id, all_features_of_all_users):
    all_features_of_id = all_features_of_all_users[user_id]
    train_set = all_features_of_id[:TRAIN_SEGMENT_COUNT]
    train_labels = list(np.zeros(TRAIN_SEGMENT_COUNT))
    for other_user in range(0, USER_COUNT):
        if other_user != user_id:
            train_set.extend(all_features_of_all_users[other_user][:segments_of_other_users]) #taking two segemnts from each other user
            train_labels.extend(list(np.ones(segments_of_other_users)))
    return train_set, train_labels


def find_top_20(probs):
    ones = []
    for prob in probs:
        if prob >= 0.5:
            ones = ones + [prob]
    
    while len(ones) > 15:
        ones.remove(min(ones))

    return min(ones)

def train_model(user_id, all_features_of_all_users):
    all_features_of_id = all_features_of_all_users[user_id]
    train_set, train_labels = build_train_set(user_id, all_features_of_all_users)
    test_set = all_features_of_id[TRAIN_SEGMENT_COUNT:SEGMENT_COUNT]
    text_clf = RandomForestClassifier(n_estimators=num_of_estimatiors)

    X_train = pd.DataFrame(data=train_set)
    text_clf.fit(X_train, train_labels)
    X_test = pd.DataFrame(data=test_set)
    predicted = text_clf.predict(X_test)

    predicted_prob = text_clf.predict_proba(X_test)[:,1]
    max20 = find_top_20(predicted_prob.copy())

    count = 0
    for pred in predicted:
        if (pred == 1) & (predicted_prob[count] >= max20) :
            predicted[count] = 1
        elif pred == 1:
            predicted[count] = 0
        count = count + 1

    #print(user_id)
    #print(predicted)
    #print(collections.Counter(predicted))
    predicted = pd.DataFrame(predicted)
    predicted = predicted.T
    predicted.to_csv("predicted.csv", mode='a', header= False, index=False)

def calculate_grade():
    real_train_user_test = pd.read_csv('train_users.csv', header=None)
    predicted_train_user_test = pd.read_csv('predicted.csv', header=None)
    predicted_train_user_test = predicted_train_user_test.iloc[0:10, :]
    true_positive = 0
    true_negative = 0
    for i in range(0, 10):
        check_true_positive = list((real_train_user_test.iloc[i] == predicted_train_user_test.iloc[i]) & (real_train_user_test.iloc[i] == 1))
        true_positive = true_positive + check_true_positive.count(True)
        check_predicted_ones = list(predicted_train_user_test.iloc[i] == 1)
        false_positive = check_predicted_ones.count(True) - check_true_positive.count(True)
        check_true_negative = list((real_train_user_test.iloc[i] == predicted_train_user_test.iloc[i]) & (real_train_user_test.iloc[i] == 0))
        true_negative = true_negative + check_true_negative.count(True)
        print(f'true_positive: {check_true_positive.count(True)} true_negative: {check_true_negative.count(True)} false positive: {false_positive}')

    classification_score = true_negative + true_positive * 9
    print(f'ones: {true_positive} zeroes:{true_negative}  classification_score: {classification_score}')

if __name__ == "__main__":
    os.remove("predicted.csv")
    tidf_grams = tidf_n_grams()
    all_features_of_all_users = get_all_features_of_all_users(tidf_grams)
    for user_id in range(0, USER_COUNT):
        train_model(user_id, all_features_of_all_users)
    calculate_grade()
