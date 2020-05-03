import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

USER_COUNT = 4
WORDS_COUNT_PER_SEGMENT = 10
SEGMENT_COUNT = 150
TRAIN_SEGMENT_COUNT = 50

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
        for j in range(0, WORDS_COUNT_PER_SEGMENT * TRAIN_SEGMENT_COUNT):
            n_gram = file_of_user.readline()[:-1]
            n_gram = n_gram + file_of_user.readline()[:-1]
            n_gram = n_gram + file_of_user.readline()[:-1]
            n_gram_str = n_gram_str + n_gram + " "
        words_per_user.append(n_gram_str[:-1])
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(words_per_user)
    dense = X.todense()
    denselist = dense.tolist()
    return pd.DataFrame(denselist, columns=vectorizer.get_feature_names())


# taking top 1000 ngrams for specific user
def best_ngrams(user_id, n_grams_tidf):
    feature_number = 1000
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
    file_of_user = open("FraudedRawData/User" + str(user_id), "r")
    top_ngrams = best_ngrams(user_id, n_grams_tidf)
    for i in range(0, SEGMENT_COUNT):
        lines = []
        for j in range(0, WORDS_COUNT_PER_SEGMENT):
            n_gram = file_of_user.readline()[:-1]
            n_gram = n_gram + file_of_user.readline()[:-1]
            n_gram = n_gram + file_of_user.readline()[:-1]
            lines.append(n_gram)
        dict = build_word_dict_feature_2(top_ngrams)
        segment_words_count = list(count_word_occurrence(dict, lines).values())
        user_segments.append(segment_words_count)
    return user_segments




def get_all_features_of_all_users():
    all_features_of_all_users = []
    for user_id in range(0, USER_COUNT):
        features_1 = separate_user_to_segment_feature_1(user_id)
        features_2 = separate_user_to_segment_feature_2(user_id, tidf_grams)
        all_features = []
        for i in range(0, SEGMENT_COUNT):
            all_features.append(features_1[i].extend(features_2[i]))
        all_features_of_all_users.append(all_features)
    return all_features_of_all_users

def build_train_set(user_id, all_features_of_all_users):
    segments_of_other_users = 2
    all_features_of_id = all_features_of_all_users[user_id]
    train_set = all_features_of_id[:TRAIN_SEGMENT_COUNT]
    train_labels = list(np.zeros(TRAIN_SEGMENT_COUNT))
    for other_user in range(0, USER_COUNT):
        if other_user != user_id:
            train_set.append(all_features_of_all_users[other_user][:segments_of_other_users]) #taking two segemnts from each other user
            train_labels.append(list(np.ones(segments_of_other_users)))
    return train_set, train_labels

def train_model(user_id, all_features_of_all_users):
    all_features_of_id = all_features_of_all_users[user_id]
    train_set, train_labels = build_train_set(user_id, all_features_of_all_users)
    test_set = all_features_of_id[TRAIN_SEGMENT_COUNT:SEGMENT_COUNT]
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


if __name__ == "__main__":
    tidf_grams = tidf_n_grams()
    all_features_of_all_users = get_all_features_of_all_users(tidf_grams)
    train_model(0, tidf_grams)

