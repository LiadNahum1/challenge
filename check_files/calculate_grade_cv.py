import pandas as pd
USER_TRAIN =10
USER_COUNT = 40
SEGMENT_COUNT = 150
TRAIN_SEGMENT_COUNT = 50
ONES = 300
ZEROES = 2700

if __name__ == "__main__":
    real_train_user_test = pd.read_csv('check/train_users.csv', header=None)
    predicted_train_user_test = pd.read_csv('predicted.csv', header=None)

    true_positive = 0
    true_negative = 0
    for i in range(0, 10):
        check_true_positive = list((real_train_user_test.iloc[i] == predicted_train_user_test.iloc[i]) & (real_train_user_test.iloc[i] == 1))
        true_positive = true_positive + check_true_positive.count(True)
        check_true_negative = list((real_train_user_test.iloc[i] == predicted_train_user_test.iloc[i]) & (real_train_user_test.iloc[i] == 0))
        true_negative = true_negative + check_true_negative.count(True)
        print(f'true_positive: {check_true_positive.count(True)} true_negative: {check_true_negative.count(True)}')

    classification_score = true_negative + true_positive * 9
    print(f'ones: {true_positive} zeroes:{true_negative}  classification_score: {classification_score}')

