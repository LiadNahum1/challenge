import pandas as pd
USER_TRAIN =10
USER_COUNT = 40
SEGMENT_COUNT = 150
TRAIN_SEGMENT_COUNT = 50
ONES = 300
ZEROES = 2700

if __name__ == "__main__":
    challenge_filled = pd.read_csv('challengeToFill.csv')
    challenge_filled = challenge_filled.drop(challenge_filled.columns[0], axis = 1)
    print(challenge_filled)
    challenge_filled_test = challenge_filled.iloc[USER_TRAIN:USER_COUNT, TRAIN_SEGMENT_COUNT:SEGMENT_COUNT]
    print(challenge_filled_test)
    all_data_frame = []
    for i in range(0, USER_COUNT-USER_TRAIN):
        all_data_frame.extend(challenge_filled_test.iloc[i])
    ones = all_data_frame.count(1)
    zeroes = all_data_frame.count(0)
    classification_score = zeroes + min(ones, ONES)*9 #assuming all True ones are predicted
    print(f'ones: {min(ones, ONES)} zeroes:{zeroes}  classification_score: {classification_score}')
    grade = 0.7*(min(100, (classification_score/4575)*95))
    print(grade)
