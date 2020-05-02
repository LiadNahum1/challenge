from sklearn.feature_extraction.text import TfidfVectorizer

USER_COUNT = 2
WORDS_COUNT = 10
WORDS_COUNT_PER_SEGMENT = 100
SEGMENT_COUNT = 50


def get_fifty_segments():
    users_lines = []
    for i in range(0, USER_COUNT):
        f = open("FraudedRawData/User" + str(i), "r")
        lines = ""
        for j in range(0, WORDS_COUNT):
            lines += f.readline()[:-1] + ' '
        users_lines.append(lines[:-1])
    return users_lines


# per user
def separate_user_to_segment(user_id):
    user_segments = []
    file_of_user = open("FraudedRawData/User" + str(user_id), "r")
    for i in range(0, SEGMENT_COUNT):
        lines = ""
        for j in range(0, WORDS_COUNT_PER_SEGMENT):
            lines += file_of_user.readline()[:-1] + ' '
        user_segments.append(lines[:-1])
    return user_segments

def build_word_dict():
    pass
def count_word_occurrence(words):
    d = dict()
    for word in words:
        if word in d:
            # Increment count of word by 1
            d[word] = d[word] + 1
        else:
            d[word] = 1
    return d

def main():
    print(command_frequency_per_segment(0))
    # vectorizer = TfidfVectorizer()
    # tf_idf_matrix = vectorizer.fit_transform(get_fifty_segments())
    # print(vectorizer.get_feature_names())


main()
