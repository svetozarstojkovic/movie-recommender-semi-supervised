import operator
import os
import random
from math import sqrt

import numpy as np
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

from enums.Types import VectorizerType, SemiSupervisedAlgorithms
from util.array_loader import get_user_movies, get_ids_movies_plots


def give_prediction_for_user(user_id, vectorizer_type, semi_supervised_algorithm):
    train_plots, classes, unknown_plots, train_names, unknown_names = get_train_and_unknown(user_id, vectorizer_type)

    model = model_and_fit(semi_supervised_algorithm, train_plots, classes)

    predicted_ratings = {}
    for i, rating in enumerate(model.predict(unknown_plots)):
        predicted_ratings[unknown_names[i]] = rating

    predicted_ratings = sorted(predicted_ratings.items(), key=operator.itemgetter(1), reverse=True)

    save_movies_for_prediction(user_id, train_names, classes, predicted_ratings)


def get_train_and_unknown(user_id, vectorizer_type):
    train_names_temp = []
    train_names = []
    classes = {}
    plots = []
    train_plots_indexes = []
    unknown_plots_indexes = []
    unknown_names = []

    for i, movie_id_title_rating_plot in enumerate(get_user_movies(user_id=user_id)):
        title = movie_id_title_rating_plot[1]
        rating = movie_id_title_rating_plot[2]
        plot = movie_id_title_rating_plot[3]

        train_names_temp.append(title)
        classes[title] = rating

    print('Done with rated movies')
    all_movies = get_ids_movies_plots()
    random_indexes = random.sample(range(len(all_movies)), 1000)
    print('All movies loaded')
    for i, movie_id_title_plot in enumerate(all_movies):
        title = movie_id_title_plot[1]
        plot = movie_id_title_plot[2]

        plots.append(plot)

        if title in train_names_temp:
            train_names.append(title)
            train_plots_indexes.append(i)
        elif i in random_indexes:
            train_names.append(title)
            classes[title] = -1
            train_plots_indexes.append(i)
        else:
            unknown_names.append(title)
            unknown_plots_indexes.append(i)

    print('Done with unrated movies')
    plots_vectorized = text_vectorizer(vectorizer_type, plots)
    print('vectorization done')

    train_plots = plots_vectorized[train_plots_indexes]
    unknown_plots = plots_vectorized[unknown_plots_indexes]

    train_plots = process_pca(train_plots.toarray())
    unknown_plots = process_pca(unknown_plots.toarray())
    print('pca done')

    output_classes = []
    for title in train_names:
        output_classes.append(classes[title])

    return train_plots, output_classes, unknown_plots, train_names, unknown_names


def save_movies_for_prediction(user_id, train_names, classes, title_predicted_rating):
    directory = '../recommendation/output' + str(user_id) + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(directory + 'train.csv', 'w', encoding='utf-8') as movies_file:
        movies_file.write('title,actual rating\n')
        for i, name in enumerate(train_names):
            movies_file.write(str(name) + ',' +
                              str(classes[i] / 2 if classes[i] != -1 else -1) + '\n')

    with open(directory + 'recommendation.csv', 'w', encoding='utf-8') as rec_file:
        rec_file.write('title,predicted rating\n')
        for i, movie_rating in enumerate(title_predicted_rating):
            title = movie_rating[0]
            predicted_rating = movie_rating[1] / 2

            rec_file.write(title + ',' + str(predicted_rating) + '\n')


def evaluate_prediction(user_id, vectorizer_type, semi_supervised_algorithm):
    train, classes, test, actual_ratings, train_names, test_names = generate_train_and_test(user_id, vectorizer_type)

    model = model_and_fit(semi_supervised_algorithm, train.toarray(), classes)

    predicted_ratings = []
    for i, rating in enumerate(model.predict(test.toarray())):
        predicted_ratings.append([test_names[i], rating])

    save_movies_for_evaluation(user_id, train_names, classes, predicted_ratings, actual_ratings)
    save_output(user_id, predicted_ratings, actual_ratings, vectorizer_type, semi_supervised_algorithm)


def generate_train_and_test(user_id, vectorizer_type):
    train_names = []
    classes = []

    test_names = []
    actual_ratings = {}

    plots = []
    plots_train_index = []
    plots_test_index = []

    stemmer = LancasterStemmer()

    for i, movie_id_title_rating_plot in enumerate(get_user_movies(user_id=user_id)):
        title = movie_id_title_rating_plot[1]
        rating = movie_id_title_rating_plot[2]
        plot = movie_id_title_rating_plot[3]

        stemmed_words = []
        words = word_tokenize(plot)
        for word in words:
            stemmed_words.append(stemmer.stem(word))

        plot = " ".join(stemmed_words)

        plots.append(plot)

        if i % 3 == 0:
            train_names.append(title)
            classes.append(rating)
            plots_train_index.append(i)
        else:
            test_names.append(title)
            plots_test_index.append(i)
            actual_ratings[title] = rating

    for _ in range(len(classes) // 3):
        classes[np.random.randint(len(classes))] = -1

    plots_vectorized = text_vectorizer(vectorizer_type, plots)

    train_plots = plots_vectorized[plots_train_index]
    test_plots = plots_vectorized[plots_test_index]

    return train_plots, classes, test_plots, actual_ratings, train_names, test_names


def save_movies_for_evaluation(user_id, train_names, classes, title_predicted_rating, test_ratings):
    directory = '../recommendation/output' + str(user_id) + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(directory + 'movies.csv', 'w', encoding='utf-8') as movies_file:
        movies_file.write('type,title,actual rating,predicted rating\n')
        for i, name in enumerate(train_names):
            movies_file.write('train,' +
                              str(name) + ',' +
                              str(classes[i] / 2 if classes[i] != -1 else -1) +
                              ',none\n')

        for i, movie_rating in enumerate(title_predicted_rating):
            title = movie_rating[0]
            predicted_rating = movie_rating[1] / 2
            actual_rating = test_ratings[movie_rating[0]] / 2

            movies_file.write('output,' +
                              title + ',' +
                              str(actual_rating) + ',' +
                              str(predicted_rating) + '\n')


def save_output(user_id, title_predicted_rating, test_ratings, vectorizer_type, semi_supervised_algoritm):
    directory = '../recommendation/output' + str(user_id) + '/'
    file = directory + 'output.csv'
    if not os.path.isfile(file):
        with open(file, 'w', encoding='utf-8') as output_file:
            output_file.write('vectorizer type,semi supervised algorithm,rmse\n')

    with open(file, 'a', encoding='utf-8') as output_file:
        actual_ratings = []
        predicted_ratings = []
        for i, movie_rating in enumerate(title_predicted_rating):
            predicted_rating = movie_rating[1] / 2
            actual_rating = test_ratings[movie_rating[0]] / 2

            predicted_ratings.append(predicted_rating)
            actual_ratings.append(actual_rating)

        rmse = sqrt(mean_squared_error(actual_ratings, predicted_ratings))
        output = str(vectorizer_type.name) + ',' + str(semi_supervised_algoritm.name) + ',' + str(rmse)
        output_file.write(output + '\n')

        print(output)


def model_and_fit(type, train_vector, classes):
    if type == SemiSupervisedAlgorithms.LABEL_PROPAGATION:
        model = LabelPropagation()
        model.fit(train_vector, classes)
        return model
    elif type == SemiSupervisedAlgorithms.LABEL_SPREADING:
        from scipy.sparse import csgraph
        model = LabelSpreading(kernel='rbf')
        model.fit(train_vector, classes)
        return model
    else:
        raise ValueError('Wrong semi supervised model type!')


def text_vectorizer(type, plots):
    if type == VectorizerType.HASHING:
        return text_vectorizer_hashing_vectorizer(plots)
    elif type == VectorizerType.TFIDF:
        return text_vectorizer_tfidf(plots)
    elif type == VectorizerType.COUNT:
        return text_vectorizer_count(plots)
    else:
        raise ValueError('Wrong text vectorizer')


def text_vectorizer_hashing_vectorizer(plots):
    vectorizer = HashingVectorizer(n_features=1000)
    return vectorizer.transform(plots)


def text_vectorizer_tfidf(plots):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(plots)


def text_vectorizer_count(plots):
    vectorizer = CountVectorizer(analyzer='word')
    return vectorizer.fit_transform(plots)


def process_pca(data):
    pca = PCA(n_components=10)
    pca.fit(data)
    return pca.transform(data)


def do_evaluation_for_user(user_id):
    for vectorizer in VectorizerType:
        for algorithm in SemiSupervisedAlgorithms:
            for _ in range(5):
                evaluate_prediction(user_id=user_id,
                                    vectorizer_type=vectorizer,
                                    semi_supervised_algorithm=algorithm)


# evaluation
# user 46 has the biggest number of rated movies
# do_evaluation_for_user(46)
# do_evaluation_for_user(4)


# do a prediction
# must use HASHING because corpus is big and memory error occurres
give_prediction_for_user(46, VectorizerType.HASHING, SemiSupervisedAlgorithms.LABEL_PROPAGATION)
