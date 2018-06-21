import operator
import os

import numpy as np


# movie_id, title, rating, plot
def get_user_movies(user_id):
    user_movies_id_rating = []
    movie_id_title_rating_plot = []
    ratings = get_ratings()

    for rating in ratings:
        current_user_id = int(rating[0])
        current_movie_id = rating[1]
        current_rating = int(float(rating[2])) * 2

        if current_user_id == user_id:
            user_movies_id_rating.append([current_movie_id, current_rating])

    for movie_id_title_plot in get_ids_movies_plots():
        movie_id = movie_id_title_plot[0]
        title = movie_id_title_plot[1]
        plot = movie_id_title_plot[2]

        for movie_id_rating in user_movies_id_rating:
            if movie_id_rating[0] == movie_id:
                movie_id_title_rating_plot.append([movie_id, title, movie_id_rating[1], plot])

    return movie_id_title_rating_plot


# movie_id, title, plot
def get_ids_movies_plots():
    return get_numpy_array('../data_numpy/plots.npy')


def number_of_rated_movies():
    users = {}
    for rating in get_ratings():
        user_id = rating[0]
        if user_id in users.keys():
            users[user_id] = users.get(user_id) + 1
        else:
            users[user_id] = 1

    print(sorted(users.items(), key=operator.itemgetter(1), reverse=True))


# user_id, movie_id, rating
def get_ratings():
    return get_numpy_array('../data_numpy/ratings.npy')


def get_numpy_array(file_location):
    if os.path.isfile(file_location):
        return np.load(file_location)
    else:
        raise ValueError('File location not existing')

