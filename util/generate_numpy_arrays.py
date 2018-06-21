import csv

import numpy as np


def generate_plots_array():
    print('Generating plots array')
    plots = []
    with open('../data_csv/movies_description.csv', 'r', encoding='utf-8') as csv_file:
        movies = csv.DictReader(csv_file)
        for row in movies:
            plots.append([row['id'], row['title'], row['description']])

    plots_np = np.array(plots)
    np.save('../data_numpy/plots', plots_np)


def generate_ratings():
    print('Generating ratings array')
    ratings_array = []
    users = []
    with open('../data_csv/ratings.csv', 'r', encoding='utf-8') as csv_file:
        ratings = csv.DictReader(csv_file)
        for row in ratings:
            if len(users) > 100:
                break

            if row['userId'] not in users:
                users.append(row['userId'])

            ratings_array.append([row['userId'], row['movieId'], row['rating']])

    ratings_np = np.array(ratings_array)
    np.save('../data_numpy/ratings', ratings_np)


generate_ratings()
