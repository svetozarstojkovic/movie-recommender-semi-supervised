import csv


def generate_id_title_description_csv():
    with open('../data_csv/movies_metadata.csv', 'r', encoding='utf-8') as csvfile:
        with open('../data_csv/movies_description.csv', 'w', encoding='utf-8') as output:
            movies = csv.DictReader(csvfile)
            output.write('id,title,description\n')
            for row in movies:
                output.write(str(row['id']) + ',' +
                             str(row['title']) + ',"' +
                             str(row['overview'].replace("\"", "")) +
                             '"\n')
