import pickle
from datetime import datetime

class UserItemData:
    def __init__(self, path, from_date=None, to_date=None, min_ratings=None):
        self.path = path
        self.from_date = datetime.strptime(from_date, '%d.%m.%Y') if from_date else None
        self.to_date = datetime.strptime(to_date, '%d.%m.%Y') if to_date else None
        self.min_ratings = min_ratings
        self.data = []
        self._read_data()

    def _read_data(self):
        with open(self.path, 'r', encoding='utf-8') as file:
            ratings_count = {}
            next(file)

            for line in file:
                fields = line.strip().split('\t')
                if len(fields) < 9:
                    print("input insufficient")
                    continue

                user_id = int(fields[0])
                movie_id = int(fields[1])
                rating = float(fields[2])

                date_time = datetime(
                    year=int(fields[5]),
                    month=int(fields[4]),
                    day=int(fields[3]),
                    hour=int(fields[6]),
                    minute=int(fields[7]),
                    second=int(fields[8])
                )
                #print(date_time)

                # conditions
                if self.from_date and date_time < self.from_date:
                    continue
                if self.to_date and date_time > self.to_date:
                    continue

                # movies, number of ratings
                ratings_count[movie_id] = ratings_count.get(movie_id, 0) + 1

                # conditions
                if self.min_ratings and ratings_count[movie_id] < self.min_ratings:
                    continue

                self.data.append((user_id, movie_id, rating, date_time))

    def nratings(self):
        return len(self.data)

    def save_data(self, save_path):
        with open(save_path, 'wb') as file:
            pickle.dump(self.data, file)

    def load_data(self, load_path):
        with open(load_path, 'rb') as file:
            self.data = pickle.load(file)
