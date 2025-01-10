import numpy as np

class STDPredictor:
    def __init__(self, n):
        self.n = n
        self.std_devs = {}
        self.movie_ratings = {}

    def fit(self, X):
        for _, movie_id, rating, _ in X.data:
            if movie_id not in self.movie_ratings:
                self.movie_ratings[movie_id] = []
            self.movie_ratings[movie_id].append(rating)

        for movie_id, ratings in self.movie_ratings.items():
            if len(ratings) >= self.n:
                std_dev = np.std(ratings)
                self.std_devs[movie_id] = std_dev

    def predict(self, userID):
        sorted_movies = sorted(self.std_devs.items(), key=lambda x: x[1], reverse=True)
        return sorted_movies
