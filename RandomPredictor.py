import random

class RandomPredictor:
    def __init__(self, min_rating, max_rating):
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.movies = {}

    def fit(self, X):
        # prepare dict
        self.movies = {movie_id: None for _, movie_id, _, _ in X.data}

    def predict(self, user_id):
        pred = {}
        for movie_id in self.movies.keys():
            pred[movie_id] = random.randint(self.min_rating, self.max_rating)
        return pred

