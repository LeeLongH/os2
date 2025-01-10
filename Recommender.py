class Recommender:
    def __init__(self, predictor):
        self.predictor = predictor
        self.seen_movies = {}

    def fit(self, X):
        # array of dicts - each person has watched movies
        self.seen_movies = {}
        for user_id, movie_id, _, _ in X.data:
            if user_id not in self.seen_movies:
                self.seen_movies[user_id] = set()
            self.seen_movies[user_id].add(movie_id)

        self.predictor.fit(X)

    def recommend(self, userID, n=10, rec_seen=True):
        predictions = self.predictor.predict(userID)

        if not rec_seen and userID in self.seen_movies:
            seen = self.seen_movies[userID]
            
            if hasattr(predictions, 'items'):
                predictions = {movie_id: rating for movie_id, rating in predictions.items() if movie_id not in seen}
            else:
                predictions = {movie_id: rating for movie_id, rating in predictions if movie_id not in seen}

        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

        return sorted_predictions[:n]
