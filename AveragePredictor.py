class AveragePredictor:
    def __init__(self, b):
        self.b = b
        self.movie_averages = {}
        self.global_avg = 0
        self.user_ratings = {}

    def fit(self, X):
        total_ratings = 0
        total_count = 0
        movie_ratings = {}
        user_ratings = {}

        # Loop thu all ratings to calc the avg movie rating
        for user_id, movie_id, rating, _ in X.data:
            if movie_id not in movie_ratings:
                movie_ratings[movie_id] = {'sum': 0, 'count': 0}
            movie_ratings[movie_id]['sum'] += rating
            movie_ratings[movie_id]['count'] += 1

            if user_id not in user_ratings:
                user_ratings[user_id] = {}
            user_ratings[user_id][movie_id] = rating

            total_ratings += rating
            total_count += 1

        self.global_avg = total_ratings / total_count if total_count > 0 else 0

        for movie_id, stats in movie_ratings.items():
            vs = stats['sum']
            n = stats['count']
            avg_rating = (vs + self.b * self.global_avg) / (n + self.b)
            self.movie_averages[movie_id] = avg_rating

        self.user_ratings = user_ratings

    def predict(self, userID):
        if userID not in self.user_ratings:
            return "No ratings found for this user."

        rated_movies = self.user_ratings[userID]
        
        unrated_movies = {movie_id: avg_rating for movie_id, avg_rating in self.movie_averages.items() if movie_id not in rated_movies}
        
        sorted_unrated_movies = sorted(unrated_movies.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_unrated_movies
