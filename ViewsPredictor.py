class ViewsPredictor:
    def __init__(self):
        self.movie_views = {}

    def fit(self, X):
        for _, movie_id, _, _ in X.data:
            if movie_id not in self.movie_views:
                self.movie_views[movie_id] = 0
            self.movie_views[movie_id] += 1

    def predict(self):
        sorted_movie_views = sorted(self.movie_views.items(), key=lambda x: x[1], reverse=True)
        return sorted_movie_views
