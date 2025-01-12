import numpy as np
import pandas as pd
from collections import defaultdict
from math import sqrt

class ItemBasedPredictor:
    def __init__(self, min_values=0, threshold=0):
        self.min_values = min_values
        self.threshold = threshold
        self.user_item_matrix = None
        self.user_item_matrix_centered = None
        self.similarity_matrix = None
        self.global_avg = 0

    def fit(self, x):
        if not hasattr(x, 'data'):
            raise ValueError("Input object must have a 'data' attribute containing user, movie, and rating information.")

        data_df = pd.DataFrame(x.data, columns=['userID', 'movieID', 'rating', 'dateTime'])
        
        all_ratings = [rating for _, _, rating, _ in x.data]
        self.global_avg = np.mean(all_ratings)

        self.user_item_matrix = data_df.pivot(index='userID', columns='movieID', values='rating')
        
        self.user_item_matrix_centered = self.user_item_matrix.sub(self.user_item_matrix.mean(axis=1), axis=0)

        self.similarity_matrix = pd.DataFrame(index=self.user_item_matrix.columns, columns=self.user_item_matrix.columns, dtype=float)

        print(f"Shape of user_item_matrix: {self.user_item_matrix.shape}")  # Izpis dimenzij matrike uporabniških ocen

        for p1 in self.user_item_matrix.columns:
            for p2 in self.user_item_matrix.columns:
                if p1 == p2:
                    self.similarity_matrix.at[p1, p2] = 1.0
                elif pd.isnull(self.similarity_matrix.at[p1, p2]):
                    sim = self.similarity(p1, p2)
                    self.similarity_matrix.at[p1, p2] = sim
                    self.similarity_matrix.at[p2, p1] = sim

        print(f"Similarity matrix calculated.")

    def similarity(self, p1, p2):
        if p1 not in self.user_item_matrix.columns or p2 not in self.user_item_matrix.columns:
            return 0

        p1_ratings = self.user_item_matrix_centered[p1]
        p2_ratings = self.user_item_matrix_centered[p2]

        common_items = p1_ratings.dropna().index.intersection(p2_ratings.dropna().index)

        if len(common_items) < self.min_values:
            return 0

        p1_common = p1_ratings.loc[common_items]
        p2_common = p2_ratings.loc[common_items]

        numerator = np.dot(p1_common, p2_common)
        denominator = np.sqrt(np.dot(p1_common, p1_common)) * np.sqrt(np.dot(p2_common, p2_common))

        if denominator == 0:
            return 0

        sim = numerator / denominator
        return sim if sim >= self.threshold else 0

    def predict(self, user_id):
        if user_id not in self.user_item_matrix.index:
            print(f"UserID {user_id} not found in the user_item_matrix. Returning empty predictions.")
            return {}

        user_ratings = self.user_item_matrix.loc[user_id]
        predictions = {}

        for movie_id in self.user_item_matrix.columns:
            if not pd.isnull(user_ratings[movie_id]):
                predictions[movie_id] = user_ratings[movie_id]
            else:
                numerator, denominator = 0, 0
                for rated_movie_id, rating in user_ratings.dropna().items():
                    sim = (
                        self.similarity_matrix.at[movie_id, rated_movie_id]
                        if movie_id in self.similarity_matrix.index and rated_movie_id in self.similarity_matrix.columns
                        else 0
                    )
                    numerator += sim * rating
                    denominator += sim

                predictions[movie_id] = numerator / denominator if denominator != 0 else self.global_avg

        return predictions

    def print_top_20_similar_movies(self, md):
        similar_pairs = []

        for movie_id1 in self.similarity_matrix.columns:
            for movie_id2 in self.similarity_matrix.columns:
                if movie_id1 != movie_id2:
                    sim = self.similarity_matrix.at[movie_id1, movie_id2]
                    if sim > 0:
                        similar_pairs.append((movie_id1, movie_id2, sim))

        if len(similar_pairs) == 0:
            print("No similar movies found.")

        similar_pairs.sort(key=lambda x: x[2], reverse=True)

        print(f"Top 20 Similar Movies:")
        for i in range(min(20, len(similar_pairs))):
            movie1, movie2, similarity = similar_pairs[i]
            title1 = md.get_title(movie1)  # Pridobimo ime filma
            title2 = md.get_title(movie2)
            
            if title1 is None or title2 is None:
                print(f"Missing movie titles for movie IDs: {movie1}, {movie2}")
                continue

            print(f"Film1: {title1}, Film2: {title2}, podobnost: {similarity}")
