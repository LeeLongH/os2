
from collections import defaultdict
import numpy as np
import pandas as pd
from math import sqrt

class ItemBasedPredictor1:
    def __init__(self, min_values=0, threshold=0):
        self.min_values = min_values
        self.threshold = threshold
        self.similarities = {}
        self.movie_ratings = defaultdict(lambda: defaultdict(float))
        self.all_fit_movies = []
        self.global_avg = 0
    
    def fit(self, X):
        all_ratings = [rating for _, _, rating, _ in X.data]
        self.global_avg = np.mean(all_ratings)
        
        for user_id, movie_id, rating, _ in X.data:
            self.movie_ratings[movie_id][user_id] = rating
        
        for movie_id, _ in self.movie_ratings.items():
            self.all_fit_movies.append(movie_id)

        for i, movie_id1 in enumerate(self.all_fit_movies):
            for movie_id2 in self.all_fit_movies[i + 1:]:
                sim = self.similarity(movie_id1, movie_id2)
                self.similarities[(movie_id1, movie_id2)] = sim
                self.similarities[(movie_id2, movie_id1)] = sim
                    
    
    def similarity(self, p1, p2):
        common_users = set(self.movie_ratings[p1].keys()).intersection(self.movie_ratings[p2].keys())
        
        # popravljena cosinusna podobnost
        num = sum((self.movie_ratings[p1][user] - self.global_avg) * (self.movie_ratings[p2][user] - self.global_avg) for user in common_users)
        denom_p1 = sqrt(sum((self.movie_ratings[p1][user] - self.global_avg) ** 2 for user in common_users))
        denom_p2 = sqrt(sum((self.movie_ratings[p2][user] - self.global_avg) ** 2 for user in common_users))
        
        if denom_p1 == 0 or denom_p2 == 0:
            return 0
        
        sim = num / (denom_p1 * denom_p2)
        
        if sim < self.threshold:
            return 0
        
        return sim

    def predict(self, userID):
        predictions = {}
        
        for movie_id in self.all_fit_movies:
            if movie_id not in self.movie_ratings or userID in self.movie_ratings[movie_id]:
                continue
            
            weighted_sum = 0
            sim_sum = 0
            
            for other_movie_id in self.all_fit_movies:
                if movie_id == other_movie_id:
                    continue
                
                sim = self.similarities.get((movie_id, other_movie_id), 0)
                if sim > 0:
                    weighted_sum += sim * (self.movie_ratings[other_movie_id].get(userID, self.global_avg) - self.global_avg)
                    sim_sum += abs(sim)
            
            if sim_sum > 0:
                predicted_rating = self.global_avg + weighted_sum / sim_sum
            else:
                predicted_rating = self.global_avg
            
            predictions[movie_id] = predicted_rating
        
        return predictions
    
    def print_top_20_similar_movies(self, md):
        similar_pairs = []
        
        for (movie_id1, movie_id2), sim in self.similarities.items():
            similar_pairs.append((movie_id1, movie_id2, sim))
        
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        for i in range(min(20, len(similar_pairs))):
            movie1, movie2, similarity = similar_pairs[i]
            title1 = md.get_title(movie1)
            title2 = md.get_title(movie2)
            print(f"Film1: {title1}, Film2: {title2}, podobnost: {similarity}")