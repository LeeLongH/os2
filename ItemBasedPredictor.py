import numpy as np
from collections import defaultdict

class ItemBasedPredictor:
    def __init__(self, min_values=0, threshold=0):
        self.min_values = min_values
        self.threshold = threshold
        self.similarities = {}
        self.movie_ratings = {}
        self.global_avg = 0
        self.testCounter = 0;

    def fit(self, X):
        self.movie_ratings = defaultdict(dict)  # {movie_id: {user_id: rating}}
        for user_id, movie_id, rating, _ in X.data:
            self.movie_ratings[movie_id][user_id] = rating
        
        all_ratings = [rating for movie_ratings in self.movie_ratings.values() for rating in movie_ratings.values()]
        self.global_avg = np.mean(all_ratings)
        
        movie_ids = list(self.movie_ratings.keys())
        self.similarities = defaultdict(dict)
        
        for i, movie1 in enumerate(movie_ids):
            for movie2 in movie_ids[i+1:]:
                sim = self.similarity(movie1, movie2)

                self.testCounter +=1
                
                if sim > 0:
                    self.similarities[movie1][movie2] = sim
                    self.similarities[movie2][movie1] = sim
                    

    def similarity(self, p1, p2):
        # Pridobimo ocene obeh filmov
        users_p1 = self.movie_ratings[p1]
        users_p2 = self.movie_ratings[p2]
        
        # Seznam skupnih uporabnikov
        common_users = set(users_p1.keys()).intersection(users_p2.keys())

        if len(common_users) < self.min_values:
            return 0

        # Pridobimo ocene za skupne uporabnike
        ratings_p1 = np.array([users_p1[user] for user in common_users])
        ratings_p2 = np.array([users_p2[user] for user in common_users])

        # Povprečne ocene uporabnikov
        avg_p1 = np.mean(ratings_p1)
        avg_p2 = np.mean(ratings_p2)

        # Odštejemo povprečja, da "centriramo" ocene
        centered_p1 = ratings_p1 - avg_p1
        centered_p2 = ratings_p2 - avg_p2

        # Izračunamo števnik (dot produkt)
        numerator = np.dot(centered_p1, centered_p2)

        # Izračunamo imenovalec (norme)
        denominator = np.sqrt(np.sum(centered_p1**2)) * np.sqrt(np.sum(centered_p2**2))

        if denominator == 0:
            return 0

        # Izračunamo popravljeno cosinusno podobnost
        sim = numerator / denominator
        return sim if sim >= self.threshold else 0


    def predict(self, user_id):
        predictions = {}
        user_ratings = {movie_id: rating for movie_id, rating in self.movie_ratings.items() if user_id in rating}

        for movie_id in self.movie_ratings.keys():
            # already rated by user
            if movie_id in user_ratings:
                continue

            weighted_sum = 0
            sim_sum = 0

            for rated_movie, user_rating in user_ratings.items():
                if movie_id in self.similarities and rated_movie in self.similarities[movie_id]: # in both sim(1,2) and sim(2,1)
                    sim = self.similarities[movie_id][rated_movie]
                    weighted_sum += sim * user_rating
                    sim_sum += abs(sim)

            if sim_sum > 0:
                predictions[movie_id] = weighted_sum / sim_sum
            else:
                predictions[movie_id] = 0

        return predictions
    
    def get_top_most_similar_pairs(self, n=20):
        #return self.testCounter
        #return len(self.similarities.items())
        sorted_similarities = sorted(
            [(movie1, movie2, sim) for movie1, movie2_dict in self.similarities.items() for movie2, sim in movie2_dict.items()],
            key=lambda x: x[2], reverse=True
        )
        return sorted_similarities[:n]
