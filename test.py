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
        """
        Priporoči filme za podanega uporabnika.
        
        Parameters:
        - userID: ID uporabnika, za katerega priporočamo filme.
        - n: Število priporočenih filmov.
        - rec_seen: Če je False, odstranimo že ogledane filme iz priporočil.

        Returns:
        - Seznam top-n priporočil (ID filma, napovedana ocena).
        """
        # Pridobi napovedi za vse izdelke za uporabnika
        predictions = self.predictor.predict_all(userID)

        # Odstrani že ogledane filme, če rec_seen ni nastavljen na True
        if not rec_seen and userID in self.seen_movies:
            seen = self.seen_movies[userID]
            predictions = {movie_id: rating for movie_id, rating in predictions.items() if movie_id not in seen}

        # Razvrsti napovedi po ocenah v padajočem vrstnem redu
        sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

        return sorted_predictions[:n]


class MovieData:
    def __init__(self, path):
        self.path = path
        self.movies = {}
        self._read_data()

    def _read_data(self):
        with open(self.path, 'r', encoding='latin1 ', errors='replace') as file:
            next(file)

            for line in file:
                fields = line.strip().split('\t')
                if len(fields) < 20:
                    print("input insufficient")
                    continue
                
                id = int(fields[0])

                def safe_convert(value, data_type=float):
                    if value == '\\N' or value == '': 
                        return None
                    try:
                        return data_type(value)
                    except ValueError:
                        return None

                self.movies[id] = {
                    "title": fields[1],
                    "movie_id": safe_convert(fields[2], int),
                    "spanishTitle": fields[3],
                    "imdbPictureURL": fields[4],
                    "year": safe_convert(fields[5], int),
                    "rtID": safe_convert(fields[6], int),
                    "rtAllCriticsRating": safe_convert(fields[7], int),
                    "rtAllCriticsNumReviews": safe_convert(fields[8], int),
                    "rtAllCriticsNumFresh": safe_convert(fields[9], int),
                    "rtAllCriticsNumRotten": safe_convert(fields[10], int),
                    "rtAllCriticsScore": safe_convert(fields[11], int),
                    "rtTopCriticsRating": safe_convert(fields[12], float),
                    "rtTopCriticsNumReviews": safe_convert(fields[13], int),
                    "rtTopCriticsNumFresh": safe_convert(fields[14], int),
                    "rtTopCriticsNumRotten": safe_convert(fields[15], int),
                    "rtTopCriticsScore": safe_convert(fields[16], int),
                    "rtAudienceRating": safe_convert(fields[17], float),
                    "rtAudienceNumRatings": safe_convert(fields[18], int),
                    "rtAudienceScore": safe_convert(fields[19], int),
                    "rtPictureURL": fields[20],
                }

    def get_title(self, movieID):
        return self.movies.get(movieID, {}).get("title", "Film not found") if movieID in self.movies else "Film not found"
    
    
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
    
    def predict_all(self, n=20):
        #return self.testCounter
        #return len(self.similarities.items())
        sorted_similarities = sorted(
            [(movie1, movie2, sim) for movie1, movie2_dict in self.similarities.items() for movie2, sim in movie2_dict.items()],
            key=lambda x: x[2], reverse=True
        )
        return sorted_similarities[:n]


# 1
md = MovieData('data/movies.dat')
uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000)
rp = ItemBasedPredictor()
rec = Recommender(rp)
rec.fit(uim)


""" print("Podobnost med filmoma 'Men in black'(1580) in 'Ghostbusters'(2716): ", rp.similarity(1580, 2716))
print("Podobnost med filmoma 'Men in black'(1580) in 'Schindler's List'(527): ", rp.similarity(1580, 527))
print("Podobnost med filmoma 'Men in black'(1580) in 'Independence day'(780): ", rp.similarity(1580, 780))
print("Predictions for 78: ") """
rec_items = rec.recommend(533, n=15, rec_seen=False)
""" for idmovie, val in rec_items:
    print("Film: {}, ocena: {}".format(md.get_title(idmovie), val)) """