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
    