from pathlib import Path
from typing import Tuple, List
import implicit
from scipy.sparse import csr_matrix
import pandas as pd
from data import CollaborativeFiltering

class ImplicitRecommender:
    def __init__(self, collaborative_filtering: CollaborativeFiltering, implicit_model: implicit.recommender_base.RecommenderBase):
        self.collaborative_filtering = collaborative_filtering
        self.implicit_model = implicit_model

    def fit(self, user_artists_matrix: csr_matrix) -> None:
        self.implicit_model.fit(user_artists_matrix)
    #Tweaking n here will depend on our use case and user preferences, playing with it will find a balance between providing sufficient recommendations and ensuring their relevance.
    def recommend(self, user_id: int, user_artists_matrix: csr_matrix, n: int = 5) -> Tuple[List[str], List[float]]:
        artist_ids, scores = self.implicit_model.recommend(user_id, user_artists_matrix.getrow(user_id), N=n)
        artists = [self.collaborative_filtering.get_artist_name_from_id(artist_id) for artist_id in artist_ids]
        return artists, scores

if __name__ == "__main__":
    collaborative_filtering = CollaborativeFiltering(
        user_artists_file=Path("../Resources/user_artists.csv"),
        artists_file=Path("../Resources/artists.csv")
    )

    # Load user artists data
    user_artists_df = pd.read_csv(collaborative_filtering.user_artists_file)

    # Load user artists matrix
    user_artists_matrix = collaborative_filtering.create_user_artists_matrix(user_artists_df)

    # Instantiate ALS using implicit (Tweak these parameters for different results)
    implicit_model = implicit.als.AlternatingLeastSquares(factors=100, iterations=20, regularization=0.1)

    # Instantiate recommender, fit, and recommend
    recommender = ImplicitRecommender(collaborative_filtering, implicit_model)
    recommender.fit(user_artists_matrix)   #user id no:             #no. of artists returned:
    artists, scores = recommender.recommend(615, user_artists_matrix, n=10)

    # Print results
    for artist, score in zip(artists, scores):
        print(f"{artist}: {score}")