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

    def recommend(self, user_id: int, user_artists_matrix: csr_matrix, n: int = 10) -> Tuple[List[str], List[float]]:
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

    # Instantiate ALS using implicit
    implicit_model = implicit.als.AlternatingLeastSquares(factors=50, iterations=10, regularization=0.01)

    # Instantiate recommender, fit, and recommend
    recommender = ImplicitRecommender(collaborative_filtering, implicit_model)
    recommender.fit(user_artists_matrix)
    artists, scores = recommender.recommend(45, user_artists_matrix, n=20)

    # Print results
    for artist, score in zip(artists, scores):
        print(f"{artist}: {score}")