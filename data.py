from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

class CollaborativeFiltering:
    def __init__(self, user_artists_file, artists_file):
        self.user_artists_file = user_artists_file
        self.artists_file = artists_file
        self.user_artists = None
        self.artist_names = None
        self.load_data()

    def load_data(self):
        # Load user artists data
        user_artists_df = pd.read_csv(self.user_artists_file, sep="\t")
        self.user_artists = self.create_user_artists_matrix(user_artists_df)

        # Load artist names
        artists_df = pd.read_csv(self.artists_file, sep="\t")
        self.artist_names = dict(zip(artists_df['id'], artists_df['name']))

    def create_user_artists_matrix(self, user_artists_df):
        user_artists_matrix = user_artists_df.pivot_table(index='userID', columns='artistID', values='weight', fill_value=0).to_numpy()
        user_artists_csr = csr_matrix(user_artists_matrix)
        return user_artists_csr

    def get_artist_name_from_id(self, artist_id):
        return self.artist_names.get(artist_id, "Unknown")

if __name__ == "__main__":
    collaborative_filtering = CollaborativeFiltering(
        user_artists_file=Path("../Resources/user_artists.csv"),
        artists_file=Path("../Resources/artists.csv")
    )

    # Access the user-artist matrix
    print(collaborative_filtering.user_artists)

    # Access artist name by ID
    artist_name = collaborative_filtering.get_artist_name_from_id(1)
    print(artist_name)