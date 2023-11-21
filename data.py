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
        user_artists_df = pd.read_csv(self.user_artists_file)
        self.user_artists = self.create_user_artists_matrix(user_artists_df)

        artists_df = pd.read_csv(self.artists_file, usecols=['id', 'name'])
        self.artist_names = dict(zip(artists_df['id'], artists_df['name']))

    def create_user_artists_matrix(self, user_artists_df):
        # Print the entire matrix
        print(user_artists_df.pivot_table(index='userID', columns='artistID', values='weight', fill_value=0))

        unique_users = user_artists_df['userID'].unique()
        unique_artists = user_artists_df['artistID'].unique()

        user_artist_matrix = np.zeros((len(unique_users), len(unique_artists)))

        for _, row in user_artists_df.iterrows():
            user_idx = np.where(unique_users == row['userID'])[0][0]
            artist_idx = np.where(unique_artists == row['artistID'])[0][0]
            user_artist_matrix[user_idx, artist_idx] = row['weight']

        return csr_matrix(user_artist_matrix)  # Convert to csr_matrix

    def get_artist_name_from_id(self, artist_id):
        return self.artist_names.get(artist_id, "Unknown")

if __name__ == "__main__":
    collaborative_filtering = CollaborativeFiltering(
        user_artists_file=Path("../Resources/user_artists.csv"),
        artists_file=Path("../Resources/artists.csv")
    )
    print(collaborative_filtering.user_artists)  # This line prints the entire matrix

    artist_name = collaborative_filtering.get_artist_name_from_id(3)
    print(artist_name)