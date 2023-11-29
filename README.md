# Music Recommendation Model:musical_note:

This project focused on creating a **music recommendation system** using collaborative filtering to create an output of 10 recommended artists based on the [last.fm](https://grouplens.org/datasets/hetrec-2011/) dataset. 

# About our project

Recommendation systems are used globally to improve businesses. It helps the product become catered toward the customer, and provides a sense of personalization. The goal of this project was to try an create a version of a music recommendation model, which is used by companies like spotify or apple music. Although our model is nothing as complex as what those companies are using, this project helped provide us with deeper understanding of machine learning techniques. 

Contributers:

- Thom Banninga
- Julia Khron
- Aron Lecznar
- Natalia Phipps

# Data Visualizations

We wanted to create some visualizations in regards various music features and trends. The visualizations can be found in this tableau link: https://public.tableau.com/app/profile/aron3046/viz/Spotify_Song_Visualizations/Story1. Although this is a different dataset than the one we used in our model, it helped provide insights into how music has evolved over time. Here are a few screenshots of those visuals:

1.) Features of music over time:

![image](https://github.com/jules-krohn/Project4_music/assets/130694752/17a9a276-3cf2-4eb2-bac0-cd454af4dd75)

 - This visual explores different music features and how they have evolved over time. The trend shows that acousticness has had an overall decrease, while features such as speechiness and loudness have increased.

2.) Popularity based on attribute:

![image](https://github.com/jules-krohn/Project4_music/assets/130694752/b3a375da-d930-476b-b0ac-fb6d6c6c5d6f)

 - This visualization allows us to see how popularity changes depending on song features. Some features have a strong correlation with popularity such as loudness, while acousticness shows a negative correlation.





## The Dataset:
The dataset is a library of networking, tagging, and music artist listening information from over 2,0000 users of the last.fm music system. The csv files we used were organized as:
* A list of artists with a unique ID and url to their page on last.fm
* A list of user-given tags (many were genre, some were the general mood or thought about the artist), each with a unique ID
* A list of user-artist pairs and a "weight", which is the number of times a user listened to that artist
* A list of tag assignments assigned to each artist by each user


## Technologies

- Python
- SQL
- tableau 
- Pyspark
- Implicit
- Pandas
- Sklearn
- lightfm
- numpy

# How we created the model

**Data Loading**:

- Reads user-artists and artists CSV files using Spark.
  
- Converts Spark DataFrames to Pandas for easier processing.
  
**CollaborativeFiltering Class**:

_Initialization (Lines 25-38)_

 - The CollaborativeFiltering class is initialized with the paths to the user-artists and artists CSV files.

 - Data is loaded from these CSV files into Spark DataFrames and then converted to Pandas DataFrames `(user_artists_df and artists_df)`.

 - The class contains methods to create a user-artist matrix `(create_user_artists_matrix)` and retrieve artist names from their IDs `(get_artist_name_from_id)`.

**ImplicitRecommender Class**

_Initialization (Lines 40-50)_

 - The ImplicitRecommender class is initialized with an instance of the CollaborativeFiltering class and an implicit model using the `implicit` library.

 - The class fits the model to the user artist matrix to generate recommendations for the user: `self.implicit_model.fit(user_artists_matrix)`. 

 - Recommendations are normalized between 0 and 1 using min-max scaling.

**LightFMRecommender Class**
 
_Initialization (Lines 52-62)_

 - The LightFMRecommender class is initialized similarly to the ImplicitRecommender class.

 - It uses the `LightFM` library for collaborative filtering.

 - Methods include fitting the model (fit) and generating recommendations for a user (recommend).

 - Precision at k is calculated as a measure of model performance.

**Data Loading and Processing**

_Load and Process Data (Lines 70-83)_

 - An instance of CollaborativeFiltering is created, and data is loaded and processed.

 - A sample user-artist matrix is created for analysis.

 - The user-artist matrix is normalized using BM25 weighting:normalized_train_matrix = `bm25_weight(user_artists_matrix)`

**Implicit ALS Model**

_Implicit ALS Model (Lines 92-106)_

 - An implicit ALS model is instantiated using the `implicit` library.

 - An instance of ImplicitRecommender is created and fitted with the normalized user-artist matrix.

 - Top recommendations for a user (e.g., user 50) are printed.

**Implicit ALS Model with Hyperparameter Tuning**

_(Lines 108-122)_

 - Another instance of the implicit ALS model is created with hyperparameter tuning.

 - Recommendations for a different user (e.g., user 4) are printed.

**LightFM Model**

_LightFM Model (Lines 124-146)_

 - A LightFM model is instantiated and fitted with the normalized user-artist matrix.

 - Recommendations and precision at k are printed for a user (e.g., user 50).


## Optimization Process:

At the beginning of creating our model, the code was very simple and only generated the top 10 artists for any given user. We knew that we wanted to continue improving the model so many different techniques were implemented, and through trial and error we were able to slightly improve the recommender. Below are a list of steps that were implemented:

1.) Get accuracy scores for our data:

A number of different libraries were used to help us understand how well our model was at recommending artists. The `surprise` library was used to generate RMSE and MAE scores, which gave us evidence of our model having a low prediction accuracy. From here we tried to implement more techniques to improve our accuracy.

![image](https://github.com/jules-krohn/Project4_music/assets/130694752/fc0ebe96-95fe-4571-987e-6c1a9f54f397)

2.) Normalize the data:

- Used bm25 from `implicit` library to weigh the matrix. This aids in reducing the impact of users that have played the same artists thousands of times, as well as reduces the weight of "popular items" in our model.

3.) LightFM

LightFM is a Python implementation for recommendation algorithms.

- We used lightFM to get a precision output for our model. Our first precision score came out to 14%, so we changed our parameters to try to get a better precision score, below are the parameters we changed:
  
  - number of epochs: number of iterations
  - learning_rate: controls how quickly the model adapts to the problem.
  - number of components:  the dimensionality of the feature latent embeddings.
  - item_alpha: L2 penalty on item features
  - user_alpa:  L2 penalty on user features

After trying different parameters we got the precision up to 25%


# Conclusion

Our model was sucessfully able to generate 10 artist recommendations, however we were unable to get our preferred precision outcome. This project was a lot of trial and error when it came to our model and there are many things we would like to implement as next steps to improve our recommender. 

- Use a different dataset:

   - There is a variety of music data that we could use for this model, it would be interesting to see if we could generate song recommendations instead of artists.

- Implement a different type of collaborative filtering

   - for this model we used a user/item type of collaborative filtering, however, there are multiple types of collaborative filtering including item/item and user/user filtering. In the future, we could try to use a different type of filtering to train our model.


# Resources 

A lot of resources were used to help us research machine learning models, and we cannot include all of them as hours were dedicated to research. However, here are a few links that could help create a model similar to what we did.

1.) [Build a spotify-like recommender system in python](https://www.youtube.com/watch?app=desktop&v=gaZKjAKfe0s)

2.) [Building and Testing Recommender Systems With Surprise](https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b)

4.) [How To Use The Surprise Library For Recommendation Engines](https://www.youtube.com/watch?v=fEd1p8-3S7w)





