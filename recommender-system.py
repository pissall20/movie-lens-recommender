import findspark
findspark.init("/home/hduser/Apps/spark")
import os
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.mllib.recommendation import ALS
import math

# Initialize spark
spark_conf = SparkConf().setAppName('Recommend').setMaster("local[2]").set("spark.executor.memory", "6g").set("spark.driver.memory", "6g")
sc = SparkContext(conf=spark_conf)
sqlContext = SQLContext(sc)
datasets_path = "/home/hduser/PycharmProjects/recommender/datasets"

# Importing the small ratings file
small_ratings_file = os.path.join(datasets_path, 'ml-latest-small', 'ratings.csv')
small_ratings_raw_data = sc.textFile(small_ratings_file)
# Getting the header
small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]

print small_ratings_raw_data.take(20)
print small_ratings_raw_data_header

# Getting the data (UserID, MovieID, Rating)
small_ratings_data = small_ratings_raw_data.filter(lambda line: line!=small_ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()

print small_ratings_data.take(3)

# Importing the movies file
small_movies_file = os.path.join(datasets_path, 'ml-latest-small', 'movies.csv')
small_movies_raw_data = sc.textFile(small_movies_file)
# Getting the header
small_movies_raw_data_header = small_movies_raw_data.take(1)[0]

# Getting the data (ID, Movie)
small_movies_data = small_movies_raw_data.filter(lambda line: line != small_movies_raw_data_header) \
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0], tokens[1])).cache()

print small_movies_data.take(3)

# Divide the data into train, validation and test
training_RDD, validation_RDD, test_RDD = small_ratings_data.randomSplit([6, 2, 2], seed=0L)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

# Training parameters

seed = 5L
iterations = 10
regularization_parameter = 0.1
# Ranks : Number of features to use
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02

# Validation parameters
min_error = float('inf')
best_rank = -1
best_iteration = -1

# Cross validation loop
for rank in ranks:
    # Create the model object and train it
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations, lambda_=regularization_parameter)
    # Predict on the validation data
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    # Join the predictions to the validation RDD
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    # Finding RMSE
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    print 'For rank %s the RMSE is %s' % (rank, error)
    if error < min_error:
        min_error = error
        best_rank = rank
print 'The best model was trained with rank %s' % best_rank

# Train model with best_rank
model = ALS.train(training_RDD, best_rank, seed=seed, iterations=iterations,lambda_=regularization_parameter)
predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())

print 'For testing (small) the RMSE is %s' % (error)


# Load the complete dataset file
complete_ratings_file = os.path.join(datasets_path, 'ml-latest', 'ratings.csv')
complete_ratings_raw_data = sc.textFile(complete_ratings_file)
complete_ratings_raw_data_header = complete_ratings_raw_data.take(1)[0]

# Getting the data like (UserID, MovieID, Rating)
complete_ratings_data = complete_ratings_raw_data.filter(lambda line: line != complete_ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]), int(tokens[1]), float(tokens[2]))).cache()

print "There are %s recommendations in the complete dataset" % (complete_ratings_data.count())

# Split into training and testing
training_RDD, test_RDD = complete_ratings_data.randomSplit([7, 3], seed=0L)
# Train the model on entire data with the best_rank
complete_model = ALS.train(training_RDD, best_rank, seed=seed, iterations=iterations, lambda_=regularization_parameter)

test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

# Make predictions
predictions = complete_model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)

# Find RMSE for test data
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())

print 'For testing data(complete) the RMSE is %s' % error

# That was content similarity. We will dive into user and content based collaborative filtering

# Load the complete movies file
complete_movies_file = os.path.join(datasets_path, 'ml-latest', 'movies.csv')
complete_movies_raw_data = sc.textFile(complete_movies_file)
complete_movies_raw_data_header = complete_movies_raw_data.take(1)[0]

# Parse (MovieID, Title, Genre)
complete_movies_data = complete_movies_raw_data.filter(lambda line: line!=complete_movies_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()

complete_movies_titles = complete_movies_data.map(lambda x: (int(x[0]), x[1]))

print "There are %s movies in the complete dataset" % (complete_movies_titles.count())

# This function takes in (MovieID, (Ratings)) and returns (MovieID, (Number of ratings, average ratings))
def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)

# Get the movie ID with all their ratings
movie_ID_with_ratings_RDD = (complete_ratings_data.map(lambda x: (x[1], x[2])).groupByKey())
# Use the helper function to get the number of ratings and average ratings
movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
# Separate RDD for (MovieID and number of ratings)
movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))


# We create a new user that is not their in the database to test the recommender
new_user_ID = 0

# The format of each line is (userID, movieID, rating)
new_user_ratings = [
     (0,260,4), # Star Wars (1977)
     (0,1,3), # Toy Story (1995)
     (0,16,3), # Casino (1995)
     (0,25,4), # Leaving Las Vegas (1995)
     (0,32,4), # Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
     (0,335,1), # Flintstones, The (1994)
     (0,379,1), # Timecop (1994)
     (0,296,3), # Pulp Fiction (1994)
     (0,858,5) , # Godfather, The (1972)
     (0,50,4) # Usual Suspects, The (1995)
    ]
new_user_ratings_RDD = sc.parallelize(new_user_ratings)
print 'New user ratings: %s' % new_user_ratings_RDD.take(10)

# Add this user to our complete ratings data
complete_data_with_new_ratings_RDD = complete_ratings_data.union(new_user_ratings_RDD)

# To check the time taken while training the model
from time import time

t0 = time()
# Train the new model with the new user
new_ratings_model = ALS.train(complete_data_with_new_ratings_RDD, best_rank, seed=seed,
                              iterations=iterations, lambda_=regularization_parameter)
tt = time() - t0

print "New model trained in %s seconds" % round(tt,3)

# To predict we will start with taking an RDD of movies the new user has not rated yet

# Get just movie IDs
new_user_ratings_ids = map(lambda x: x[1], new_user_ratings)
# Keep just those not on the ID list (thanks Lei Li for spotting the error!)
new_user_unrated_movies_RDD = (complete_movies_data.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (new_user_ID, x[0])))

# Use the input RDD, new_user_unrated_movies_RDD, with new_ratings_model.predictAll() to predict new ratings for the movies
new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)
print new_user_recommendations_RDD.take(10)

# Transform new_user_recommendations_RDD into pairs of the form (Movie ID, Predicted Rating)
new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
print new_user_recommendations_rating_RDD.take(10)

# Join (MovieID, Predicted Rating) to Movie Titles and Movie Rating Counts
new_user_recommendations_rating_title_and_count_RDD = \
    new_user_recommendations_rating_RDD.join(complete_movies_titles).join(movie_rating_counts_RDD)
print new_user_recommendations_rating_title_and_count_RDD.take(10)

# Transform the joint RDD to (Title, Predicted Rating, Ratings Count)
new_user_recommendations_rating_title_and_count_RDD = \
    new_user_recommendations_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))

# Get movies with highest rating, filtering out movies which have been rated less than 25 times
top_movies = new_user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2]>=25).takeOrdered(25, key=lambda x: -x[1])

print ('TOP recommended movies (with more than 25 reviews):\n%s' %
        '\n'.join(map(str, top_movies)))

# How to predict for a particular movie for a user using the same trained model
my_movie = sc.parallelize([(0, 500)]) # Quiz Show (1994)
individual_movie_rating_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)
individual_movie_rating_RDD.take(1)

model_path = os.path.join('..', 'models', 'movie_lens_als')

# Save model
model.save(sc, model_path)

