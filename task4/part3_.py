import numpy as np

# Sample review score matrix (6 users, 5 movies)
review_scores = np.array([
    [8, 5, np.nan, 1, 7],  # User 1 has not reviewed movie 3
    [4, 5, np.nan, 9, 5],  # User 2 has not reviewed movie 3
    [6, 2, 4, 2, 8],
    [7, 3, 6, 7, 3],
    [np.nan, 4, 8, 8, 6],  # User 5 has not reviewed movie 1
    [9, 9, 10, 6, 8]
])

# Initialize a matrix to store Euclidean distances
num_users, num_movies = review_scores.shape
euclidean_distances = np.zeros((num_users, num_users))

# Calculate Euclidean distances between pairs of users
for i in range(num_users):
    for j in range(num_users):
        mask = ~np.isnan(review_scores[i]) & ~np.isnan(review_scores[j])  # Mask for common ratings
        if np.any(mask):  # If there are common ratings
            common_ratings_i = review_scores[i][mask]
            common_ratings_j = review_scores[j][mask]
            squared_diff = np.sum((common_ratings_i - common_ratings_j) ** 2)  # Calculate squared differences
            euclidean_distances[i, j] = squared_diff  # Store the squared difference
            euclidean_distances[j, i] = squared_diff  # Distance matrix is symmetric

# Calculate similarity using the formula: 1 / (1 + sqrt(sum of squared differences))
eucl_similarities = 1 / (1 + np.sqrt(euclidean_distances))

print(eucl_similarities)

#Calculate Pearson correlation
pearson_correlations = np.zeros((num_users, num_users))

# Calculate Pearson correlation between pairs of users
for i in range(num_users):
    for j in range(num_users):
       mask = ~np.isnan(review_scores[i]) & ~np.isnan(review_scores[j])  # Mask for common ratings
       if np.any(mask):  # If there are common ratings
        common_ratings_i = review_scores[i][mask]
        common_ratings_j = review_scores[j][mask]
        #First find the sum of the ratings for each user
        sum_i = np.sum(common_ratings_i)
        sum_j = np.sum(common_ratings_j)
        #Then find the sum of the squared ratings for each user
        sum_squared_i = np.sum(common_ratings_i**2)
        sum_squared_j = np.sum(common_ratings_j**2)   

        #Then find the sum of the product of the ratings for each user
        pSum = np.dot(common_ratings_i, common_ratings_j)
        #Then find the number of movies that both users have rated
        num_movies = np.sum(mask)
        # numerator
        numerator = pSum - (sum_i*sum_j/num_movies)
        # denominator
        denominator = np.sqrt((sum_squared_i - sum_i**2/num_movies)*(sum_squared_j - sum_j**2/num_movies))
        # Pearson correlation
        pearson_correlations[i, j] = numerator / denominator
        pearson_correlations[j, i] = numerator / denominator
# find the similarity
pearson_similarities = (1 + pearson_correlations)/2

print(pearson_similarities)




# For User 2, find the nearest neighbors using Euclidean distance and Pearson correlation
k = 2

# For User 2, find the nearest neighbors using Euclidean distance
user_2_euclidean_similarities = eucl_similarities[1]  # Similarities of User 2 with all other users
user_2_euclidean_similarities[1] = 0  # Exclude similarity with itself
nearest_neighbors_euclidean = np.argsort(user_2_euclidean_similarities)[::-1][:k]
print("Nearest neighbors for User 2 using Euclidean distance:", nearest_neighbors_euclidean+1)

# For User 2, find the nearest neighbors using Pearson correlation
user_2_pearson_similarities = pearson_similarities[1]  # Similarities of User 2 with all other users
user_2_pearson_similarities[1] = 0  # Exclude similarity with itself
nearest_neighbors_pearson = np.argsort(user_2_pearson_similarities)[::-1][:k]
print("Nearest neighbors for User 2 using Pearson correlation:", nearest_neighbors_pearson+1)

movie_index = 2
# For User 2, predict the review score for Movie 3(column index = 2) using Euclidean distance

# Filter the nearest neighbors who have rated Movie 3
nearest_neighbors_rated_movie_3 = nearest_neighbors_euclidean[~np.isnan(review_scores[nearest_neighbors_euclidean, 2])]

# Calculate weighted average of scores for Movie 3
weighted_sum = 0
total_similarity = 0
for neighbor_index in nearest_neighbors_rated_movie_3:
    similarity = eucl_similarities[1][neighbor_index]
    weighted_sum += similarity * review_scores[neighbor_index, 2]
    total_similarity += similarity

# Predict the review score for Movie 3 for User 2
predicted_score = weighted_sum / total_similarity
print("Predicted review score for Movie 3 for User 2 using Euclidean distance:", predicted_score)

# For User 2, predict the review score for Movie 3(column index = 2) using Pearson correlation
nearest_neighbors_rated_movie_3 = nearest_neighbors_pearson[~np.isnan(review_scores[nearest_neighbors_pearson, 2])]

# Calculate weighted average of scores for Movie 3
weighted_sum = 0
total_similarity = 0
for neighbor_index in nearest_neighbors_rated_movie_3:
    similarity = pearson_similarities[1][neighbor_index]
    weighted_sum += similarity * review_scores[neighbor_index, 2]
    total_similarity += similarity

# Predict the review score for Movie 3 for User 2
predicted_score = weighted_sum / total_similarity
print("Predicted review score for Movie 3 for User 2 using Pearson correlation:", predicted_score)


# Find the biggest similarity values for all users
for i in range(num_users):
    print("User", i+1, ":", np.argsort(eucl_similarities[i])[::-1]+1)
    print("User", i+1, ":", np.argsort(pearson_similarities[i])[::-1]+1)
    #the biggest similarity values for all users
    print("User", i+1, ":", np.sort(eucl_similarities[i])[::-1])
    print("User", i+1, ":", np.sort(pearson_similarities[i])[::-1])