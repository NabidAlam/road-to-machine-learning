# Recommender Systems Guide

Comprehensive guide to building recommendation systems, from collaborative filtering to deep learning approaches.

## Table of Contents

- [Introduction](#introduction)
- [Types of Recommender Systems](#types-of-recommender-systems)
- [Collaborative Filtering](#collaborative-filtering)
- [Content-Based Filtering](#content-based-filtering)
- [Hybrid Approaches](#hybrid-approaches)
- [Evaluation Metrics](#evaluation-metrics)
- [Implementation Examples](#implementation-examples)
- [Advanced Techniques](#advanced-techniques)
- [Resources](#resources)

---

## Introduction

**Recommender Systems** are information filtering systems that predict user preferences and suggest items (products, movies, articles, etc.) that users might like.

### Why Recommender Systems Matter

- **Personalization**: Tailor experiences to individual users
- **Discovery**: Help users find new items
- **Business Value**: Increase engagement, sales, retention
- **Scale**: Handle millions of users and items

### Real-World Applications

- **E-commerce**: Product recommendations (Amazon, eBay)
- **Streaming**: Content recommendations (Netflix, Spotify)
- **Social Media**: Content feed (Facebook, Twitter)
- **News**: Article recommendations
- **Dating**: Match suggestions

---

## Types of Recommender Systems

### 1. Collaborative Filtering
- Uses user-item interactions
- "Users who liked X also liked Y"
- No need for item features

### 2. Content-Based Filtering
- Uses item features
- "Items similar to what you liked"
- Requires item metadata

### 3. Hybrid Approaches
- Combines multiple methods
- Often performs best

---

## Collaborative Filtering

### User-Based Collaborative Filtering

Find users similar to target user, recommend items they liked.

**Steps:**
1. Find similar users (using cosine similarity, Pearson correlation)
2. Get items liked by similar users
3. Recommend items target user hasn't seen

**Example:**
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# User-item matrix (rows: users, columns: items)
# Values: ratings (1-5) or binary (liked/not liked)
ratings = np.array([
    [5, 4, 0, 0, 1],  # User 1
    [4, 5, 0, 0, 0],  # User 2
    [0, 0, 5, 4, 0],  # User 3
    [0, 0, 4, 5, 0],  # User 4
    [1, 0, 0, 0, 5]   # User 5
])

def user_based_recommend(user_id, ratings, n_recommendations=3):
    """
    Recommend items to a user based on similar users
    """
    # Calculate user similarity
    user_similarity = cosine_similarity(ratings)
    
    # Get similar users (excluding self)
    similar_users = np.argsort(user_similarity[user_id])[::-1][1:]
    
    # Calculate weighted ratings from similar users
    user_ratings = ratings[user_id]
    recommendations = np.zeros(ratings.shape[1])
    
    for similar_user in similar_users:
        similarity = user_similarity[user_id, similar_user]
        similar_user_ratings = ratings[similar_user]
        
        # Only consider items target user hasn't rated
        mask = (user_ratings == 0) & (similar_user_ratings > 0)
        recommendations[mask] += similarity * similar_user_ratings[mask]
    
    # Get top recommendations
    recommended_items = np.argsort(recommendations)[::-1][:n_recommendations]
    return recommended_items

# Example: Recommend to user 0
recommendations = user_based_recommend(0, ratings)
print(f"Recommended items for user 0: {recommendations}")
```

### Item-Based Collaborative Filtering

Find items similar to items user liked, recommend similar items.

**Steps:**
1. Calculate item-item similarity
2. For items user liked, find similar items
3. Recommend items user hasn't seen

**Example:**
```python
def item_based_recommend(user_id, ratings, n_recommendations=3):
    """
    Recommend items based on item similarity
    """
    # Calculate item similarity (transpose matrix)
    item_similarity = cosine_similarity(ratings.T)
    
    # Get user's ratings
    user_ratings = ratings[user_id]
    
    # Find items user has rated
    rated_items = np.where(user_ratings > 0)[0]
    
    # Calculate recommendation scores
    recommendations = np.zeros(ratings.shape[1])
    
    for rated_item in rated_items:
        rating = user_ratings[rated_item]
        similar_items = item_similarity[rated_item]
        
        # Weight by similarity and user's rating
        recommendations += similar_items * rating
    
    # Set already rated items to 0
    recommendations[rated_items] = 0
    
    # Get top recommendations
    recommended_items = np.argsort(recommendations)[::-1][:n_recommendations]
    return recommended_items

# Example
recommendations = item_based_recommend(0, ratings)
print(f"Recommended items for user 0: {recommendations}")
```

### Matrix Factorization

Decompose user-item matrix into lower-dimensional matrices.

**Singular Value Decomposition (SVD)**:
```python
from scipy.sparse.linalg import svds

def matrix_factorization_recommend(ratings, n_factors=2, n_recommendations=3):
    """
    Use SVD for matrix factorization
    """
    # Perform SVD
    U, sigma, Vt = svds(ratings, k=n_factors)
    
    # Reconstruct matrix
    sigma_matrix = np.diag(sigma)
    predicted_ratings = U @ sigma_matrix @ Vt
    
    return predicted_ratings

# Example
predicted = matrix_factorization_recommend(ratings)
print("Predicted ratings matrix:")
print(predicted)
```

---

## Content-Based Filtering

Uses item features to recommend similar items.

**Steps:**
1. Extract item features
2. Create item profile
3. Create user profile (from liked items)
4. Recommend items similar to user profile

**Example:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example: Movie recommendations based on genres
movies = [
    "Action Adventure Sci-Fi",
    "Action Adventure",
    "Comedy Romance",
    "Drama Romance",
    "Horror Thriller",
    "Action Sci-Fi"
]

# User liked movies (indices)
user_liked = [0, 1, 5]  # Action/Sci-Fi movies

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
movie_vectors = vectorizer.fit_transform(movies)

# Create user profile (average of liked movies)
user_profile = movie_vectors[user_liked].mean(axis=0)

# Calculate similarity to all movies
similarities = cosine_similarity(user_profile, movie_vectors)[0]

# Get recommendations (excluding already liked)
recommendations = np.argsort(similarities)[::-1]
recommendations = [r for r in recommendations if r not in user_liked]

print(f"Recommended movies: {recommendations[:3]}")
```

---

## Hybrid Approaches

Combine multiple methods for better performance.

### Weighted Hybrid
```python
def hybrid_recommend(user_id, ratings, content_similarity, 
                    alpha=0.5, n_recommendations=3):
    """
    Combine collaborative and content-based filtering
    """
    # Collaborative filtering score
    collab_score = item_based_recommend(user_id, ratings, 
                                        n_recommendations=ratings.shape[1])
    
    # Content-based score
    content_score = content_similarity[user_id]
    
    # Weighted combination
    hybrid_score = alpha * collab_score + (1 - alpha) * content_score
    
    # Get top recommendations
    recommended_items = np.argsort(hybrid_score)[::-1][:n_recommendations]
    return recommended_items
```

---

## Evaluation Metrics

### 1. Precision@K
Proportion of recommended items that are relevant.

```python
def precision_at_k(recommended, relevant, k):
    """
    Precision at K
    """
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    recommended_set = set(recommended_k)
    
    if len(recommended_set) == 0:
        return 0
    
    return len(recommended_set & relevant_set) / len(recommended_set)
```

### 2. Recall@K
Proportion of relevant items that were recommended.

```python
def recall_at_k(recommended, relevant, k):
    """
    Recall at K
    """
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    recommended_set = set(recommended_k)
    
    if len(relevant_set) == 0:
        return 0
    
    return len(recommended_set & relevant_set) / len(relevant_set)
```

### 3. Mean Average Precision (MAP)
Average precision across all users.

### 4. Root Mean Squared Error (RMSE)
For rating prediction tasks.

---

## Implementation Examples

### Using Surprise Library

```bash
pip install surprise
```

```python
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# Load data (user, item, rating)
data = Dataset.load_builtin('ml-100k')

# Split data
trainset, testset = train_test_split(data, test_size=0.2)

# Train model (SVD)
algo = SVD()
algo.fit(trainset)

# Make predictions
predictions = algo.test(testset)

# Evaluate
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print(f"RMSE: {rmse}, MAE: {mae}")
```

### Using TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow import keras

def build_recommender_model(n_users, n_items, embedding_dim=50):
    """
    Neural collaborative filtering model
    """
    # User embedding
    user_input = keras.layers.Input(shape=(), name='user_id')
    user_embedding = keras.layers.Embedding(n_users, embedding_dim)(user_input)
    user_vec = keras.layers.Flatten()(user_embedding)
    
    # Item embedding
    item_input = keras.layers.Input(shape=(), name='item_id')
    item_embedding = keras.layers.Embedding(n_items, embedding_dim)(item_input)
    item_vec = keras.layers.Flatten()(item_embedding)
    
    # Concatenate
    concat = keras.layers.Concatenate()([user_vec, item_vec])
    
    # Dense layers
    dense1 = keras.layers.Dense(128, activation='relu')(concat)
    dropout1 = keras.layers.Dropout(0.5)(dense1)
    dense2 = keras.layers.Dense(64, activation='relu')(dropout1)
    dropout2 = keras.layers.Dropout(0.5)(dense2)
    output = keras.layers.Dense(1, activation='sigmoid')(dropout2)
    
    model = keras.Model([user_input, item_input], output)
    model.compile(optimizer='adam', loss='binary_crossentropy', 
                 metrics=['accuracy'])
    
    return model

# Example usage
model = build_recommender_model(n_users=1000, n_items=500)
model.summary()
```

---

## Advanced Techniques

### 1. Deep Learning for Recommendations

- Neural Collaborative Filtering
- Wide & Deep Learning
- DeepFM

### 2. Sequential Recommendations

- RNN/LSTM for session-based recommendations
- Transformer models

### 3. Context-Aware Recommendations

- Include context (time, location, device)
- Tensor factorization

### 4. Cold Start Problem

**New User:**
- Use demographic data
- Popular items
- Content-based recommendations

**New Item:**
- Use item features
- Content-based similarity

---

## Resources

### Libraries

1. **Surprise**: [Documentation](https://surprise.readthedocs.io/)
   - Scikit-learn for recommender systems

2. **Implicit**: [Documentation](https://implicit.readthedocs.io/)
   - Fast collaborative filtering

3. **LightFM**: [Documentation](https://making.lyst.com/lightfm/docs/home.html)
   - Hybrid recommender systems

### Datasets

1. **MovieLens**: [Website](https://grouplens.org/datasets/movielens/)
   - Movie ratings dataset

2. **Amazon Product Data**: [Kaggle](https://www.kaggle.com/datasets)
   - Product reviews and ratings

### Books

1. **"Recommender Systems Handbook"** by Ricci et al.
   - Comprehensive reference

2. **"Building Recommender Systems with Machine Learning and AI"** by Frank Kane
   - Practical guide

### Papers

1. **"Matrix Factorization Techniques for Recommender Systems"**
   - Koren et al., 2009

2. **"Neural Collaborative Filtering"**
   - He et al., 2017

---

## Key Takeaways

1. **Start Simple**: Begin with collaborative filtering
2. **Hybrid is Best**: Combine multiple approaches
3. **Evaluate Properly**: Use appropriate metrics
4. **Handle Cold Start**: Plan for new users/items
5. **Scale Matters**: Consider scalability for production

---

**Note**: Recommender systems are critical for many applications. This guide covers basics. For production systems, consider advanced techniques, A/B testing, and real-time recommendations.

