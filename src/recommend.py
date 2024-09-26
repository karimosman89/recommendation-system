import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors

def load_data(filepath):
    return pd.read_csv(filepath)

def recommend(user_id, data, n_recommendations=5):
    user_data = data[data['user_id'] == user_id]
    if user_data.empty:
        return []
    model = NearestNeighbors(n_neighbors=n_recommendations)
    model.fit(data[['item_id', 'rating']])
    distances, indices = model.kneighbors(user_data[['item_id', 'rating']])
    recommendations = data.iloc[indices.flatten()]['item_id'].values
    return recommendations

if __name__ == "__main__":
    data = load_data('data/user_data.csv')
    user_id = 1  # Example user ID
    recommendations = recommend(user_id, data)
    print("Recommendations for user:", recommendations)
