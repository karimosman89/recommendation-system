import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_user_item_data(file_path):
    """Load the user-item interaction data."""
    return pd.read_csv(file_path)

def get_user_recommendations(user_id, data, num_recommendations=5):
    """Provide item recommendations for the given user."""
    user_data = data[data['user_id'] == user_id].drop(columns=['user_id'])
    user_profile = user_data.mean(axis=0)

    similarity_matrix = cosine_similarity(user_profile.values.reshape(1, -1), data.drop(columns=['user_id']).values)
    recommendations = similarity_matrix.argsort()[0][-num_recommendations:]

    return data.iloc[recommendations]

