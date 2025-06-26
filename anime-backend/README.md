# Anime Recommendation Backend

This Flask API serves personalized anime recommendations using a pre-trained deep learning model.

## 🧠 Model

- TensorFlow/Keras model trained on MyAnimeList user-anime ratings.
- Uses user and anime encodings to predict unseen anime ratings.

## 📁 Files Required

Place these in the project root:

- `model_df.pkl`
- `anime_df.pkl`
- `anime_recommender_model.keras`
- `anime_model_metadata.pkl`

## 🚀 Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
