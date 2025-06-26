# anime-backend/app.py
import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import gdown



# Helper to download model_df.pkl if not found
def download_model_df_from_drive():
    file_id = "1r6_PVcdJBqzEdokcqiu6jN93k5h-42y9"  # Replace with your file ID
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = os.path.join(file_path, 'model_df.pkl')
    if not os.path.exists(output_path):
        print("📦 Downloading model_df.pkl from Google Drive...")
        gdown.download(url, output_path, quiet=False)
        print("✅ Downloaded model_df.pkl.")

#https://drive.google.com/file/d/1kAdrIDujYtM5x__sWUrxQqC_0pwEi_G8/view?usp=sharing
def download_anime_recommender_model_from_drive():
    file_id = "1kAdrIDujYtM5x__sWUrxQqC_0pwEi_G8"  # Replace with your file ID
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = os.path.join(file_path, 'anime_recommender_model.keras')
    if not os.path.exists(output_path):
        print("📦 Downloading anime_recommender_model.keras from Google Drive...")
        gdown.download(url, output_path, quiet=False)
        print("✅ Downloaded anime_recommender_model.keras.")

#https://drive.google.com/file/d/1yAnbvxagJr-kZ_y2RqQzPlPgiUPbiykP/view?usp=sharing
def download_metadata_from_drive():
    file_id = "1yAnbvxagJr-kZ_y2RqQzPlPgiUPbiykP"  # Replace with your actual ID
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = os.path.join(file_path, 'anime_model_metadata.pkl')
    if not os.path.exists(output_path):
        print("📦 Downloading anime_model_metadata.pkl from Google Drive...")
        gdown.download(url, output_path, quiet=False)
        print("✅ Downloaded anime_model_metadata.pkl.")

# Helper to download model_df.pkl if not found   https://drive.google.com/file/d/1FN1ZQmledk_9npLFxaNg27WRDv0gsqQT/view?usp=sharing
def download_anime_df_from_drive():
    file_id = "1FN1ZQmledk_9npLFxaNg27WRDv0gsqQT"  # Replace with your file ID
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = os.path.join(file_path, 'anime_df.pkl')
    if not os.path.exists(output_path):
        print("📦 Downloading anime_df.pkl from Google Drive...")
        gdown.download(url, output_path, quiet=False)
        print("✅ Downloaded anime_df.pkl.")




# Load files from current directory
file_path = './'

# Globals
model_df = None
user_encoder = None
anime_encoder = None
encoded_to_user_id = None
encoded_to_anime_id = None
model = None
anime_df = None
data_load_error = None

# Load Data & Model
try:
    print("\U0001F680 Loading data and model...")
    download_model_df_from_drive() 

    model_df = pd.read_pickle(os.path.join(file_path, 'model_df.pkl'))
    # user_encoder = LabelEncoder()
    # anime_encoder = LabelEncoder()
    # model_df['user'] = user_encoder.fit_transform(model_df['user_id'])
    # model_df['anime'] = anime_encoder.fit_transform(model_df['anime_id'])

    # encoded_to_user_id = dict(zip(model_df['user'], model_df['user_id']))
    # encoded_to_anime_id = dict(zip(model_df['anime'], model_df['anime_id']))

    download_anime_df_from_drive() 

    anime_df = pd.read_pickle(os.path.join(file_path, 'anime_df.pkl'))
    
    # Load your DataFrame
    model_df = pd.read_pickle("model_df.pkl")
    anime_df = pd.read_pickle("anime_df.pkl")

    # Encode
    user_encoder = LabelEncoder()
    anime_encoder = LabelEncoder()

    model_df['user'] = user_encoder.fit_transform(model_df['user_id'])
    model_df['anime'] = anime_encoder.fit_transform(model_df['anime_id'])

    print("[✅] Encoding complete.")
    print(f"Unique users: {model_df['user'].nunique()}, Unique anime: {model_df['anime'].nunique()}")
    print(model_df[['user_id', 'anime_id', 'user', 'anime', 'rating']].head())

    # Create reverse mappings
    encoded_to_user_id = dict(zip(model_df['user'], model_df['user_id']))
    encoded_to_anime_id = dict(zip(model_df['anime'], model_df['anime_id']))

    # Save metadata cleanly
    with open("anime_model_metadata.pkl", "wb") as f:
        pickle.dump({
            'user_encoder': user_encoder,
            'anime_encoder': anime_encoder,
            'encoded_to_user_id': encoded_to_user_id,
            'encoded_to_anime_id': encoded_to_anime_id
        }, f)

    print("[✅] Metadata saved without DummyEncoder.")

    download_anime_recommender_model_from_drive()

    model = load_model(os.path.join(file_path, "anime_recommender_model.keras"))

    download_metadata_from_drive()
    with open(os.path.join(file_path, "anime_model_metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
        user_encoder = metadata['user_encoder']
        anime_encoder = metadata['anime_encoder']
        encoded_to_user_id = metadata['encoded_to_user_id']
        encoded_to_anime_id = metadata['encoded_to_anime_id']

    anime_df = pd.read_pickle(os.path.join(file_path, 'anime_df.pkl'))

    print("\u2705 Data and model loaded.")
except Exception as e:
    data_load_error = f"\u274C Error loading data or model: {e}"
    print(data_load_error)

# Flask app
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Anime Recommendation API is live. Use /recommend/<user_id>."

@app.route('/recommend/<int:target_user_id>', methods=['GET'])
def recommend_anime(target_user_id):
    global model_df, user_encoder, anime_encoder, encoded_to_user_id, encoded_to_anime_id, model, anime_df

    try:
        if target_user_id not in user_encoder.classes_:
            return jsonify({"error": f"User ID {target_user_id} not found."}), 404

        encoded_user = user_encoder.transform([target_user_id])[0]
        n_animes = len(anime_encoder.classes_)
        all_anime_ids = np.arange(n_animes)

        rated_animes = model_df[model_df['user'] == encoded_user]['anime'].values
        animes_to_predict = np.setdiff1d(all_anime_ids, rated_animes)

        if len(animes_to_predict) == 0:
            return jsonify({"message": "No new recommendations available."}), 200

        user_input = np.full(len(animes_to_predict), encoded_user)
        predicted_ratings = model.predict([user_input, animes_to_predict], batch_size=2048, verbose=0)

        top_N = min(10, len(animes_to_predict))
        top_indices = predicted_ratings.flatten().argsort()[-top_N:][::-1]
        top_animes = animes_to_predict[top_indices]
        top_scores = predicted_ratings.flatten()[top_indices]

        recommendations = []
        for enc_id, score in zip(top_animes, top_scores):
            orig_id = encoded_to_anime_id.get(enc_id)
            if orig_id:
                name_row = anime_df[anime_df['MAL_ID'] == orig_id]
                if not name_row.empty:
                    anime_name = name_row['Name'].values[0]
                    recommendations.append({
                        "anime_name": anime_name,
                        "predicted_rating": round(float(score), 2)
                    })

        return jsonify({
            "user_id": target_user_id,
            "recommendations": recommendations
        }), 200, {'Content-Type': 'application/json'}

    except Exception as e:
        print(f"Recommendation error: {e}")
        return jsonify({"error": f"Internal error: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)