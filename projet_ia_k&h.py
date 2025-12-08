# Détecter les langues
#!pip install langdetect

# Transformers (Hugging Face)
#!pip install transformers

# Flask pour créer une API
#!pip install Flask

# pyngrok pour exposer Flask sur le web
#!pip install pyngrok

# Torch (PyTorch) nécessaire pour Transformers
#!pip install torch

#!pip install Flask pyngrok torch transformers langdetect sentencepiece flask-cors

#!pip install deepface

import numpy as np

from langdetect import detect, DetectorFactory
from transformers import pipeline
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import warnings
import base64
from io import BytesIO
from PIL import Image
from deepface import DeepFace
import numpy as np
import os
import socket
import requests

warnings.filterwarnings("ignore")
DetectorFactory.seed = 0

# ===================================================================
# CONFIGURATION GLOBALE
# ===================================================================
FLASK_PORT = 8000
# IMPORTANT: Remplacez par votre vrai token ngrok si vous en avez un.
# Le token actuel est un exemple tronqué.
NGROK_AUTH_TOKEN = "36FQDs0nofoEyBBixNM1aae1zNb_2xGnqe4zXWYT5haWdjy34"

# APIs Externes
SPOTIFY_ID = "269fe39af4bd43bd90bd163438e10188"
SPOTIFY_SECRET = "f04086525d914561b3a35e660ab7bcb8"
TMDB_KEY = "32d4a321cbba2d509be06d4223ad6467"

# ===================================================================
# CLASSE DE RECOMMANDATIONS
# ===================================================================
class EmotionRecommender:
    def __init__(self, spotify_id, spotify_secret, tmdb_key):
        self.spotify_id = spotify_id
        self.spotify_secret = spotify_secret
        self.tmdb_key = tmdb_key
        self.spotify_token = None

        self.emotion_config = {
            'happy': {
                'spotify_seeds': ['happy', 'party', 'pop', 'dance'],
                'tmdb_genres': [35, 10751, 16], # Comédie, Familial, Animation
                'activity': "Organisez un appel vidéo avec un ami proche pour partager votre bonne humeur; Sortez faire une promenade énergique dans un parc; Planifiez un événement futur.",
                'meal': "Un Wok coloré aux légumes croquants et poulet grillé avec sauce soja-miel; Un smoothie tropical vitaminé et des fruits frais; Une salade de pâtes estivale et légère."
            },
            'sad': {
                'spotify_seeds': ['chill', 'relax', 'calm', 'piano'],
                'tmdb_genres': [35, 10751], # Comédie, Familial (pour soulager)
                'activity': "Prenez un bain chaud avec de la musique douce; Regardez un film comique pour vous changer les idées; Écrivez dans un journal intime pour clarifier vos émotions.",
                'meal': "Un bol réconfortant de soupe miso avec tofu soyeux; Un chocolat chaud onctueux avec des marshmallows; Une part de gâteau au chocolat pour le réconfort."
            },
            'anxious': {
                'spotify_seeds': ['meditation', 'ambient', 'sleep', 'nature'],
                'tmdb_genres': [99, 10402], # Documentaire, Musique (calme)
                'activity': "Pratiquez 15 minutes de respiration profonde (4-7-8); Faites du yoga doux ou des étirements; Écoutez une méditation guidée pour calmer votre esprit.",
                'meal': "Une salade tiède aux légumes grillés avec avocat, saumon, noix et vinaigrette citronnée; Une tisane à la camomille ou à la verveine; Un bol de flocons d'avoine chaud avec du miel."
            },
            'angry': {
                'spotify_seeds': ['workout', 'rock', 'intense', 'pop'],
                'tmdb_genres': [28, 16, 35], # Action (libération), Animation, Comédie (distraction)
                'activity': "Faites un entraînement intense (course, boxe, HIIT) pour libérer l'énergie négative; Nettoyez et réorganisez une petite zone de votre maison; Criez dans un coussin pour exprimer votre frustration.",
                'meal': "Un burger épicé fait maison avec jalapeños; Des tacos au poulet mariné pimenté; Un jus de fruits très acidulé pour un reset sensoriel."
            },
            'neutral': {
                'spotify_seeds': ['pop', 'indie', 'alternative', 'folk'],
                'tmdb_genres': [35, 18, 10749], # Comédie, Drame, Romance
                'activity': "Explorez une nouvelle activité créative (dessin, écriture, cuisine); Faites une balade tranquille pour clarifier vos pensées; Lisez le premier chapitre d'un nouveau livre.",
                'meal': "Un bowl équilibré avec quinoa, légumes variés, poulet grillé et sauce tahini; Une pizza faite maison (polyvalente); Un plat de pâtes crémeuses."
            },
            'fear': {
                'spotify_seeds': ['calm', 'ambient', 'sleep', 'relax'],
                'tmdb_genres': [35, 10751], # Comédie, Familial (pour un sentiment de sécurité)
                'activity': "Rapprochez-vous de vos proches (appel ou câlin); Regardez un film réconfortant que vous avez déjà vu; Pratiquez la pleine conscience ou le grounding (ancrage).",
                'meal': "Un thé chaud apaisant (camomille, verveine); Des toasts à l'avocat et aux œufs, simple et nourrissant; Un plat de riz au lait sucré."
            },
            'surprise': {
                'spotify_seeds': ['upbeat', 'pop', 'indie', 'dance'],
                'tmdb_genres': [16, 12, 35], # Animation, Aventure, Comédie
                'activity': "Essayez une activité spontanée : visitez un lieu inattendu; Apprenez un tour de magie simple; Faites un jeu de société ou de cartes avec des amis.",
                'meal': "Un plat de cuisine du monde que vous n'avez jamais essayé (Ex: Pad Thai); Des crêpes salées ou sucrées; Un dessert très coloré et original."
            },
            'disgust': {
                'spotify_seeds': ['upbeat', 'positive', 'funk', 'pop'],
                'tmdb_genres': [35, 16], # Comédie pour se changer les idées, Animation
                'activity': "Nettoyez et réorganisez une petite zone de votre maison; Faites de l'exercice pour vous rafraîchir l'esprit; Lavez votre linge pour une sensation de propreté.",
                'meal': "Un jus de fruits frais détoxifiant; Un plat d'agrumes très acidulé pour un reset sensoriel; Une salade de roquette amère avec vinaigrette pour couper l'envie."
            },
            'joy': {
                'spotify_seeds': ['happy', 'dance', 'funk', 'upbeat'],
                'tmdb_genres': [10751, 35, 14], # Familial, Comédie, Fantastique
                'activity': "Planifiez une fête ou une sortie avec des amis; Documentez votre joie avec des photos ou une vidéo; Écoutez votre album préféré et dansez.",
                'meal': "Un plateau de sushis frais; Un dessert élaboré et festif (Fondue au chocolat); Un barbecue ou un repas en extérieur (si possible)."
            }
        }
        # ===================================================================

        self._get_spotify_token()

    def _get_spotify_token(self):
        try:
            auth_string = f"{self.spotify_id}:{self.spotify_secret}"
            auth_bytes = auth_string.encode("utf-8")
            auth_base64 = base64.b64encode(auth_bytes).decode("utf-8")

            url = "https://accounts.spotify.com/api/token"
            headers = {
                "Authorization": f"Basic {auth_base64}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            data = {"grant_type": "client_credentials"}

            response = requests.post(url, headers=headers, data=data, timeout=10)
            response.raise_for_status()

            self.spotify_token = response.json()["access_token"]
            print("Token Spotify obtenu")
            return True
        except Exception as e:
            print(f"ERREUR TOKEN SPOTIFY: {e}")
            self.spotify_token = None
            return False

    def recommend_music(self, emotion):
        emotion = emotion.lower()
        config = self.emotion_config.get(emotion, self.emotion_config['happy'])

        if not self.spotify_token:
            if not self._get_spotify_token():
                raise Exception("Token Spotify invalide")

        playlists = []
        headers = {"Authorization": f"Bearer {self.spotify_token}"}
        for seed in config['spotify_seeds']: 
            try:
                url = f"https://api.spotify.com/v1/search?q={seed}&type=playlist&limit=5"
                response = requests.get(url, headers=headers, timeout=15)

                if response.status_code == 401:
                    self._get_spotify_token()
                    headers = {"Authorization": f"Bearer {self.spotify_token}"}
                    response = requests.get(url, headers=headers, timeout=15)

                if response.status_code != 200:
                    continue

                data = response.json()

                if 'playlists' in data and 'items' in data['playlists']:
                    for item in data['playlists']['items'][:4]: 
                        if item and 'name' in item and 'external_urls' in item:
                            image_url = None
                            if 'images' in item and len(item['images']) > 0:
                                image_url = item['images'][0].get('url')

                            playlists.append({
                                'name': item['name'],
                                'url': item['external_urls'].get('spotify', '#'),
                                'image': image_url,
                                'description': item.get('description', '')[:120]
                            })
            except Exception as e:
                print(f"Erreur recherche playlist '{seed}': {e}")
                continue

        if not playlists:
            return [{'name': f"Playlist basée sur: {', '.join(config['spotify_seeds'])}", 'url': '#', 'image': None}]

        return playlists[:8] 

    def recommend_movies(self, emotion):
        emotion = emotion.lower()
        config = self.emotion_config.get(emotion, self.emotion_config['happy'])

        movies = []

        for genre_id in config['tmdb_genres']:
            try:
                url = "https://api.themoviedb.org/3/discover/movie"
                params = {
                    'api_key': self.tmdb_key,
                    'with_genres': genre_id,
                    'sort_by': 'popularity.desc',
                    'vote_count.gte': 500,
                    'language': 'fr-FR',
                    'page': 1
                }

                response = requests.get(url, params=params, timeout=15)

                if response.status_code != 200:
                    continue

                data = response.json()

                for movie in data.get('results', [])[:3]:
                    poster_path = movie.get('poster_path')
                    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None

                    movies.append({
                        'title': movie['title'],
                        'overview': movie.get('overview', 'Pas de synopsis disponible')[:150],
                        'rating': round(movie.get('vote_average', 0), 1),
                        'poster': poster_url,
                        'release_date': movie.get('release_date', '')[:4]
                    })
            except Exception as e:
                print(f"Erreur TMDB genre {genre_id}: {e}")
                continue

        seen = set()
        unique_movies = []
        for m in movies:
            if m['title'] not in seen:
                seen.add(m['title'])
                unique_movies.append(m)

        if not unique_movies:
            return [{'title': 'Le Fabuleux Destin d\'Amélie Poulain', 'overview': 'Un film léger pour un réconfort général.', 'rating': 8.3}]

        return unique_movies[:6] 
    def recommend_activities_multiple(self, emotion):
        emotion = emotion.lower()
        config = self.emotion_config.get(emotion, self.emotion_config['happy'])
        return [item.strip() for item in config['activity'].split(';') if item.strip()]

    def recommend_meals_multiple(self, emotion):
        emotion = emotion.lower()
        config = self.emotion_config.get(emotion, self.emotion_config['happy'])
        return [item.strip() for item in config['meal'].split(';') if item.strip()]

app = Flask(__name__)
CORS(app)

print(f"🔄 Chargement des modèles...")

device = 'cpu'
HF_DEVICE = -1 # Utiliser CPU par défaut

try:
    translator_fr = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en", device=HF_DEVICE)
    translator_ar = pipeline("translation", model="Helsinki-NLP/opus-mt-ar-en", device=HF_DEVICE)
    emotion_analyzer = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", device=HF_DEVICE)
    print("✅ Modèles textuels chargés")
except Exception as e:
    print(f"Erreur modèles HuggingFace: {e}")
    translator_fr, translator_ar, emotion_analyzer = None, None, None

try:
    recommender = EmotionRecommender(SPOTIFY_ID, SPOTIFY_SECRET, TMDB_KEY)
    print("✅ Système de recommandations initialisé")
except Exception as e:
    print(f"❌ Erreur système de recommandations: {e}")
    recommender = None


# ===================================================================
# FONCTION D'ANALYSE ÉMOTIONNELLE MULTILINGUE 
# ===================================================================
def detect_emotion_multilang(text):
    if not emotion_analyzer:
        return {'label': 'Error', 'score': 0.0}

    if not text.strip():
        return {'label': 'Neutral', 'score': 1.0}

    try:
        lang = detect(text)
    except:
        lang = 'en' 

    translated_text = text
    if lang == "fr" and translator_fr:
        try:
            translated_text = translator_fr(text)[0]['translation_text']
        except Exception as e:
            print(f"Erreur traduction FR: {e}")
            pass
    elif lang == "ar" and translator_ar:
        try:
            translated_text = translator_ar(text)[0]['translation_text']
        except Exception as e:
            print(f"Erreur traduction AR: {e}")
            pass

    result = emotion_analyzer(translated_text)[0]
    score = float(result['score'])
    return {'label': result['label'], 'score': score}

# ===================================================================
# MAPPING DES ÉMOTIONS 
# ===================================================================
EMOTION_MAPPING = {
    'joy': 'happy',
    'love': 'happy',
    'sadness': 'sad',
    'anger': 'angry',
    'fear': 'fear',
    'surprise': 'surprise',
    'disgust': 'disgust',
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'angry',
    'neutral': 'neutral',
    'disgust': 'disgust',
    'surprise': 'surprise'
}

# ===================================================================
# ROUTES API
# ===================================================================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'online',
        'message': 'API Analyse Émotionnelle + Recommandations',
        'endpoints': {
            'analyze_mood': '/analyze_mood [POST]',
            'recommendations': '/get_recommendations/<emotion> [GET]',
            'health': '/health [GET]'
        }
    })

@app.route('/analyze_mood', methods=['POST'])
def analyze_mood_api():
    """Analyse l'émotion à partir de texte ou d'image"""
    try:
        data = request.get_json()
        input_text = data.get('text', '').strip()
        input_image = data.get('image', None)

        response = None

        # Analyse d'image
        if input_image:
            image = None
            try:
                if "," in input_image:
                    encoded_data = input_image.split(",", 1)[1]
                else:
                    encoded_data = input_image

                image_data = base64.b64decode(encoded_data)
                image = Image.open(BytesIO(image_data)).convert('RGB')

                if image.size[0] == 0 or image.size[1] == 0:
                    return jsonify({'error': 'Image invalide'}), 400

            except Exception as e:
                return jsonify({'error': f'Erreur décodage image: {str(e)}'}), 400

            if image:
                try:
                    img_array = np.array(image)
                    analysis = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=True, detector_backend='opencv')

                    if isinstance(analysis, list):
                        analysis = analysis[0]

                    dominant_emotion = analysis.get('dominant_emotion', 'neutral').lower()

                    score_value = analysis.get('emotion', {}).get(dominant_emotion, 100)

                    mapped_emotion = EMOTION_MAPPING.get(dominant_emotion, 'neutral')

                    response = {
                        'label': mapped_emotion.capitalize(),
                        'score': float(round(score_value / 100, 4))
                    }

                    print(f"Image: {dominant_emotion} (Score: {score_value:.2f}%) → {mapped_emotion}")

                except ValueError as e:
                    if "Face could not be detected" in str(e):
                        response = {'label': 'Neutral', 'score': 1.0}
                        print("Aucune face détectée, retour au Neutre.")
                    else:
                        print(f"Erreur DeepFace: {str(e)}")
                        return jsonify({'error': f'Erreur DeepFace: {str(e)}'}), 500
                except Exception as e:
                    print(f"Erreur DeepFace générale: {str(e)}")
                    return jsonify({'error': f'Erreur DeepFace générale: {str(e)}'}), 500

        # Analyse de texte
        elif input_text:
            emotion_result = detect_emotion_multilang(input_text)
            detected_emotion = emotion_result['label'].lower()

            mapped_emotion = EMOTION_MAPPING.get(detected_emotion, 'neutral')

            response = {
                'label': mapped_emotion.capitalize(),
                'score': round(emotion_result['score'], 4)
            }

            print(f"Texte: {detected_emotion} (Score: {emotion_result['score']:.4f}) → {mapped_emotion}")

        else:
            return jsonify({'error': 'Veuillez fournir du texte ou une image'}), 400

        # Réponse finale de la détection
        return jsonify(response)

    except Exception as e:
        print(f" Erreur inattendue dans /analyze_mood: {e}")
        return jsonify({'error': f'Erreur interne du serveur: {str(e)}'}), 500


@app.route('/get_recommendations/<emotion>', methods=['GET'])
def get_recommendations_api(emotion):
    """Récupère les recommandations basées sur l'émotion (MODIFIÉE)"""
    if not recommender:
        return jsonify({'error': 'Le système de recommandation n\'est pas initialisé.'}), 503

    emotion = emotion.lower()

    try:
        # Utilisation des nouvelles méthodes pour obtenir des listes
        recommendations = {
            'emotion': emotion.capitalize(),
            'movies': recommender.recommend_movies(emotion), # Max 6 films
            'music': recommender.recommend_music(emotion), # Max 8 playlists
            'meal': recommender.recommend_meals_multiple(emotion), # 3 plats
            'activity': recommender.recommend_activities_multiple(emotion) # 3 activités
        }

        return jsonify(recommendations)

    except Exception as e:
        print(f"❌ Erreur lors de la récupération des recommandations pour {emotion}: {e}")
        return jsonify({'error': f'Erreur lors de la récupération des recommandations: {str(e)}'}), 500

# ===================================================================
# LANCEMENT DU SERVEUR
# ===================================================================

def run_server():
    print(f"Démarrage du serveur Flask sur le port {FLASK_PORT}...")

    # 1. Configurer ngrok
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    try:
        http_tunnel = ngrok.connect(FLASK_PORT, bind_tls=True)
        public_url = http_tunnel.public_url
        print(f" Tunnel ngrok créé: {public_url}")

        # Le frontend a besoin de cette URL pour l'API
        print("\n=======================================================")
        print(f" METTRE À JOUR LE FRONTEND AVEC CETTE URL ")
        print(f"CONST API_BASE_URL = '{public_url}'")
        print("=======================================================\n")

        # 2. Lancer Flask
        app.run(port=FLASK_PORT, debug=False, use_reloader=False)

    except Exception as e:
        print(f"Erreur ngrok ou Flask: {e}")


if __name__ == '__main__':
    run_server()






