# from flask import Flask, request, jsonify
# import joblib
# import re
# import random

# app = Flask(__name__)

# #Load Trained Models
# svm_model = joblib.load("language_detector_svm.pkl")
# tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# #Predefined Responses in Different Languages
# responses = {
#     "English": ["Hello! How can I help you?", "Nice to meet you!", "How's your day going?"],
#     "French": ["Bonjour! Comment puis-je vous aider?", "Enchanté!", "Comment se passe votre journée?"],
#     "Spanish": ["¡Hola! ¿Cómo puedo ayudarte?", "¡Mucho gusto!", "¿Cómo va tu día?"],
#     "German": ["Hallo! Wie kann ich helfen?", "Schön, dich kennenzulernen!", "Wie läuft dein Tag?"],
#     "Italian": ["Ciao! Come posso aiutarti?", "Piacere di conoscerti!", "Come sta andando la giornata?"]
# }

# #Text Preprocessing Function
# def preprocess_text(text):
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
#     text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
#     return text

# #Function to Detect Language
# def detect_language(text):
#     text = preprocess_text(text)
#     text_tfidf = tfidf_vectorizer.transform([text])
#     prediction = svm_model.predict(text_tfidf)[0]
#     return prediction

# #Flask Route for Chatbot
# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.json
#     user_message = data.get("message", "")

#     if not user_message:
#         return jsonify({"error": "No message provided"}), 400

#     # Detect Language
#     detected_language = detect_language(user_message)

#     # Get Response in Detected Language
#     chatbot_reply = random.choice(responses.get(detected_language, ["Sorry, I don't understand."]))

#     return jsonify({"language": detected_language, "response": chatbot_reply})

# # Run Flask App
# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, request, jsonify
import joblib
import re
import random
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Configure logging
logging.basicConfig(level=logging.INFO)

# Load Trained Models with Exception Handling
try:
    svm_model = joblib.load("language_detector_svm.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    logging.error("Error loading models: %s", e)
    raise

# Predefined Responses in Different Languages
responses = {
    "English": ["Hello! How can I help you?", "Nice to meet you!", "How's your day going?"],
    "French": ["Bonjour! Comment puis-je vous aider?", "Enchanté!", "Comment se passe votre journée?"],
    "Spanish": ["¡Hola! ¿Cómo puedo ayudarte?", "¡Mucho gusto!", "¿Cómo va tu día?"],
    "German": ["Hallo! Wie kann ich helfen?", "Schön, dich kennenzulernen!", "Wie läuft dein Tag?"],
    "Italian": ["Ciao! Come posso aiutarti?", "Piacere di conoscerti!", "Come sta andando la giornata?"]
}

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    # For multilingual support, adjust regex if necessary
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Function to Detect Language
def detect_language(text):
    processed_text = preprocess_text(text)
    text_tfidf = tfidf_vectorizer.transform([processed_text])
    prediction = svm_model.predict(text_tfidf)[0]
    return prediction

# Flask Route for Chatbot
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Detect Language
        detected_language = detect_language(user_message)
        logging.info("Detected language: %s", detected_language)
    except Exception as e:
        logging.error("Language detection error: %s", e)
        return jsonify({"error": "Language detection failed"}), 500

    # Get Response in Detected Language
    chatbot_reply = random.choice(responses.get(detected_language, ["Sorry, I don't understand."]))

    return jsonify({"language": detected_language, "response": chatbot_reply})

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)  # Set debug=False in production
