import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import spacy
import nltk
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

nltk.download('punkt')
nltk.download('stopwords')

# Initialize Flask App
app = Flask(__name__)

# Paths & Configurations
DATA_PATH = 'food2.csv'
MODEL_PATH = 'food_nutrition_model.h5'
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Food Nutrition Analyzer Class
class FoodNutritionAnalyzer:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, on_bad_lines='skip', quotechar='"', delimiter=',')
        self._preprocess_data()
        self.nlp = spacy.load("en_core_web_sm")
        self.summarizer = pipeline("summarization")
        self.model = self.load_model()

    def _preprocess_data(self):
        self.df['Allergens'].fillna('None', inplace=True)
        scaler = MinMaxScaler()
        nutrient_cols = ['Calories', 'Protein (g)', 'Carbs (g)', 'Fats (g)']
        self.df[nutrient_cols] = scaler.fit_transform(self.df[nutrient_cols])
        self.df['Nutrient Density'] = self.df[nutrient_cols].sum(axis=1)
        self.df = pd.get_dummies(self.df, columns=['Category', 'Type'], drop_first=True)
        self.df['Allergens_None'] = self.df['Allergens'].apply(lambda x: 1 if x == 'None' else 0)

        self.X = self.df.drop(['ID', 'Name', 'Description', 'Ingredients'], axis=1, errors='ignore')
        self.y = self.df.get('Allergens_None', pd.Series(np.zeros(len(self.df))))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file {MODEL_PATH} not found.")
            return None
        try:
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def train_random_forest_classifier(self):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        y_pred = rf_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy, classification_report(self.y_test, y_pred)

    def process_food_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                return "Error: Unable to read image", None

            img = cv2.resize(img, (224, 224))
            img = np.expand_dims(img, axis=0) / 255.0

            if self.model is None:
                return "Error: Model not available", None

            predictions = self.model.predict(img)
            predicted_class = np.argmax(predictions, axis=1)[0]

            if predicted_class >= len(self.df):
                return "Unknown Food Item", None

            food_details = self.df.iloc[predicted_class]
            return {
                'food_item': food_details['Name'],
                'calories': food_details['Calories'],
                'protein': food_details['Protein (g)'],
                'fat': food_details['Fats (g)'],
                'ingredients': food_details['Ingredients'],
                'description': food_details['Description']
            }
        except Exception as e:
            return f"Error: {e}", None

    def suggest_consumption(self, food_item, user_allergy):
        food_details = self.df[self.df['Name'] == food_item]
        if food_details.empty:
            return "Food item not found."

        allergens = food_details['Allergens'].values[0]
        if isinstance(allergens, str) and user_allergy.lower() in allergens.lower():
            return f"Warning: This food contains {user_allergy}. Avoid consumption!"
        return "Safe for consumption."

food_analyzer = FoodNutritionAnalyzer(DATA_PATH)

# ✅ Routes
@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/monitoring')
def monitoring():
    return render_template('Moniter.html')

@app.route('/train_model', methods=['POST'])
def train_model():
    accuracy, report = food_analyzer.train_random_forest_classifier()
    return jsonify({'accuracy': accuracy, 'report': report})

# ✅ Image Upload & Prediction Route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        result = food_analyzer.process_food_image(filepath)

        if isinstance(result, dict):
            return jsonify(result)
        return jsonify({'error': result}), 400

    return jsonify({"error": "Invalid file format. Upload JPG or PNG."}), 400

# ✅ Check Allergy Route
@app.route('/check_allergy', methods=['POST'])
def check_allergy():
    food_item = request.form.get('food_item')
    user_allergy = request.form.get('allergy')

    if not food_item or not user_allergy:
        return jsonify({"error": "Food item or allergy not provided"}), 400

    suggestion = food_analyzer.suggest_consumption(food_item, user_allergy)
    return jsonify({'suggestion': suggestion})

# ✅ Utility Function to Check Allowed File Types
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
