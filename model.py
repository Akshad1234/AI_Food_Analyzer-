import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import spacy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

class FoodNutritionAnalyzer:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, on_bad_lines='skip', quotechar='"', delimiter=',')
        self._preprocess_data()
        self.nlp = spacy.load("en_core_web_sm")
        self.summarizer = pipeline("summarization")

    def _preprocess_data(self):
        self.df['Allergens'].fillna('None', inplace=True)  # Ensure no missing allergens
        scaler = MinMaxScaler()
        self.df[['Calories', 'Protein (g)', 'Carbs (g)', 'Fats (g)']] = scaler.fit_transform(
            self.df[['Calories', 'Protein (g)', 'Carbs (g)', 'Fats (g)']])
        self.df['Nutrient Density'] = (self.df['Calories'] + self.df['Protein (g)'] +
                                       self.df['Carbs (g)'] + self.df['Fats (g)'])
        self.df = pd.get_dummies(self.df, columns=['Category', 'Type'], drop_first=True)  # Keep 'Allergens' as-is
        self.df['Allergens_None'] = self.df['Allergens'].apply(lambda x: 1 if x == 'None' else 0)  # Mark 'None'
        self.X = self.df.drop(['ID', 'Name', 'Description', 'Ingredients'], axis=1, errors='ignore')
        self.y = self.df.get('Allergens_None', pd.Series(np.zeros(len(self.df))))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def train_random_forest_classifier(self):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        y_pred = rf_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy, classification_report(self.y_test, y_pred)

    def process_food_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            img = cv2.resize(img, (224, 224))
            img = np.expand_dims(img, axis=0)
            img = img / 255.0
            model = tf.keras.models.load_model('food_nutrition_model.h5')
            predictions = model.predict(img)
            predicted_class = np.argmax(predictions, axis=1)[0]
            if predicted_class >= len(self.df):
                return "Unknown Food Item", None
            food_item = self.df.iloc[predicted_class]['Name']
            
            # Retrieve relevant details for the detected food item
            food_details = self.df[self.df['Name'] == food_item].iloc[0]
            calories = food_details['Calories']
            protein = food_details['Protein (g)']
            fat = food_details['Fats (g)']
            ingredients = food_details['Ingredients']
            description = food_details['Description']
            
            return food_item, calories, protein, fat, ingredients, description
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

st.title("🍽️ AI-Powered Food Label & Nutrition Analyzer")
data_path = 'food2.csv'
food_analyzer = FoodNutritionAnalyzer(data_path)

if st.button("🚀 Train Random Forest Classifier"):
    accuracy, report = food_analyzer.train_random_forest_classifier()
    st.write(f"🎯 Accuracy: {accuracy}")
    st.text(report)

uploaded_file = st.file_uploader("📤 Upload a food image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    file_path = "uploaded_food.jpg"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    food_item, calories, protein, fat, ingredients, description = food_analyzer.process_food_image(file_path)
    st.write(f"🍕 Detected Food Item: {food_item}")
    
    if food_item != "Unknown Food Item":
        st.write(f"💥 Calories: {calories}")
        st.write(f"💪 Protein: {protein}g")
        st.write(f"🧈 Fats: {fat}g")
        st.write(f"📝 Ingredients: {ingredients}")
        st.write(f"📜 Description: {description}")
    
    user_allergy = st.text_input("⚠️ Enter your allergy (e.g., nuts, gluten, etc.):")
    if st.button("🔎 Check Allergy Safety") and user_allergy:
        suggestion = food_analyzer.suggest_consumption(food_item, user_allergy)
        st.write(suggestion)
