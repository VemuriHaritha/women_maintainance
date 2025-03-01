import numpy as np
import os
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster
import google.generativeai as genai  # Import Gemini API
from dotenv import load_dotenv  # Fix: Import dotenv for loading .env variables

load_dotenv()
# Set up Gemini API Key
GEMINI_API_KEY = os.getenv("API_KEY")  # Replace with your actual Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("women_health_data.csv")
    data.fillna(data.median(), inplace=True)  # Handle missing values
    return data

data = load_data()

# Features for clustering
features = ["age", "bmi", "exercise_freq", "diet_quality", "sleep_hours", "stress_levels"]
X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hierarchical Clustering
@st.cache_resource
def hierarchical_clustering(X_scaled):
    return linkage(X_scaled, method='ward')

linkage_matrix = hierarchical_clustering(X_scaled)
num_clusters = 3  # Adjust based on dendrogram analysis
hierarchical_clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
data['Hierarchical_Cluster'] = hierarchical_clusters

# Gaussian Mixture Model (GMM)
@st.cache_resource
def train_gmm(X_scaled, num_clusters):
    gmm = GaussianMixture(n_components=num_clusters, covariance_type='full', random_state=42)
    gmm.fit(X_scaled)
    return gmm

gmm = train_gmm(X_scaled, num_clusters)
data['GMM_Cluster'] = gmm.predict(X_scaled)

# Generate personalized health advice using Gemini
def get_gemini_advice(user_input, user_features):
    prompt = f"""
    I am a women's health assistant. Based on the following health parameters:
    - Age: {user_features['age']}
    - BMI: {user_features['bmi']}
    - Exercise Frequency: {user_features['exercise_freq']}
    - Diet Quality: {user_features['diet_quality']}
    - Sleep Hours: {user_features['sleep_hours']}
    - Stress Levels: {user_features['stress_levels']}
    
    And the user's reported health concern: "{user_input}".
    
    Please provide a detailed and personalized health recommendation.
    """
    
    model = genai.GenerativeModel("gemini-1.5-pro")  # Use Gemini Pro model
    response = model.generate_content(prompt)
    
    return response.text  # Extract text response

# Streamlit UI
st.markdown(
    "<h1 style='text-align: center; color: #E75480;'>HerWellAI - Revolutionizing Women's Health with AI.</h1>",
    unsafe_allow_html=True
)

# User Input
with st.form(key='user_input_form'):
    age = st.slider("Age", 18, 80, 30)
    bmi = st.slider("BMI", 15, 40, 22)
    exercise_freq = st.slider("Exercise Frequency (times per week)", 0, 7, 3)
    diet_quality = st.slider("Diet Quality (1-10)", 1, 10, 7)
    sleep_hours = st.slider("Sleep Hours Per Night", 3, 10, 7)
    stress_levels = st.slider("Stress Levels (1-10)", 1, 10, 5)
    user_problem = st.text_area("Describe your health concerns or symptoms:")
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    user_data = {
        "age": age,
        "bmi": bmi,
        "exercise_freq": exercise_freq,
        "diet_quality": diet_quality,
        "sleep_hours": sleep_hours,
        "stress_levels": stress_levels
    }
    gemini_advice = get_gemini_advice(user_problem, user_data)
    
    st.subheader("Doctor's Advice")
    st.write(gemini_advice)
    st.balloons()

# Save processed data
data.to_csv("women_health_segmented.csv", index=False)
print("Segmentation Complete. Data saved to 'women_health_segmented.csv'")
