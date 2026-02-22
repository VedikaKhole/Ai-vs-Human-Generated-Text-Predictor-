import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources (only first time)
nltk.download('stopwords')
nltk.download('wordnet')

# Load trained model and vectorizer
model = joblib.load("AI_generated_predictor.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaning function (SAME as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Streamlit UI
st.title("AI vs Human Text Detector")
st.write("Enter a text passage to check whether it is AI-generated or Human-written.")

user_input = st.text_area("Enter Text Here", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 0:
            st.success("🧑 Human Generated Text")
        else:
            st.success("🤖 AI Generated Text")
