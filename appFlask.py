from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download once (safe even if already downloaded)
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load trained model & vectorizer
model = joblib.load("AI_generated_predictor.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaning (SAME as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        user_text = request.form.get("user_text", "")

        if user_text.strip():
            cleaned = clean_text(user_text)
            vectorized = vectorizer.transform([cleaned])
            result = model.predict(vectorized)[0]

            # SIMPLE labels for HTML logic
            prediction = "Human" if result == 0 else "AI"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)



