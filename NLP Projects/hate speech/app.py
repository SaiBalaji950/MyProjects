from flask import Flask, request, jsonify, render_template, redirect, session, url_for, flash
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load trained XGBoost model & TF-IDF Vectorizer
with open("xgbmodel.pickle", "rb") as model_file:
    model = pickle.load(model_file)

with open("TfidfVectorizer.pickle", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Flask app
app = Flask(__name__)
app.secret_key = 'abcd123'
users = {}

# Text Cleaning Function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('signup'))
        
        if username not in users:
            users[username] = {'password': password}
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('User already exists', 'error')
            return redirect(url_for('signup'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            session['username'] = username
            flash('Successfully logged in', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/index')
def index():
    if 'username' in session:
        return render_template('index.html')
    else:
        flash('You need to log in first', 'error')
        return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debugging: Print form data
        print("Form Data:", request.form)

        # Get the tweet text using the correct form field name
        tweet_text = request.form.get("reviewText", "").strip()

        if not tweet_text:
            flash("Tweet text is required!", "error")
            return redirect(url_for("index"))  # Redirect to the form

        # Preprocess and vectorize the input
        processed_tweet = vectorizer.transform([preprocess_text(tweet_text)])

        # Make a prediction
        prediction = model.predict(processed_tweet)[0]

        # Map numerical prediction to label
        class_mapping = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}
        predicted_label = class_mapping.get(prediction, "Unknown")

        return render_template("result.html", tweet=tweet_text, result=predicted_label)

    except Exception as e:
        print("Error:", str(e))
        flash("An error occurred during prediction.", "error")
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
