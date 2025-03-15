from flask import Flask, request, render_template, redirect, url_for, flash, session
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load trained XGBoost model & TF-IDF Vectorizer
try:
    with open("xgbmodel.pickle", "rb") as model_file:
        model = pickle.load(model_file)
    
    with open("TfidfVectorizer.pickle", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    if not hasattr(vectorizer, "idf_"):  # Ensure vectorizer is fitted
        raise ValueError("Error: TfidfVectorizer is not fitted!")

except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

users ={}
# Flask app
app = Flask(__name__)
app.secret_key = 'abcd123'
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
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tweet_text = request.form.get("tweetText", "").strip()  # Ensure name matches HTML form

    if not tweet_text:
        return render_template("index.html", error="Tweet text is required!")  # Stay on the same page

    print("Raw Input:", tweet_text)

    try:
        processed_tweet = vectorizer.transform([preprocess_text(tweet_text)])
        print("Processed Text:", processed_tweet)

        prediction = model.predict(processed_tweet)[0]
        print("Prediction:", prediction)

        class_mapping = {0: "Negative (Hate Speech)", 1: "Positive (Non-Hate Speech)"}  
        predicted_label = class_mapping.get(prediction, "Unknown")

        return render_template("result.html", tweet=tweet_text, result=predicted_label)

    except Exception as e:
        print(f"Prediction Error: {e}")
        return render_template("index.html", error="Error in processing the tweet!")

if __name__ == "__main__":
    app.run(debug=True)
