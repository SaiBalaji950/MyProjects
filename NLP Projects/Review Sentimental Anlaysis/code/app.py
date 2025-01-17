from flask import Flask, request, jsonify, render_template, redirect, session, url_for, flash
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os  # Make sure this line is added
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

# Flask app
app = Flask(__name__)
app.secret_key = 'abcd123'
users = {}

# Load trained model
model = tf.keras.models.load_model('Reviews.h5')

# Recreate tokenizer
MAX_WORDS = 10000
MAX_LEN = 200
stop_words = set(stopwords.words('english'))

# Load dataset and preprocess reviews
a = pd.read_csv('amazon_reviews.csv')
data = a.drop(['itemName', 'verified', 'feature', 'reviewTime', 'summary'], axis=1)

def clean_text(text):
    """Clean text by removing non-alphabet characters and stopwords."""
    if isinstance(text, str):
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabet characters
        text = ' '.join(word for word in text.split() if word.lower() not in stop_words)  # Remove stopwords
    else:
        text = ""
    return text

data['cleaned_reviewText'] = data['reviewText'].apply(clean_text)

# Fit tokenizer
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(data['cleaned_reviewText'])

def preprocess_review(review_text):
    """Preprocess a single review for prediction."""
    cleaned_text = clean_text(review_text)
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequences, maxlen=MAX_LEN)
    return padded

y_test_true = [1, 0, 1, 1, 0]  
y_test_pred = [1, 0, 1, 0, 0]  
y_train_true = [1, 1, 0, 0, 1]  
y_train_pred = [1, 1, 0, 1, 1]  
df = pd.DataFrame({
    'reviewText': ['Good product', 'Bad quality', 'Excellent', 'Not worth it', 'Okay product'],
    'rating': [5, 1, 4, 2, 3]
})

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
        review_text = request.form['reviewText']
        if not review_text:
            raise ValueError("Review text is required.")
        processed_review = preprocess_review(review_text)
        prediction = model.predict(processed_review)
        predicted_rating = np.argmax(prediction) + 1
        helpfulness = "Helpful" if predicted_rating in [3, 4, 5] else "Unhelpful"
        return render_template(
            'result.html',
            review=review_text,
            rating=predicted_rating,
            helpfulness=helpfulness,
            stars='★' * predicted_rating + '☆' * (5 - predicted_rating)
        )
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/performance')
def performance():
    # Confusion Matrix for test and train
    cm_test = confusion_matrix(y_test_true, y_test_pred)
    cm_train = confusion_matrix(y_train_true, y_train_pred)

    # Plot confusion matrix for test set
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_test, annot=True, fmt='g', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    cm_test_path = os.path.join('static', 'cm_test.png')
    plt.savefig(cm_test_path)
    plt.close()

    # Plot confusion matrix for train set
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_train, annot=True, fmt='g', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title("Confusion Matrix (Train Set)")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    cm_train_path = os.path.join('static', 'cm_train.png')
    plt.savefig(cm_train_path)
    plt.close()

    # Pie chart for the distribution of ratings
    

    return render_template('performance.html', cm_test_path=cm_test_path, cm_train_path=cm_train_path)
@app.route('/chart')
def chart():
    rating_counts = df['rating'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Blues", len(rating_counts)))
    plt.title('Distribution of Ratings')
    pie_chart_path = os.path.join('static', 'rating_distribution.png')
    plt.savefig(pie_chart_path)
    plt.close()
    return render_template('chart.html',pie_chart_path=pie_chart_path)
if __name__ == "__main__":
    app.run(port=5001)
