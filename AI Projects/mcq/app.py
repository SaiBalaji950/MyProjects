import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from tensorflow.keras.models import load_model
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'abcd123'

# Load the trained model
model = load_model('mcq_generator.h5')

# Load dataset (Assuming CSV format)
df = pd.read_csv('mcq.csv')  # Ensure the file is available

# User database (Dictionary for now, can be replaced with a DB)
users = {}

# Function to get random questions
def get_random_questions(num_questions):
    return df.sample(n=num_questions).to_dict(orient='records')

# Preprocess questions for model input
def preprocess_question(prompt):
    return np.array([prompt])  # Adjust based on model input requirements

# Predict answer using GAN model
def predict_answer(prompt):
    processed_prompt = preprocess_question(prompt)
    prediction = model.predict(processed_prompt)
    predicted_label = np.argmax(prediction)
    return chr(65 + predicted_label)  # Convert index (0-4) to ('A'-'E')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            flash('Username already exists. Try logging in.', 'error')
            return redirect(url_for('signup'))
        users[username] = generate_password_hash(password)
        flash('Signup successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and check_password_hash(users[username], password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('select_questions'))
        flash('Invalid username or password.', 'error')
        return redirect(url_for('login'))
    return render_template('login.html')
@app.route('/select_questions', methods=['GET', 'POST'])
def select_questions():
    if 'username' not in session:
        flash('You must be logged in to take the quiz.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        num_questions = int(request.form['num_questions'])
        session['num_questions'] = num_questions

        # Fetch and store the required number of questions in session
        session['questions'] = get_random_questions(num_questions)

        return redirect(url_for('quiz'))
    return render_template('select_questions.html')

@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    if 'username' not in session:
        flash('You must be logged in to take the quiz.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        user_answers = {key: request.form[key] for key in request.form}
        session['user_answers'] = user_answers
        return redirect(url_for('result'))

    # Retrieve stored questions from session
    questions = session.get('questions', [])

    if not questions:  # If no questions are stored, redirect to select_questions
        flash('No questions found. Please select the number of questions again.', 'error')
        return redirect(url_for('select_questions'))

    return render_template('quiz.html', questions=questions)


@app.route('/result')
def result():
    if 'username' not in session:
        flash('You must be logged in to view results.', 'error')
        return redirect(url_for('login'))

    questions = session.get('questions', [])
    user_answers = session.get('user_answers', {})

    correct = 0
    total = len(questions)
    detailed_results = []

    for q in questions:
        correct_option = q['answer']  # This is "A", "B", etc.
        correct_answer_text = q[correct_option]  # Get the full text (e.g., "4")

        user_answer_option = user_answers.get(q['prompt'], 'Not Answered')
        user_answer_text = q.get(user_answer_option, "Invalid Option")  # Convert A->Answer

        is_correct = (user_answer_option == correct_option)  # Compare letter only
        if is_correct:
            correct += 1

        detailed_results.append({
            "question": q['prompt'],
            "correct_answer": f"{correct_option} ({correct_answer_text})",  # A (4)
            "user_answer": f"{user_answer_option} ({user_answer_text})",  # B (5) etc.
            "is_correct": is_correct
        })

    accuracy = (correct / total) 

    return render_template('result.html', accuracy=accuracy, results=detailed_results)


@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(port=5000, debug=True)
