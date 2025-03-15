from flask import Flask, render_template, request, session, redirect, url_for, flash
import pickle
import numpy as np
import pandas as pd
from collections import Counter

app = Flask(__name__)
app.secret_key = 'abcd123'

# Load trained model and scaler
model = pickle.load(open('model.pickle', 'rb'))
scaler = pickle.load(open('scaler.pickle', 'rb'))
users = {}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/user_registration', methods=['GET', 'POST'])
def user_registration():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('user_registration'))
        if username not in users:
            users[username] = {'password': password}
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('user_login'))
        else:
            flash('User already exists', 'error')
            return redirect(url_for('user_registration'))
    return render_template('user_registration.html')

@app.route('/user_login', methods=['GET', 'POST'])
def user_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            session['username'] = username
            flash('Successfully logged in', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
            return redirect(url_for('user_login'))
    return render_template('user_login.html')

@app.route('/index')
def index():
    if 'username' in session:
        return render_template('index.html')
    else:
        flash('You need to log in first', 'error')
        return redirect(url_for('user_login'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            # Collect features in the correct order
            data = [
                float(request.form['baseline_value']),
                float(request.form['accelerations']),
                float(request.form['fetal_movement']),
                float(request.form['uterine_contractions']),
                float(request.form['light_decelerations']),
                float(request.form['severe_decelerations']),
                float(request.form['prolongued_decelerations']),
                float(request.form['abnormal_short_term_variability']),
                float(request.form['mean_value_of_short_term_variability']),
                float(request.form['percentage_of_time_with_abnormal_long_term_variability']),
                float(request.form['mean_value_of_long_term_variability']),
                float(request.form['histogram_width']),
                float(request.form['histogram_min']),
                float(request.form['histogram_max']),
                float(request.form['histogram_number_of_peaks']),
                float(request.form['histogram_number_of_zeroes']),
                float(request.form['histogram_mode']),
                float(request.form['histogram_mean']),
                float(request.form['histogram_median']),
                float(request.form['histogram_variance']),
                float(request.form['histogram_tendency'])
            ]
            
            # Create DataFrame with proper column names to avoid scaler warnings
            feature_names = [
                'baseline_value', 'accelerations', 'fetal_movement', 'uterine_contractions',
                'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
                'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
                'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability',
                'histogram_width', 'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
                'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median',
                'histogram_variance', 'histogram_tendency'
            ]
            df_input = pd.DataFrame([data], columns=feature_names)
            
            # Scale input and predict
            data_scaled = scaler.transform(df_input)
            prediction = model.predict(data_scaled)[0]
            # Map prediction to fetal health labels
            health_labels = {1: "Normal", 2: "Suspect", 3: "Pathological"}
            predicted_label = health_labels.get(prediction, "Unknown")
            
            return render_template('result.html', health=predicted_label)
    except ValueError as ve:
        flash('Invalid input value. Please enter numerical data.', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        flash('An error occurred. Please try again later.', 'error')
        return redirect(url_for('index'))

@app.route('/performance')
def performance():
    df = pd.read_csv('fetal_health.csv')
    disorder_counts = Counter(df['fetal_health'])
    labels = list(disorder_counts.keys())
    values = list(disorder_counts.values())
    return render_template('performance.html', labels=labels, values=values)

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
