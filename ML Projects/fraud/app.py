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

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            # Ensure these keys match exactly the names in your HTML form
            data = [
                float(request.form['step']),
                float(request.form['type']),
                float(request.form['amount']),
                float(request.form['oldbalanceOrg']),
                float(request.form['newbalanceOrig']),
                float(request.form['oldbalanceDest']),
                float(request.form['newbalanceDest'])
            ]
            
            # Debug: print raw data
            print("Raw Input Data:", data)
            
            # Scale input data using the same scaler as during training
            data_scaled = scaler.transform([data])
            print("Scaled Data:", data_scaled)
            
            # Predict the sleep disorder
            prediction = model.predict(data_scaled)[0]
            print("Raw Prediction:", prediction)
            
            # Mapping predicted class to readable labels
            disorder_labels = {0: "Legitimate(Non-Fraud)", 1: "Fraud"}
            predicted_label = disorder_labels.get(prediction, "Unknown")
            
            return render_template('result.html', fraud=predicted_label)

    except ValueError as ve:
        return f"Invalid input value: {str(ve)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"
@app.route('/performance')
def performance():
    df = pd.read_csv('fraud.csv')
    disorder_counts = Counter(df['isFraud'])
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
