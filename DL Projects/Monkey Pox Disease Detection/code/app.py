from flask import Flask, render_template, request,redirect,url_for,flash,session
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
users={}
app = Flask(__name__)
app.secret_key='abcd123'
class_names = ['Chickenpox', 'Measles', 'Monkeypox', 'Normal']

# Ensure directories exist
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/processed', exist_ok=True)

def load_model():
    model = tf.keras.models.load_model('Xception_monkey_pox.h5')
    return model

with app.app_context():
    model = load_model()

def import_and_predict(image_path, model):
    image = Image.open(image_path).convert('RGB')
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    img = np.asarray(image)
    img = img / 255.0  

    if img.shape[-1] == 4:  
        img = img[..., :3]

    img_reshape = np.expand_dims(img, axis=0)  
    predictions = model.predict(img_reshape)
    return predictions

def apply_image_processing(image_path):
    # Load image in color mode
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Apply erosion
    kernel = np.ones((5, 5), np.uint8)
    eroded_image = cv2.erode(original_image, kernel, iterations=1)

    # Apply dilation
    dilated_image = cv2.dilate(original_image, kernel, iterations=1)

    # Save processed images and return their paths
    erosion_image_path = os.path.join('static/processed', 'eroded_image.jpg')
    dilation_image_path = os.path.join('static/processed', 'dilated_image.jpg')
    cv2.imwrite(erosion_image_path, eroded_image)
    cv2.imwrite(dilation_image_path, dilated_image)

    return erosion_image_path, dilation_image_path

remedies = {
    "Chickenpox": "Take antiviral medication and avoid scratching.",
    "Monkeypox": "Isolate and consult a healthcare provider.",
    "Measles": "Rest, hydrate, and use fever reducers.",
    "Normal": "No disease detected. Maintain a healthy lifestyle."
}

import time
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

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            try:
                file_path = os.path.join('static/uploads', file.filename)
                file.save(file_path)

                # Predict the disease
                predictions = import_and_predict(file_path, model)
                predicted_class = class_names[np.argmax(predictions)]
                accuracy = round(np.max(predictions) * 100, 2)
                remedy = remedies.get(predicted_class, "No remedy available.")

                # Process the image (apply erosion and dilation)
                erosion_image_path, dilation_image_path = apply_image_processing(file_path)

                # Generate a unique timestamp for cache-busting
                timestamp = int(time.time())

                return render_template(
                    'result.html',
                    disease=predicted_class,
                    accuracy=accuracy,
                    remedy=remedy,
                    real_image_path=f'/static/uploads/{file.filename}?t={timestamp}',
                    erosion_image_path=f'/static/processed/eroded_image.jpg?t={timestamp}',
                    dilation_image_path=f'/static/processed/dilated_image.jpg?t={timestamp}'
                )

            except Exception as e:
                return render_template('error.html', message=str(e))
    return render_template('index.html')
@app.route('/performance')
def performance():
    labels=['Chickenpox', 'Measles', 'Monkeypox', 'Normal']
    values=[107,91,279,293]
    return render_template('performance.html',labels=labels,values=values)
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
