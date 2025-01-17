from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tfp
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)

app.secret_key = 'abcd123'

# Load the model
model = load_model('NASNetMobile_fruit_classification.h5')
class_names = ['Bad Quality_Fruits', 'Good Quality_Fruits', 'Mixed Quality_Fruits']
users = {}
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 30 * 1024 * 1024  

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    """Check if the uploaded file is a valid image."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def import_and_predict(image_path, model):
    """Process the image and use the model for prediction."""
    image = Image.open(image_path).convert('RGB')
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    img = np.asarray(image)
    img = img / 255.0

    if img.shape[-1] == 4:
        img = img[..., :3]

    img_reshape = np.expand_dims(img, axis=0)
    predictions = model.predict(img_reshape)

    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]

    return predicted_class, confidence

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username already exists
        if username in users:
            flash('Username already exists. Please choose a different one.', 'error')
            return redirect(url_for('signup'))

        # Store the new user in memory (no password hashing for simplicity)
        users[username] = password
        flash('Signup successful! You can now login.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username exists and the password matches
        if username in users and users[username] == password:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('predict'))
        else:
            flash('Invalid username or password. Please try again.', 'error')
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
        if file and allowed_file(file.filename):
            try:
                # Save the uploaded image
                file_path = os.path.join('static/uploads', file.filename)
                file.save(file_path)

                # Process the image and get predictions
                predicted_class, accuracy = import_and_predict(file_path, model)

                # Remedy based on prediction
                remedies = {
                    "Bad Quality_Fruits": "This fruit is of bad quality, discard it.",
                    "Good Quality_Fruits": "This fruit is of good quality, safe to eat.",
                    "Mixed Quality_Fruits": "The fruit quality is mixed, proceed with caution.",
                }

                remedy = remedies.get(predicted_class, "No remedy available.")

                # Convert to L*a*b* color space using OpenCV for grading
                try:
                    # Read the image using OpenCV
                    image_cv = cv2.imread(file_path)
                    
                    # Convert the image to L*a*b* color space
                    lab_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2Lab)
                    
                    # Split into L, a, b channels
                    L, a, b = cv2.split(lab_image)
                    
                    # Process the L (lightness) channel
                    graded_L = cv2.equalizeHist(L)
                    
                    # Merge the graded L channel back with a and b channels
                    graded_lab = cv2.merge((graded_L, a, b))
                    
                    # Convert back to BGR for saving as an image
                    graded_bgr = cv2.cvtColor(graded_lab, cv2.COLOR_Lab2BGR)
                    
                    # Save the grading image
                    grading_image_path = f'static/uploads/grading_{file.filename}'
                    cv2.imwrite(grading_image_path, graded_bgr)
                except Exception as e:
                    flash(f'Error in grading image processing: {str(e)}', 'error')
                    return redirect(url_for('index'))

                # Real image path
                real_image_path = f'/static/uploads/{file.filename}'

                # Return the result page with the real and grading images
                return render_template(
                    'result.html',
                    disease=predicted_class,
                    accuracy=round(accuracy * 100, 2),
                    remedy=remedy,
                    real_image_path=real_image_path,
                    grading_image_path=grading_image_path
                )
            except Exception as e:
                flash(f'Error: {str(e)}', 'error')
                return redirect(url_for('index'))
        else:
            flash('Invalid file format or file is too large.', 'error')
            return redirect(url_for('index'))
    return render_template('index.html')
@app.route('/performance')
def performance():
    labels = ['Bad Quality_Fruits', 'Good Quality_Fruits', 'Mixed Quality_Fruits']  
    values = [6788, 11664, 1074]
    return render_template('performance.html', labels=labels, values=values)

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(port=5001)
