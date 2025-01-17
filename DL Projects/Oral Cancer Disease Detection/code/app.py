from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import hashlib
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

app.secret_key = os.urandom(24)  # Use a secure random secret keyss

# Load the model
model = load_model('MobileNetV2_oral_cancer.h5')
class_names = ['augmented_data', 'original_data']

folder_class_mapping = {
    'augmented_benign': 'benign',
    'benign_lesions': 'benign',
    'augmented_malignant': 'malignant',
    'malignant_lesions': 'malignant'
}

# In-memory user storage (for demo purposes)
users = {}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB limit

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

    # Define the two categories (original_data, augmented_data)
    class_names = ['augmented_data', 'original_data']
    output_classes = {
        'augmented_data': ['augmented_benign', 'augmented_malignant'],
        'original_data': ['benign_lesions', 'malignant_lesions']
    }

    # Define mappings for the output classes
    augmented_data_mapping = {
        'augmented_benign': 'benign',
        'augmented_malignant': 'malignant'
    }

    original_data_mapping = {
        'benign_lesions': 'benign',
        'malignant_lesions': 'malignant'
    }

    # Determine the predicted class category (original_data or augmented_data)
    predicted_category = class_names[predicted_class_idx]
    
    # Based on the predicted category, get the corresponding output class
    if predicted_category == 'augmented_data':
        output_class = output_classes['augmented_data'][predicted_class_idx]
        # Mapping for augmented data
        predicted_class = augmented_data_mapping.get(output_class, 'Unknown')
    else:
        output_class = output_classes['original_data'][predicted_class_idx]
        # Mapping for original data
        predicted_class = original_data_mapping.get(output_class, 'Unknown')

    return predicted_class, predictions[0][predicted_class_idx]


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

        # Hash the password and store the new user in memory
        hashed_password = generate_password_hash(password)
        users[username] = hashed_password
        flash('Signup successful! You can now login.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username exists and the password matches
        if username in users and check_password_hash(users[username], password):
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
                # Create image paths for the color conversions (RGB, HSV, etc.)
                real_image_path = f'/static/uploads/{file.filename}'
                rgb_image_path = f'/static/uploads/rgb_{file.filename}'
                hsv_image_path = f'/static/uploads/hsv_{file.filename}'
                ycbcr_image_path = f'/static/uploads/ycbcr_{file.filename}'
                hls_image_path = f'/static/uploads/hls_{file.filename}'
                xyz_image_path= f'/static/uploads/xyz_{file.filename}'

                # Optional: Generate and save color-converted images (example)
                image = Image.open(file_path)
                rgb_image = image.convert('RGB')
                rgb_image.save(f'static/uploads/rgb_{file.filename}')

                # For HSV, YCbCr, HLS, you could use OpenCV (cv2) or PIL to convert
                # Example with OpenCV (ensure OpenCV is installed)
                import cv2
                img = cv2.imread(file_path)

                hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                cv2.imwrite(f'static/uploads/hsv_{file.filename}', hsv_image)

                ycbcr_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                cv2.imwrite(f'static/uploads/ycbcr_{file.filename}', ycbcr_image)

                hls_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
                cv2.imwrite(f'static/uploads/hls_{file.filename}', hls_image)
                
                xyz_image = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
                cv2.imwrite(f'static/uploads/xyz_{file.filename}', xyz_image)

                return render_template(
                    'result.html',
                    disease=predicted_class,
                    accuracy=round(accuracy * 100, 2),
                    real_image_path=real_image_path,
                    rgb_image_path=rgb_image_path,
                    hsv_image_path=hsv_image_path,
                    ycbcr_image_path=ycbcr_image_path,
                    hls_image_path=hls_image_path,
                    xyz_image_path=xyz_image_path
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
    labels=['augmented_data','original_data']
    values=[2310,323]
    return render_template('performance.html',labels=labels,values=values)

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(port=5002)