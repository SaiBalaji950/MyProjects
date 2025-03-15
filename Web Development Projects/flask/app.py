from flask import Flask, render_template, request, session, redirect, url_for, flash
from flask_mail import Mail, Message
from random import randint
import sqlite3

app = Flask(__name__)
app.secret_key = 'abcd123'

# Set up the SQLite database and create the Users table if it doesn't exist
connection = sqlite3.connect('Balaji.db', check_same_thread=False)
cursor = connection.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS Users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    )
''')
connection.commit()

# Configuring Mail Server (Using Google SMTP)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USE_TLS'] = False
mail = Mail(app)

@app.route('/', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not username or not email or not password or not confirm_password:
            flash("All fields are required!", "danger")
            return render_template('signup.html')

        if password != confirm_password:
            flash("Passwords do not match! Please try again.", "danger")
            return render_template('signup.html')

        try:
            cursor.execute("INSERT INTO Users (username, email, password) VALUES (?, ?, ?)",
                           (username, email, password))
            connection.commit()
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Email already exists!", "danger")

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            flash("Email and password are required!", "danger")
            return render_template('login.html')

        cursor.execute("SELECT * FROM Users WHERE email=? AND password=?", (email, password))
        user = cursor.fetchone()
        if user:
            session['user'] = user[1]  # username
            session['user_email'] = user[2]  # email
            flash("Login successful!", "success")
            return redirect(url_for('otp'))
        else:
            flash("Invalid credentials!", "danger")
    
    return render_template('login.html')

@app.route('/otp', methods=['GET', 'POST'])
def otp():
    if 'user_email' not in session:
        flash("Please log in first.", "danger")
        return redirect(url_for('login'))
    sender_email = session['user_email']
    if request.method == 'POST':
        app_password = request.form.get('app_password')
        recipient_email = request.form.get('recipient_email')
        session['app_password'] = app_password
        session['recipient_email'] = recipient_email
        app.config['MAIL_USERNAME'] = sender_email
        app.config['MAIL_PASSWORD'] = app_password
        global mail
        mail = Mail(app)
        try:
            otp_val = randint(100000, 999999)
            session['otp'] = otp_val
            msg = Message('Your OTP Code', sender=sender_email, recipients=[recipient_email])
            msg.body = f"Your OTP code is: {otp_val}"
            with mail.connect() as conn:
                conn.send(msg)
            flash("OTP sent successfully! Check the recipient email.", "success")
            return redirect(url_for('verify'))
        except Exception as e:
            flash(f"Failed to send OTP: {e}", "danger")
    return render_template('otp.html', sender_email=sender_email)

@app.route('/forgot-password')
def forgot_password():
    return render_template('forgot_password.html')

@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        user_otp = request.form.get('otp')
        if 'otp' in session:
            try:
                if int(user_otp) == session['otp']:
                    flash("Email verification successful!", "success")
                    session.pop('otp', None)
                    return render_template('success.html')
                else:
                    flash("Incorrect OTP. Please try again.", "danger")
                    return render_template('failure.html')
            except ValueError:
                flash("Invalid OTP format.", "danger")
                return render_template('failure.html')
        else:
            flash("No OTP generated. Please try again.", "danger")
            return redirect(url_for('otp'))
    return render_template('verify.html')

@app.route('/view')
def view():
    if 'user' not in session:
        return redirect('/login')

    cursor.execute('SELECT * FROM Users')
    result = cursor.fetchall()

    return render_template('view.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
