from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a strong secret in production

# --- Grade to grade point mapping ---
grade_points = {
    'A+': 10, 
    'A': 9, 
    'B': 8, 
    'C': 7, 
    'D': 6, 
    'E': 5, 
    'F': 0
}

def calculate_sgpa(subjects):
    """Calculate SGPA from subjects list (grade, credits)."""
    total_points = sum(grade_points.get(grade, 0) * float(credits) for grade, credits in subjects)
    total_credits = sum(float(credits) for _, credits in subjects)
    return round(total_points / total_credits, 2) if total_credits > 0 else 0

# --- Database Setup ---
def create_users_table():
    """Creates the users table if it doesn't exist."""
    with sqlite3.connect('users.db') as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL)''')
        conn.commit()

create_users_table()  # Ensure table is created

# --- Routes ---
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        with sqlite3.connect('users.db') as conn:
            c = conn.cursor()
            try:
                c.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                          (username, password))
                conn.commit()
                flash('Signup successful! Please log in.', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Username already exists. Try logging in.', 'error')
                return redirect(url_for('signup'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        with sqlite3.connect('users.db') as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username = ? AND password = ?", 
                      (username, password))
            user = c.fetchone()

            if user:
                session['username'] = username
                flash('Login successful!', 'success')
                return redirect(url_for('student_info'))
            else:
                flash('Invalid username or password.', 'error')
                return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('home'))

@app.route('/student_info', methods=['GET', 'POST'])
def student_info():
    """Collects student details."""
    if request.method == 'POST':
        session['student'] = {
            'roll_no': request.form['roll_no'],
            'name': request.form['name'],
            'department': request.form['department']
        }
        session['semesters'] = {}
        session['subject_counts'] = {}
        return redirect(url_for('subject_count', sem=1))
    return render_template('student_info.html')

@app.route('/semester/<int:sem>/subject_count', methods=['GET', 'POST'])
def subject_count(sem):
    """Stores the number of subjects per semester."""
    if request.method == 'POST':
        count = request.form.get('subject_count')
        if count and count.isdigit() and int(count) > 0:
            subject_counts = session.get('subject_counts', {})
            subject_counts[str(sem)] = int(count)
            session['subject_counts'] = subject_counts
            return redirect(url_for('semester', sem=sem))
        else:
            return "Please enter a valid number of subjects.", 400
    return render_template('subject_count.html', sem=sem)

@app.route('/semester/<int:sem>', methods=['GET', 'POST'])
def semester(sem):
    """Input subject details for the semester."""
    subject_counts = session.get('subject_counts', {})
    subject_count_for_sem = subject_counts.get(str(sem))
    
    if not subject_count_for_sem:
        return redirect(url_for('subject_count', sem=sem))
    
    if request.method == 'POST':
        subjects = []
        for i in range(1, subject_count_for_sem + 1):
            subject = request.form.get(f'subject_{i}')
            grade = request.form.get(f'grade_{i}')
            credits = request.form.get(f'credits_{i}')
            if subject and grade and credits:
                subjects.append((grade, credits))
        semesters = session.get('semesters', {})
        semesters[str(sem)] = subjects
        session['semesters'] = semesters
        
        return redirect(url_for('subject_count', sem=sem+1)) if sem < 8 else redirect(url_for('sgpa'))
    
    return render_template('semester.html', sem=sem, subject_count=subject_count_for_sem)

@app.route('/sgpa')
def sgpa():
    """Compute and display SGPA for each semester."""
    semesters = session.get('semesters', {})
    sgpa_dict = {sem: calculate_sgpa(semesters.get(str(sem), [])) for sem in range(1, 9)}
    session['sgpa'] = sgpa_dict
    return render_template('sgpa.html', sgpa=sgpa_dict)

@app.route('/cgpa')
def cgpa():
    """Calculate overall CGPA and percentage."""
    semesters = session.get('semesters', {})
    all_subjects = [subject for sem in range(1, 9) for subject in semesters.get(str(sem), [])]
    overall_cgpa = calculate_sgpa(all_subjects)
    percentage = round(overall_cgpa * 9.5, 2)
    return render_template('cgpa.html', cgpa=overall_cgpa, percentage=percentage)

if __name__ == '__main__':
    app.run(debug=True)
