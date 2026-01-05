# Flask Web Development Complete Guide

Comprehensive guide to building web applications and APIs with Flask for data science and machine learning.

## Table of Contents

- [Introduction to Flask](#introduction-to-flask)
- [Getting Started](#getting-started)
- [Routing and Views](#routing-and-views)
- [Templates and Jinja2](#templates-and-jinja2)
- [Forms and User Input](#forms-and-user-input)
- [REST APIs](#rest-apis)
- [Database Integration](#database-integration)
- [Authentication and Sessions](#authentication-and-sessions)
- [Deployment](#deployment)
- [Flask vs Streamlit](#flask-vs-streamlit)
- [Practice Exercises](#practice-exercises)

---

## Introduction to Flask

### What is Flask?

Flask is a lightweight, flexible Python web framework that makes it easy to build web applications and APIs. It's perfect for:
- Building REST APIs for ML models
- Creating web dashboards
- Developing full-stack applications
- Prototyping quickly

### Why Flask for Data Science?

**Advantages:**
- **Lightweight**: Minimal dependencies, easy to learn
- **Flexible**: You choose what you need
- **Pythonic**: Works seamlessly with data science libraries
- **RESTful**: Easy to build APIs
- **Extensible**: Large ecosystem of extensions

**Use Cases:**
- ML model APIs
- Data visualization dashboards
- Web scraping interfaces
- Data processing services
- Authentication systems

### Flask vs Streamlit

| Feature | Flask | Streamlit |
|---------|-------|-----------|
| **Control** | Full control | Limited customization |
| **Learning Curve** | Moderate | Easy |
| **Use Case** | Custom apps, APIs | Quick dashboards |
| **Deployment** | More setup | Easier |
| **Flexibility** | High | Medium |

**Choose Flask when:**
- You need custom UI/UX
- Building REST APIs
- Need authentication/authorization
- Complex routing requirements
- Full control over frontend

**Choose Streamlit when:**
- Quick prototyping
- Simple dashboards
- Minimal frontend code
- Rapid development

---

## Getting Started

### Installation

```bash
pip install flask flask-cors
```

### Basic Flask Application

```python
from flask import Flask

# Create Flask application instance
app = Flask(__name__)

# Define route
@app.route('/')
def home():
    return '<h1>Welcome to Flask!</h1>'

# Run application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

**Run the application:**
```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

### Application Structure

```
project/
├── app.py                 # Main application file
├── templates/             # HTML templates
│   ├── base.html
│   └── index.html
├── static/               # Static files (CSS, JS, images)
│   ├── css/
│   └── js/
├── models/               # ML models
├── utils/                # Utility functions
└── requirements.txt
```

---

## Routing and Views

### Basic Routes

```python
from flask import Flask

app = Flask(__name__)

# Simple route
@app.route('/')
def index():
    return 'Home Page'

# Route with path variable
@app.route('/user/<username>')
def show_user(username):
    return f'User: {username}'

# Route with type conversion
@app.route('/post/<int:post_id>')
def show_post(post_id):
    return f'Post ID: {post_id}'

# Multiple routes for same function
@app.route('/about')
@app.route('/info')
def about():
    return 'About Page'

# HTTP methods
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Handle POST request
        return 'Login successful'
    # Handle GET request
    return 'Login form'
```

### URL Building

```python
from flask import url_for

# Generate URLs
url_for('index')  # Returns '/'
url_for('show_user', username='john')  # Returns '/user/john'
```

---

## Templates and Jinja2

### Basic Templates

**templates/base.html:**
```html
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}{% endblock %}</title>
</head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a>
        <a href="{{ url_for('about') }}">About</a>
    </nav>
    {% block content %}{% endblock %}
</body>
</html>
```

**templates/index.html:**
```html
{% extends "base.html" %}

{% block title %}Home{% endblock %}

{% block content %}
<h1>Welcome</h1>
<p>Hello, {{ name }}!</p>
{% endblock %}
```

**Rendering templates:**
```python
from flask import render_template

@app.route('/')
def index():
    return render_template('index.html', name='John')
```

### Template Variables and Filters

```python
@app.route('/dashboard')
def dashboard():
    data = {
        'users': 150,
        'revenue': 50000,
        'date': datetime.now()
    }
    return render_template('dashboard.html', **data)
```

**Template with filters:**
```html
<p>Users: {{ users|int }}</p>
<p>Revenue: ${{ revenue|currency }}</p>
<p>Date: {{ date|strftime('%Y-%m-%d') }}</p>
```

### Control Structures

```html
{% if users > 100 %}
    <p>High user count!</p>
{% else %}
    <p>Growing user base</p>
{% endif %}

{% for user in users %}
    <li>{{ user.name }}</li>
{% endfor %}
```

---

## Forms and User Input

### Handling Forms

**templates/form.html:**
```html
<form method="POST" action="{{ url_for('submit_form') }}">
    <input type="text" name="username" required>
    <input type="email" name="email" required>
    <button type="submit">Submit</button>
</form>
```

**Processing form data:**
```python
from flask import request, redirect, url_for, flash

@app.route('/form', methods=['GET', 'POST'])
def submit_form():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        
        # Process data
        flash(f'Welcome, {username}!')
        return redirect(url_for('index'))
    
    return render_template('form.html')
```

### Using Flask-WTF (Recommended)

```bash
pip install flask-wtf
```

```python
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Email

class ContactForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Submit')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    form = ContactForm()
    if form.validate_on_submit():
        # Process form
        flash('Form submitted successfully!')
        return redirect(url_for('index'))
    return render_template('contact.html', form=form)
```

---

## REST APIs

### Building REST APIs

```python
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for API

# GET endpoint
@app.route('/api/users', methods=['GET'])
def get_users():
    users = [
        {'id': 1, 'name': 'John'},
        {'id': 2, 'name': 'Jane'}
    ]
    return jsonify(users)

# POST endpoint
@app.route('/api/users', methods=['POST'])
def create_user():
    data = request.get_json()
    # Process data
    return jsonify({'message': 'User created', 'id': 1}), 201

# PUT endpoint
@app.route('/api/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    data = request.get_json()
    # Update user
    return jsonify({'message': 'User updated'})

# DELETE endpoint
@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    # Delete user
    return jsonify({'message': 'User deleted'}), 200
```

### ML Model API

```python
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].tolist()
        
        return jsonify({
            'prediction': int(prediction),
            'probability': probability
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
```

**Testing the API:**
```python
import requests

response = requests.post('http://localhost:5000/api/predict', 
                        json={'features': [1, 2, 3, 4]})
print(response.json())
```

---

## Database Integration

### SQLite with Flask

```python
from flask import Flask
import sqlite3

app = Flask(__name__)

def get_db():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/api/data')
def get_data():
    conn = get_db()
    cursor = conn.execute('SELECT * FROM users')
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(users)
```

### SQLAlchemy (Recommended)

```bash
pip install flask-sqlalchemy
```

```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email
        }

# Create tables
with app.app_context():
    db.create_all()

@app.route('/api/users')
def get_users():
    users = User.query.all()
    return jsonify([user.to_dict() for user in users])
```

---

## Authentication and Sessions

### Basic Session Management

```python
from flask import Flask, session, redirect, url_for, request

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Verify credentials
        if username == 'admin' and password == 'password':
            session['username'] = username
            return redirect(url_for('dashboard'))
    
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return f'Welcome, {session["username"]}!'
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))
```

### Flask-Login (Recommended)

```bash
pip install flask-login
```

```python
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required

app = Flask(__name__)
app.secret_key = 'your-secret-key'
login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route('/login', methods=['POST'])
def login():
    # Verify credentials
    user = User(1)
    login_user(user)
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
@login_required
def dashboard():
    return 'Protected dashboard'

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))
```

---

## Deployment

### Production Server

```python
# Use Gunicorn for production
# pip install gunicorn

# Run: gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

**Build and run:**
```bash
docker build -t flask-app .
docker run -p 5000:5000 flask-app
```

### Environment Variables

```python
import os
from flask import Flask

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key')
app.config['DATABASE_URL'] = os.environ.get('DATABASE_URL')
```

---

## Flask vs Streamlit

### When to Use Flask

- Building REST APIs
- Custom UI/UX requirements
- Complex routing
- Authentication/authorization needed
- Full control over frontend
- Production applications

### When to Use Streamlit

- Quick prototyping
- Simple dashboards
- Data exploration tools
- Minimal frontend code
- Rapid development
- Internal tools

### Example: Same App in Both

**Flask:**
```python
@app.route('/dashboard')
def dashboard():
    data = get_data()
    return render_template('dashboard.html', data=data)
```

**Streamlit:**
```python
import streamlit as st

data = get_data()
st.dataframe(data)
st.plotly_chart(create_chart(data))
```

---

## Practice Exercises

### Exercise 1: Basic Flask App
Create a Flask app with:
- Home page
- About page
- Contact form

### Exercise 2: ML Model API
Build a REST API that:
- Accepts feature data
- Returns predictions
- Handles errors gracefully

### Exercise 3: Dashboard
Create a web dashboard that:
- Displays data visualizations
- Has user authentication
- Shows real-time updates

---

## Additional Resources

**Official Documentation:**
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Flask Tutorial](https://flask.palletsprojects.com/tutorial/)

**Extensions:**
- Flask-SQLAlchemy: Database ORM
- Flask-Login: User authentication
- Flask-WTF: Form handling
- Flask-CORS: Cross-origin requests
- Flask-RESTful: REST API building

**Best Practices:**
- Use environment variables for configuration
- Implement error handling
- Use blueprints for large applications
- Add logging
- Write tests
- Use production server (Gunicorn)

---

**Remember**: Flask gives you flexibility and control. Start simple and add features as needed!

