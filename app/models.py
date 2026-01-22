from . import db, login_manager
from flask_login import UserMixin
from datetime import datetime

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- USER TABLE (For Login) ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='analyst') # 'admin' or 'analyst'

# --- CUSTOMER TABLE (For Churn Data) ---
class Customer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.String(50), unique=True) # The ID from the CSV
    gender = db.Column(db.String(10), nullable=False)
    senior_citizen = db.Column(db.Integer)
    partner = db.Column(db.String(5))
    dependents = db.Column(db.String(5))
    tenure = db.Column(db.Integer)
    phone_service = db.Column(db.String(5))
    multiple_lines = db.Column(db.String(20))
    internet_service = db.Column(db.String(20))
    online_security = db.Column(db.String(20))
    online_backup = db.Column(db.String(20))
    device_protection = db.Column(db.String(20))
    tech_support = db.Column(db.String(20))
    streaming_tv = db.Column(db.String(20))
    streaming_movies = db.Column(db.String(20))
    contract = db.Column(db.String(20))
    paperless_billing = db.Column(db.String(5))
    payment_method = db.Column(db.String(50))
    monthly_charges = db.Column(db.Float)
    total_charges = db.Column(db.Float)
    churn = db.Column(db.String(5))
    
    # Prediction Result (We will fill this later with AI)
    churn_probability = db.Column(db.Float, default=0.0)