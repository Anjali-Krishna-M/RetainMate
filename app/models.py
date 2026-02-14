from . import db, login_manager
from flask_login import UserMixin
from datetime import datetime

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='analyst')

class Customer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.String(50))
    
    # --- DEMOGRAPHICS ---
    gender = db.Column(db.String(10))
    senior_citizen = db.Column(db.Integer)
    partner = db.Column(db.String(5))
    dependents = db.Column(db.String(5))
    
    # --- SERVICES ---
    tenure = db.Column(db.Integer)
    phone_service = db.Column(db.String(5))
    multiple_lines = db.Column(db.String(20))     # NEW
    internet_service = db.Column(db.String(20))
    online_security = db.Column(db.String(20))
    online_backup = db.Column(db.String(20))      # NEW
    device_protection = db.Column(db.String(20))
    tech_support = db.Column(db.String(20))
    streaming_tv = db.Column(db.String(20))       # NEW
    streaming_movies = db.Column(db.String(20))   # NEW
    
    # --- BILLING ---
    contract = db.Column(db.String(20))
    paperless_billing = db.Column(db.String(5))   # NEW
    payment_method = db.Column(db.String(50))
    monthly_charges = db.Column(db.Float)
    total_charges = db.Column(db.Float)
    
    # --- AI RESULTS ---
    churn = db.Column(db.String(5))
    churn_probability = db.Column(db.Float)
    main_reason = db.Column(db.String(100))
    suggestion = db.Column(db.String(200))
    
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

class ActivityLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    action = db.Column(db.String(100))
    details = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)