import os
import pickle
import uuid
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from .models import User, Customer, db

main = Blueprint('main', __name__)

base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, '../ml_model/model.pkl')
scaler_path = os.path.join(base_path, '../ml_model/scaler.pkl')

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    model = None
    scaler = None

@main.route('/')
def home():
    return render_template('home.html')

@main.route('/dashboard')
@login_required
def dashboard():
    total = Customer.query.count()
    churn = Customer.query.filter_by(churn='Yes').count()
    safe = Customer.query.filter_by(churn='No').count()
    rate = round((churn / total * 100), 1) if total > 0 else 0
    return render_template('dashboard.html', user=current_user, total=total, churn_c=churn, safe_c=safe, rate=rate)

@main.route('/customers')
@login_required
def customers_list():
    customers = Customer.query.order_by(Customer.id.desc()).all()
    return render_template('customers.html', customers=customers)

@main.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('main.dashboard'))
        else:
            flash('Invalid credentials.', 'danger')
    return render_template('auth/login.html')

@main.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already exists.', 'warning')
        else:
            new_user = User(email=email, username=request.form.get('username'), password_hash=generate_password_hash(request.form.get('password'), method='scrypt'))
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user)
            return redirect(url_for('main.dashboard'))
    return render_template('auth/register.html')

@main.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.login'))

@main.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    if not model:
        flash("Model not loaded.", "danger")
        return redirect(url_for('main.dashboard'))

    try:
        f = request.form
        def map_yes_no(val): return 1 if val == 'Yes' else 0
        def map_service(val): return 2 if val == 'Yes' else (1 if 'No internet' in val else 0)

        features = [
            1 if f.get('gender') == 'Male' else 0,
            int(f.get('SeniorCitizen')),
            map_yes_no(f.get('Partner')),
            map_yes_no(f.get('Dependents')),
            int(f.get('tenure')),
            map_yes_no(f.get('PhoneService')),
            2 if f.get('MultipleLines')=='Yes' else (1 if 'No phone' in f.get('MultipleLines') else 0),
            1 if f.get('InternetService')=='Fiber optic' else (0 if f.get('InternetService')=='DSL' else 2),
            map_service(f.get('OnlineSecurity')),
            map_service(f.get('OnlineBackup', 'No')),
            map_service(f.get('DeviceProtection')),
            map_service(f.get('TechSupport')),
            map_service(f.get('StreamingTV', 'No')),
            map_service(f.get('StreamingMovies', 'No')),
            1 if f.get('Contract')=='One year' else (2 if f.get('Contract')=='Two year' else 0),
            map_yes_no(f.get('PaperlessBilling')),
            0 if 'Bank' in f.get('PaymentMethod') else (1 if 'Credit' in f.get('PaymentMethod') else (3 if 'Mailed' in f.get('PaymentMethod') else 2)),
            float(f.get('MonthlyCharges')),
            float(f.get('TotalCharges'))
        ]

        final_features = scaler.transform([features])
        prediction = model.predict(final_features)
        proba = model.predict_proba(final_features)[0][1]
        
        result = "Yes" if prediction[0] == 1 else "No"
        score = round(proba * 100, 2)
        ui_text = "CHURN" if result == "Yes" else "SAFE"

        new_customer = Customer(
            customer_id=str(uuid.uuid4())[:8],
            gender=f.get('gender'), contract=f.get('Contract'),
            monthly_charges=float(f.get('MonthlyCharges')),
            total_charges=float(f.get('TotalCharges')),
            churn=result, churn_probability=score
        )
        db.session.add(new_customer)
        db.session.commit()

        total = Customer.query.count()
        churn_c = Customer.query.filter_by(churn='Yes').count()
        safe_c = Customer.query.filter_by(churn='No').count()
        rate = round((churn_c / total * 100), 1) if total > 0 else 0
        
        return render_template('dashboard.html', prediction_text=ui_text, risk_score=score, user=current_user, total=total, churn_c=churn_c, safe_c=safe_c, rate=rate)

    except Exception as e:
        flash(f"Error: {e}", "danger")
        return redirect(url_for('main.predict'))