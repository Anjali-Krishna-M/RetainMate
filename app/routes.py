import os
import pickle
import json
import uuid
import pandas as pd
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.exc import IntegrityError
from functools import wraps
from .models import User, Customer, ActivityLog, db

main = Blueprint('main', __name__)

# --- 1. LOAD AI ASSETS ---
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, '../ml_model/model.pkl')
scaler_path = os.path.join(base_path, '../ml_model/scaler.pkl')
columns_path = os.path.join(base_path, '../ml_model/model_columns.pkl')
metrics_path = os.path.join(base_path, '../ml_model/metrics.json')

try:
    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))
    model_columns = pickle.load(open(columns_path, 'rb'))
except FileNotFoundError:
    print("❌ Error: AI models not found. Please run train_model.py first.")
    model, scaler, model_columns = None, None, []

try:
    with open(metrics_path) as f:
        ai_metrics = json.load(f)
except:
    ai_metrics = {"models": {}, "importance": {"labels": [], "data": []}}


# --- 2. RISK ENGINE ---
def generate_risk_report(data, probability):
    factors = []
    
    # Contract
    contract = data.get('Contract', 'Month-to-month')
    if contract == 'Month-to-month':
        factors.append({'name': 'Contract', 'status': 'Critical', 'detail': 'Month-to-month is highly volatile.'})
    elif contract == 'Two year':
        factors.append({'name': 'Contract', 'status': 'Secure', 'detail': 'Long-term contract locked in.'})
        
    # Tenure
    try: tenure = int(data.get('tenure', 0))
    except: tenure = 0
    if tenure < 6: factors.append({'name': 'Tenure', 'status': 'Critical', 'detail': 'New customer (<6 months).'})
    
    # Financials
    try: charges = float(data.get('MonthlyCharges', 0))
    except: charges = 0
    if charges > 80: factors.append({'name': 'Price', 'status': 'Warning', 'detail': f'High monthly cost (${charges}).'})

    # Internet Type
    internet = data.get('InternetService', 'No')
    if internet == 'Fiber optic': factors.append({'name': 'Tech', 'status': 'Warning', 'detail': 'Fiber users switch often.'})
    
    # Support (New Factor)
    tech_support = data.get('TechSupport', 'No')
    if tech_support == 'No': factors.append({'name': 'Support', 'status': 'Warning', 'detail': 'No Tech Support enabled.'})

    critical = [f for f in factors if f['status'] == 'Critical']
    
    if critical:
        main_reason = critical[0]['detail']
        suggestion = "Offer 12-Month Loyalty Discount."
    elif probability > 50:
        main_reason = "Usage Pattern Anomalies"
        suggestion = "Schedule Customer Success Call."
    else:
        main_reason = "Customer is Healthy"
        suggestion = "No action required."

    return main_reason, suggestion, factors


# --- 3. HELPER: CREATE CUSTOMER OBJECT ---
def create_customer_entry(data, result, prob, reason, suggestion):
    # Safely get all fields with defaults
    return Customer(
        customer_id=str(data.get('customerID', uuid.uuid4()))[:8],
        gender=data.get('gender', 'Unknown'),
        senior_citizen=int(data.get('SeniorCitizen', 0)),
        partner=data.get('Partner', 'No'),
        dependents=data.get('Dependents', 'No'),
        tenure=int(data.get('tenure', 0)),
        phone_service=data.get('PhoneService', 'No'),
        multiple_lines=data.get('MultipleLines', 'No'),
        internet_service=data.get('InternetService', 'No'),
        online_security=data.get('OnlineSecurity', 'No'),
        online_backup=data.get('OnlineBackup', 'No'),
        device_protection=data.get('DeviceProtection', 'No'),
        tech_support=data.get('TechSupport', 'No'),
        streaming_tv=data.get('StreamingTV', 'No'),
        streaming_movies=data.get('StreamingMovies', 'No'),
        contract=data.get('Contract', 'Month-to-month'),
        paperless_billing=data.get('PaperlessBilling', 'Yes'),
        payment_method=data.get('PaymentMethod', 'Electronic check'),
        monthly_charges=float(data.get('MonthlyCharges', 0)),
        total_charges=float(data.get('TotalCharges', 0)),
        churn=result,
        churn_probability=round(prob, 1),
        main_reason=reason,
        suggestion=suggestion
    )


# --- 4. ROUTES ---
# (Admin Decorator omitted for brevity, assumes you have it from previous code)
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            flash("⚠️ Admin privileges required.", "danger")
            return redirect(url_for('main.dashboard'))
        return f(*args, **kwargs)
    return decorated_function

@main.route('/')
def home(): return render_template('home.html')

@main.route('/dashboard')
@login_required
def dashboard():
    total = Customer.query.count()
    churn = Customer.query.filter_by(churn='Yes').count()
    rate = round((churn / total * 100), 1) if total > 0 else 0
    logs = ActivityLog.query.order_by(ActivityLog.timestamp.desc()).limit(10).all()
    return render_template('dashboard.html', total=total, churn=churn, safe=(total-churn), rate=rate, logs=logs, user=current_user, importance=ai_metrics.get('importance', {}))

@main.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    try:
        f = request.form.to_dict()
        df = pd.DataFrame([f])
        
        # Numeric Conversion
        for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Preprocessing
        df_encoded = pd.get_dummies(df)
        for col in model_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[model_columns]
        
        # Predict
        if model:
            final_features = scaler.transform(df_encoded)
            prediction = model.predict(final_features)[0]
            proba = model.predict_proba(final_features)[0][1] * 100
            result = "Yes" if prediction == 1 else "No"
        else:
            proba = 0; result = "No"

        reason, suggestion, factors = generate_risk_report(f, proba)
        
        # Save Full Data
        new_c = create_customer_entry(f, result, proba, reason, suggestion)
        db.session.add(new_c)
        db.session.add(ActivityLog(user_id=current_user.id, action="Single Prediction", details=f"Result: {result}"))
        db.session.commit()
        
        return render_template('predict.html', result=new_c, factors=factors)

    except Exception as e:
        flash(f"Error: {e}", "danger")
        return redirect(url_for('main.predict'))

@main.route('/upload_csv', methods=['GET', 'POST'])
@login_required
def upload_csv():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            try:
                df = pd.read_csv(file)
                # Cleaning
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
                if 'tenure' in df.columns: df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0)
                
                # Predict
                df_encoded = pd.get_dummies(df)
                for col in model_columns:
                    if col not in df_encoded.columns: df_encoded[col] = 0
                df_encoded = df_encoded[model_columns]
                
                if model:
                    features = scaler.transform(df_encoded)
                    predictions = model.predict(features)
                    probs = model.predict_proba(features)[:, 1] * 100
                else:
                    predictions = [0]*len(df); probs = [0]*len(df)
                
                count = 0
                for i, row in df.iterrows():
                    res = "Yes" if predictions[i] == 1 else "No"
                    row_dict = row.to_dict()
                    reason, suggest, _ = generate_risk_report(row_dict, probs[i])
                    
                    new_c = create_customer_entry(row_dict, res, probs[i], reason, suggest)
                    db.session.add(new_c)
                    count += 1
                
                db.session.add(ActivityLog(user_id=current_user.id, action="Batch Upload", details=f"Processed {count} rows"))
                db.session.commit()
                flash(f"✅ Analyzed {count} customers!", "success")
                return redirect(url_for('main.customers_list'))
            except Exception as e:
                flash(f"CSV Error: {e}", "danger")
    return render_template('upload.html')

@main.route('/customers')
@login_required
def customers_list():
    customers = Customer.query.order_by(Customer.id.desc()).limit(500).all() # Limit for speed
    return render_template('customers.html', customers=customers)

# --- Standard Auth & Admin Routes (Copy from previous if needed) ---
@main.route('/admin')
@login_required
@admin_required
def admin_panel():
    users = User.query.all()
    logs = ActivityLog.query.order_by(ActivityLog.timestamp.desc()).limit(50).all()
    return render_template('admin.html', models=ai_metrics.get('models',{}), winner=ai_metrics.get('winner','Unknown'), users=users, logs=logs)

@main.route('/comparison')
@login_required
@admin_required
def comparison():
    return render_template('comparison.html', models=ai_metrics.get('models',{}), winner=ai_metrics.get('winner','Unknown'))

@main.route('/reset_db')
@login_required
@admin_required
def reset_db():
    try:
        db.session.query(Customer).delete()
        db.session.commit()
        flash("Database cleared.", "warning")
    except: db.session.rollback()
    return redirect(url_for('main.admin_panel'))

@main.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            email = request.form.get('email')
            role = 'admin' if 'admin' in email.lower() else 'analyst'
            new_user = User(email=email, username=request.form.get('username'), password_hash=generate_password_hash(request.form.get('password')), role=role)
            db.session.add(new_user); db.session.commit(); login_user(new_user)
            return redirect(url_for('main.dashboard'))
        except IntegrityError: db.session.rollback(); flash('Email taken.', 'warning')
    return render_template('auth/register.html')

@main.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form.get('email')).first()
        if user and check_password_hash(user.password_hash, request.form.get('password')):
            login_user(user); return redirect(url_for('main.dashboard'))
        flash('Invalid credentials', 'danger')
    return render_template('auth/login.html')

@main.route('/logout')
@login_required
def logout(): logout_user(); return redirect(url_for('main.login'))