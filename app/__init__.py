from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

# Initialize plugins
db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    app = Flask(__name__)
    
    # SECRET KEY (Used for security - keep this random)
    app.config['SECRET_KEY'] = 'nasma-secret-key-123'
    
    # DATABASE CONFIGURATION (Creates a file named churn.db)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///churn.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Initialize extensions with the app
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'main.login' # Redirect here if user isn't logged in

    # Import routes (We will create this file next)
    from .routes import main
    app.register_blueprint(main)
    
    # Import models so the DB knows they exist
    from .models import User, Customer

    # Create the database automatically
    with app.app_context():
        db.create_all()

    return app