
# RetainMate: Intelligent Customer Churn Prediction System

**Project Type:** Enterprise Web Application | **Domain:** Machine Learning & Data Analytics

![Status](https://img.shields.io/badge/Status-Production%20Ready-success) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Table of Contents
1. [Project Abstract](#1-project-abstract)
2. [Key Features & Tech Stack](#2-key-features--tech-stack)
3. [Step-by-Step Installation Guide](#3-step-by-step-installation-guide)
4. [How to Run the Application](#4-how-to-run-the-application)
5. [User Guide (How to Demo)](#5-user-guide-how-to-demo)
6. [File Structure & Code Explanation](#6-file-structure--code-explanation)
7. [Troubleshooting Common Errors](#7-troubleshooting-common-errors)
8. [Viva Voce Cheat Sheet](#8-viva-voce-cheat-sheet)

---

## 1. Project Abstract
**RetainMate** is an advanced analytics platform designed to assist telecom service providers in reducing customer attrition (churn). By leveraging machine learning algorithms, the system analyzes historical customer data to predict the likelihood of churn for individual users.

Beyond simple prediction, the system implements **Explainable AI (XAI)** techniques to interpret model decisions, providing actionable insights (e.g., "High Risk due to Month-to-Month Contract"). This transparency allows stakeholders to implement targeted retention strategies. The platform features a role-based administration dashboard, batch data processing capabilities, and automated model performance evaluation.

---

## 2. Key Features & Tech Stack

### **Key Features**
* **Predictive Analytics Engine:** Utilizes Random Forest and Logistic Regression algorithms to classify customers as "High Risk" or "Safe" with high accuracy.
* **Explainable AI (XAI) Module:** Deconstructs prediction logic to identify key risk factors (Contract Type, Tenure, Pricing) for each customer.
* **Batch Processing System:** Supports bulk CSV uploads for analyzing large datasets (10,000+ records) simultaneously.
* **Role-Based Access Control (RBAC):** Secure authentication system separating 'Analyst' (Read-Only) and 'Admin' (Manage/Delete) privileges.
* **Automated Model Evaluation:** Includes a comparison module that trains multiple algorithms and selects the optimal model based on F1-Score and Accuracy.
* **Interactive Dashboard:** A high-contrast, data-centric UI designed for operational environments.

### **Technology Stack**
* **Backend Framework:** Python 3.10+, Flask (Microframework).
* **Database:** SQLite with SQLAlchemy ORM.
* **Machine Learning:** Scikit-Learn, Pandas, NumPy.
* **Visualization:** Chart.js (Frontend), Matplotlib/Seaborn (Backend EDA).
* **Frontend:** HTML5, CSS3 (Custom Design System), Bootstrap 5, Jinja2 Templating.

---

## 3. Step-by-Step Installation Guide

**Prerequisite:** Ensure [Python](https://www.python.org/downloads/) and [VS Code](https://code.visualstudio.com/) are installed on the local machine. **Crucial:** During Python installation, ensure the "Add Python to PATH" option is checked.

### **Step 1: Clone the Repository**
Open the terminal or command prompt and execute the following commands to download the project code.

```bash
# Clone the repository (Replace URL with your specific Git link)
git clone [https://github.com/YOUR_USERNAME/RetainMate.git](https://github.com/YOUR_USERNAME/RetainMate.git)

# Navigate into the project directory
cd RetainMate

```

### **Step 2: Environment Configuration**

It is recommended to use a virtual environment to isolate project dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

```

*(Upon success, you will see `(venv)` appear at the start of the command line).*

### **Step 3: Dependency Installation**

Install the required Python libraries listed in `requirements.txt`.

```bash
pip install flask flask-sqlalchemy flask-login pandas scikit-learn matplotlib seaborn

```

---

## 4. How to Run the Application

**Note:** Perform these steps every time the application needs to be launched.

### **Phase A: Model Training Pipeline**

The machine learning model must be trained on the dataset before the application can perform predictions. This script handles data preprocessing, encoding, model training, and serialization.

**Command:**

```bash
python ml_model/train_model.py

```

*Expected Output:* `✅ PIPELINE COMPLETE. READY FOR WEB APP.`

### **Phase B: Launching the Web Server**

Start the Flask application server to access the user interface.

**Command:**

```bash
python run.py

```

*Output:* The server will start at `http://127.0.0.1:5000/`.

* **To Access:** Hold `Ctrl` and click the link, or open a web browser and visit `http://127.0.0.1:5000`.

---

## 5. User Guide (How to Demo)

### **1. Administrative Access (Backdoor)**

The system uses email-pattern recognition for role assignment.

* Navigate to **Register**.
* Register with an email containing the substring `admin` (e.g., `admin@retainmate.com`).
* **Result:** The system grants **Admin Privileges**, unlocking the "Admin Console" and "Model Comparison" features.

### **2. Single Customer Analysis**

* Navigate to **New Analysis**.
* Input customer data. For a **High Risk** demonstration, use:
* **Contract:** Month-to-month
* **Tenure:** < 5 Months
* **Monthly Charges:** > $80
* **Internet Service:** Fiber Optic


* Click **Run AI Analysis**. Show the "Risk Factor Analysis" table explaining *why* the risk is high.

### **3. Batch Data Processing**

* Navigate to **Batch Upload**.
* Upload the dataset file (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).
* The system will process all rows and populate the database view.

### **4. Admin Features**

* Click **Admin Panel**.
* Show the **Model Comparison Table** (Comparing Logistic Regression vs Random Forest).
* Show the **Activity Logs** (Tracking user actions for security auditing).

---

## 6. File Structure & Code Explanation

This section maps the project architecture for technical review.

```text
RetainMate/
│
├── app/                      # 🌐 APPLICATION CORE
│   ├── __init__.py           # Application Factory: Initializes App, Database, & Auth
│   ├── routes.py             # Controller: Manages HTTP requests and business logic
│   ├── models.py             # Data Layer: Defines User, Customer, & Log schemas
│   ├── templates/            # Presentation Layer: HTML files (Frontend)
│   │   ├── dashboard.html    # Main Analytics Dashboard
│   │   ├── predict.html      # Prediction Input Form
│   │   └── admin.html        # Secure Admin Interface
│   └── static/               # Static Assets
│       └── css/style.css     # Custom Dark Mode Styling
│
├── ml_model/                 # 🧠 MACHINE LEARNING CORE
│   ├── train_model.py        # ETL Pipeline: Preprocessing, Training, Evaluation
│   ├── model.pkl             # Serialized Model Artifact
│   └── metrics.json          # Performance Metrics Store
│
├── instance/                 # 🗄️ DATABASE STORAGE
│   └── churn.db              # SQLite Database File
│
└── run.py                    # 🔑 ENTRY POINT: Starts the Flask Server

```

---

## 7. Troubleshooting Common Errors

| Error Message | Probable Cause | Solution |
| --- | --- | --- |
| **"ModuleNotFoundError"** | Dependencies not installed. | Run `pip install flask pandas scikit-learn` in the terminal. |
| **"IntegrityError"** | Email already registered. | Use a different email address or reset the database via Admin Panel. |
| **"AI Model Not Found"** | Training script not run. | Execute `python ml_model/train_model.py` to generate the model files. |
| **"Python is not recognized"** | PATH variable missing. | Reinstall Python and ensure "Add Python to PATH" is checked. |

---

## 8. Viva Voce Cheat Sheet

### **Q1: Which algorithm was selected and why?**

> **Answer:** "The system employs a **Random Forest Classifier**. While Logistic Regression was tested, Random Forest provided superior performance in handling non-linear relationships and interactions between features (e.g., the relationship between Tenure and Contract Type). It also provides `feature_importances_`, which is essential for the Explainable AI module."

### **Q2: How is data imbalance handled?**

> **Answer:** "Churn datasets typically exhibit class imbalance (fewer churners than non-churners). To mitigate bias, the model uses the `class_weight='balanced'` parameter, which adjusts weights inversely proportional to class frequencies, ensuring the model does not overlook the minority class."

### **Q3: Explain the 'Explainable AI' feature.**

> **Answer:** "The system does not treat the model as a 'black box.' By extracting feature importance scores and analyzing specific input vectors (e.g., 'Month-to-Month Contract'), the application maps technical risk probabilities to human-readable business insights using a deterministic logic layer in `routes.py`."

### **Q4: How is the application secured?**

> **Answer:** "Security is handled via **Flask-Login** for session management. Passwords are hashed using **Scrypt** (via Werkzeug security) before storage to prevent plaintext leaks. Additionally, **Role-Based Access Control (RBAC)** is implemented using Python decorators (`@admin_required`) to restrict sensitive routes."

---

```

```