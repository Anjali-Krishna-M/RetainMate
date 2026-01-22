
# 🔮 RetainMate - Customer Churn Prediction AI

Hello! 👋 Here is how to set up and run the Churn Prediction project on your computer. 

Follow these steps exactly, one by one.

---

### 🛠️ Step 1: Install Python and Git
Before starting, make sure you have:
1. **Python** installed (Check by typing `python --version` in your terminal).
2. **Git** installed (Check by typing `git --version`).

---

### 🚀 Step 2: Clone the Project
1. Open your **Command Prompt (cmd)** or **Terminal**.
2. Navigate to the folder where you want to save the project (e.g., Desktop):
   ```bash
   cd Desktop

```

3. Clone the repository (Copy this command):
```bash
git clone <PASTE_YOUR_GITHUB_REPO_LINK_HERE>

```


4. Enter the project folder:
```bash
cd RetainMate

```



---

### 📦 Step 3: Create a Virtual Environment

We need a virtual environment to keep our libraries organized so they don't mess up your computer.

**For Windows:**

```bash
python -m venv venv

```

**For Mac/Linux:**

```bash
python3 -m venv venv

```

---

### 🔌 Step 4: Activate the Virtual Environment

You need to turn the environment "On".

**For Windows:**

```bash
venv\Scripts\activate

```

**For Mac/Linux:**

```bash
source venv/bin/activate

```

*(You will know it worked if you see `(venv)` appear at the start of your command line).*

---

### 📥 Step 5: Install Dependencies

Now we install Flask, Machine Learning libraries, and everything else needed.

```bash
pip install -r requirements.txt

```

*(Wait for this to finish installing everything).*

---

### 🧠 Step 6: Train the AI Brain

Before the website can predict anything, we need to generate the machine learning model files (`model.pkl` and `scaler.pkl`).

Run this command:

```bash
python ml_model/train_model.py

```

*(You should see a message saying "Model trained and saved successfully!").*

---

### ▶️ Step 7: Run the Website

Now we start the application!

```bash
python run.py

```

If it works, you will see a message saying:

`Running on http://127.0.0.1:5000`

1. Open your web browser (Chrome/Edge).
2. Go to: **https://www.google.com/url?sa=E&source=gmail&q=http://127.0.0.1:5000**
3. Since this is a fresh database, click **"Get Started"** to create a new account.
4. Log in and start predicting!

---

### ❌ How to Stop

To stop the server, go back to your terminal and press **CTRL + C**.
To exit the virtual environment, type: `deactivate`.

```
