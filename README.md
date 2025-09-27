# NLP Emotion Analysis
Analysing Emotions using NLP
# NLP Emotion Analyzer

A web-based application that predicts the **emotions of a given text** using Natural Language Processing (NLP) techniques. This app can detect emotions like **sadness, anger, joy, love, fear, and surprise** from user input text. The project is built using **Python, Scikit-learn, NLTK, and Streamlit**, and can be deployed online using **Render**.

---

## Table of Contents
- [Features](#features)
- [Demo](#demo)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Deployment](#deployment)
- [Links](#links)
- [License](#license)

---

## Features
- Predicts **emotion of a sentence**: sadness, anger, love, surprise, fear, joy.
- Shows **confidence scores** for each emotion.
- Interactive **Streamlit web interface**.
- Preprocessing includes lowercasing, removing punctuation and numbers, stopwords removal.
- Supports both **Count Vectorizer** and **TF-IDF Vectorizer** models.

---

## Demo
Access the live demo once deployed on Render: [🔗 Live App on Render](https://your-app-name.onrender.com)

**Example:**

| Input Text                       | Predicted Emotion | Confidence |
|---------------------------------|-----------------|------------|
| I am feeling so happy today!     | joy             | 95%        |
| I feel so sad and hopeless       | sadness         | 91%        |

![App Screenshot](app_inerface.png)


## Folder Structure

NLP-Emotion-Analysis/
│── .gitignore
│── README.md
│── requirements.txt
│── Procfile # For Render deployment
│── app.py # Streamlit app
│
├── models/ # Trained model + vectorizer + label map
│ ├── emotion_model.pkl
│ ├── vectorizer.pkl
│ └── label_map.json
│
├── src/ # Source code
│ ├── init.py
│ ├── preprocessing.py # Text cleaning functions
│ ├── train.py # Training + saving model
│ ├── predict.py # Load model + make predictions
│ └── utils.py
│
├── notebooks/ # Jupyter notebooks for experimentation
│ └── emotion_training.ipynb
│
└── data/ # Dataset
└── train.txt




---

## Installation

1. Clone the repository:


git clone https://github.com/kamal1128/NLP-Emotion-Analysis.git
cd NLP-Emotion-Analysis
Create and activate a virtual environment:



python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
Install required packages:



pip install -r requirements.txt
Download NLTK data (if not included):



import nltk
nltk.download('punkt')
nltk.download('stopwords')

Run the Streamlit app:



streamlit run app.py
Enter a sentence in the text area.

Click Predict.

View predicted emotion and confidence scores for all classes.

Model Details
Models used: Logistic Regression (best) and SVM.

Vectorization: CountVectorizer (best) or TF-IDF Vectorizer.

Saved artifacts:

emotion_model.pkl → trained model

vectorizer.pkl → CountVectorizer

label_map.json → mapping from numeric labels to emotion names

Label Mapping Example:

{
  "0": "sadness",
  "1": "anger",
  "2": "love",
  "3": "surprise",
  "4": "fear",
  "5": "joy"
}
Deployment on Render
Ensure requirements.txt and Procfile exist in the root directory.

Procfile content:

web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
Push your repository to GitHub.

Sign up on Render.

Create a new Web Service and connect your GitHub repo.

Render will install dependencies, build your app, and assign a public URL.

Access your deployed app online via the Render URL.

Links
GitHub Repository: https://github.com/kamal1128/NLP-Emotion-Analysis

LinkedIn: www.linkedin.com/in/sai-kamal-kandukuri-404288305

Live App on Render: https://nlp-emotion-analysis.onrender.com/

License
This project is licensed under the MIT License.
Feel free to use, modify, and share.


