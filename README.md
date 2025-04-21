# FakeNewsDetection
# 📰 Fake News Detection using Machine Learning

This project is a Fake News Detection system built using **Logistic Regression**, **TF-IDF Vectorization**

## 🚀 Project Overview

Fake news is a growing concern in the modern information age. This project uses Natural Language Processing (NLP) and Machine Learning techniques to classify news articles as **Fake** or **Real**.

## 📂 Dataset

We use two publicly available datasets:

- `Fake.csv` – contains fake news articles
- `True.csv` – contains real news articles

Each article has:
- Title
- Text content
- Label (0 = Fake, 1 = Real)

## 🧠 Model Pipeline

1. **Data Loading & Merging**
2. **Preprocessing**
   - Combining title & text
   - Lowercasing, removing punctuation & links
3. **Feature Extraction**
   - Using `TfidfVectorizer` (max_features=5000)
4. **Model Training**
   - Logistic Regression (primary model)
   - Also experimented with SVM, Naive Bayes, Decision Tree, XGBoost
5. **Evaluation**
   - Accuracy score
   - Classification report (Precision, Recall, F1-score)


## ✅ Accuracy

The Logistic Regression model achieves strong accuracy on the test set:


