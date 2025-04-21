import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

fake_data = pd.read_csv("Dataset/Fake.csv")
real_data = pd.read_csv("Dataset/True.csv")

fake_data["class"] = 0
real_data["class"] = 1

"""Combine datasets"""

df = pd.concat([fake_data,real_data], ignore_index=True)
df = df[['title', 'text', 'class']]

# Combine title and text
df['content'] = df['title'] + " " + df['text']

df = df.sample(frac =1)

df.head()

"""Cleaning the text"""

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned'] = df['content'].apply(clean_text)

"""Vectorization"""

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned'])
y = df['class']

"""Train-test split"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Training Logistic Classifier"""

from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

"""Predict and evaluate"""

y_pred_lr = lr_model.predict(X_test)
print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

"""Training DecisionTree"""

from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

"""Predict and evaluate"""

y_pred_dt = dt_model.predict(X_test)
print("Decision Tree")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

"""Confusion matrix"""

cm = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake','Real'], yticklabels=['Fake','Real'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

"""Test with input"""

def predict_news(news):
    cleaned = clean_text(news)
    vectorized = vectorizer.transform([cleaned])
    pred = lr_model.predict(vectorized)[0]
    return "Fake" if pred == 0 else "Real"

news_input = "President Donald Trump called on the U.S. Postal Service on Friday to charge “much more” to ship packages for Amazon (AMZN.O), picking another fight with an online retail giant he has criticized in the past"
print("Prediction:", predict_news(news_input))
