# -*- coding: utf-8 -*-
"""
Created on Wed May 20 19:48:50 2026

@author: MAKSIM
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib



#TODO: не забыить про сохранение модели!! через joblib

FILE_PATH = "dataset.csv"

df = pd.read_csv(FILE_PATH)

print(df.head())
print(df["humor"].unique())

X = df["text"].astype(str)

y = df["humor"].astype(int)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# #----------------TRAINING------------------------------------------------------

# # Prepearing TF-IDF + Kernel SVM pipline:
# model = Pipeline([
#     (
#         "vectorizer",
#         TfidfVectorizer(
#             lowercase=True,
#             stop_words = "english",
#             ngram_range = (1,2),
#             max_features = 10000,
#             sublinear_tf = True
#         )
#     ),
#     (
#         "method",
#         SVC(
#             kernel="rbf",   
#             C=1.0,
#             gamma="scale"
#         )
#     )
# ])


# # Training the model (applying the pipline):
# model.fit(X_train, y_train)

# # Saving the model
# joblib.dump(
#     model,
#     "kernel_svm_humor.pkl"
# )

# #-----------------------------------------------------------------------------

df_new = pd.read_excel('kaggle_dataset.xlsx') # for testing
X_test_new = df_new['text'].astype(str)
Y_test_new = df_new['label']





model = joblib.load("kernel_svm_humor.pkl")


# Predticions:
y_pred = model.predict(X_test_new)


# Evaluating:
print("\nAccuracy:")
print(accuracy_score(Y_test_new, y_pred))

print("\nClassification Report:")
print(classification_report(Y_test_new, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test_new, y_pred))



# # Testing on other arbitary texts:
# examples = [
#     "I told my computer I needed a break, and it froze.",
#     "The weather is warm today."
# ]

# predictions = model.predict(examples)

# print("\nCustom predictions:")

# for text, pred in zip(examples, predictions):
#     label = "HUMOR" if pred == 1 else "NOT HUMOR"
#     print(f"{text} --> {label}")