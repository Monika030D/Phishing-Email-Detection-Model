import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("dataset.csv").dropna()
X_train, X_test, y_train, y_test = train_test_split(
    df["email_text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
)

features = FeatureUnion([
    ("word_tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2), sublinear_tf=True)),
    ("char_tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3,5), sublinear_tf=True))
])

model = Pipeline([
    ("features", features),
    ("classifier", LogisticRegression(max_iter=2000))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("\n=== Phishing Email Detection Model ===")
print(f"Accuracy: {accuracy:.2%}\n")
print(classification_report(y_test, pred, zero_division=0))

cm = confusion_matrix(y_test, pred, labels=["Phishing", "Safe"])
ConfusionMatrixDisplay(cm, display_labels=["Phishing", "Safe"]).plot()
plt.title("Phishing Email Detection - Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=160)
plt.close()

with open("model/phishing_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("results/accuracy.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy:.2%}\n")

print("Model saved: model/phishing_model.pkl")
print("Confusion matrix saved: results/confusion_matrix.png")
