# 1. Setup and Library Import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("Libraries imported successfully.")

# 2. Data Loading and Preparation
try:
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]
except FileNotFoundError:
    print("⚠️ 'spam.csv' not found. Using a dummy dataset instead.")
    data = {
        "label": ["ham", "spam", "ham", "spam", "spam", "ham"],
        "text": [
            "Hey there, how are you?",
            "WINNER! You have won a prize!",
            "See you soon.",
            "URGENT! Claim your gift card!",
            "Free entry in 2 a weekly competition. Text WIN to 80086 now!",
            "I'll call you later."
        ]
    }
    df = pd.DataFrame(data)

# Encode labels (ham=0, spam=1)
df["label"] = df["label"].str.lower().map({"ham": 0, "spam": 1})

print("\n✅ Dataset prepared. First 5 rows:")
print(df.head())

# Show class balance
print("\n📊 Class distribution:")
print(df["label"].value_counts())

# 3. Train/Test Split
X = df["text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
)

print(f"\n🔹 Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# 4. Build Pipeline (TF-IDF + Naive Bayes)
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("nb", MultinomialNB())
])

# 5. Train Model
model.fit(X_train, y_train)
print("\n✅ Model training complete.")

# 6. Model Evaluation
if len(X_test) > 0:
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"])

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", class_report)

    # Confusion Matrix Heatmap
    plt.figure(figsize=(5,4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# 7. Spam Prediction Function (with confidence score)
def predict_spam(text):
    proba = model.predict_proba([text])[0]
    prediction = model.predict([text])[0]
    label = "🚨 SPAM" if prediction == 1 else "✅ HAM (Not Spam)"
    confidence = proba[prediction]
    return f"{label} (confidence: {confidence:.2f})"

# 8. Test with New Messages
print("\n--- Testing with New Samples ---")
test_samples = [
    "Congratulations! You've won a FREE trip to Las Vegas! Click here to claim your prize.",
    "Hey, do you want to grab lunch tomorrow?",
    "URGENT! Your account has been compromised. Please verify your details immediately.",
    "Meeting rescheduled to 3 PM. See you there."
]

for sample in test_samples:
    print(f"'{sample}' -> {predict_spam(sample)}")
