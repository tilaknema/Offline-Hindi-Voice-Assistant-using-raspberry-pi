# ✅ Corrected Training Code (Hindi Intent Classification)
# Save as: train_intent.py

import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

# -----------------------------
# Step 0: Clean text
# -----------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\u0900-\u097Fa-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# Step 1: Load intent data
# -----------------------------
texts = []
labels = []

with open("intent_data_hi1.txt", "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue

        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            print(f"⚠️ Skipping invalid line: {line}")
            continue

        sentence, intent = parts
        texts.append(clean_text(sentence))
        labels.append(intent)

print(f"📘 Total samples: {len(texts)}")
print(f"📘 Total intents: {len(set(labels))}")

# -----------------------------
# ✅ Step 2: NEW Vectorizer (char-level for Hindi)
# -----------------------------
vectorizer = TfidfVectorizer(
    analyzer="char_wb",     # 🔥 character-level features
    ngram_range=(3,5),
    min_df=2
)

X = vectorizer.fit_transform(texts)

# -----------------------------
# Step 3: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)

# -----------------------------
# Step 4: Train Model
# -----------------------------
model = LinearSVC(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Step 5: Evaluation
# -----------------------------
test_acc = model.score(X_test, y_test)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, digits=3, zero_division=0))

# -----------------------------
# Step 6: Cross Validation
# -----------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, labels, cv=skf, scoring="accuracy")
print(f"\n✅ CV Accuracy (mean ± std): {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# -----------------------------
# Step 7: Save Model
# -----------------------------
joblib.dump(model, "intent_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\n💾 Model and vectorizer saved successfully")
