# ============================================================
#  PROJECT: Breast Cancer Prediction (Binary Classification)
#  Dataset : sklearn.datasets.load_breast_cancer
#  Model   : Logistic Regression
#  Author  : YBI Foundation Internship Project
# ============================================================

# ── STEP 1: Libraries Import ────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

# ── STEP 2: Dataset Load ────────────────────────────────────
data = load_breast_cancer()

# DataFrame banana
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target   # 0 = Malignant, 1 = Benign

print("Dataset Shape:", df.shape)
print("\nTarget Distribution:\n", df['target'].value_counts())
print("\nFirst 5 Rows:\n", df.head())

# ── STEP 3: EDA (Exploratory Data Analysis) ─────────────────
print("\nMissing Values:\n", df.isnull().sum().sum())
print("\nBasic Stats:\n", df.describe())

# Target distribution plot
plt.figure(figsize=(5, 4))
sns.countplot(x='target', data=df,
              palette=['tomato', 'steelblue'])
plt.xticks([0, 1], ['Malignant (0)', 'Benign (1)'])
plt.title('Cancer Type Distribution')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('target_distribution.png', dpi=150)
plt.show()

# ── STEP 4: Feature & Target Define ─────────────────────────
X = df.drop('target', axis=1)   # 30 features
y = df['target']                 # label

# ── STEP 5: Train-Test Split ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2529)

print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

# ── STEP 6: Feature Scaling ──────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── STEP 7: Model Training ───────────────────────────────────
model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train)

# ── STEP 8: Prediction ───────────────────────────────────────
y_pred = model.predict(X_test_scaled)

# ── STEP 9: Model Evaluation ─────────────────────────────────
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                             target_names=['Malignant', 'Benign']))

# ── STEP 10: Confusion Matrix & Feature Importance ──────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'],
            ax=axes[0])
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Top 10 Features by Model Coefficient
coef = np.abs(model.coef_[0])
feat_series = pd.Series(coef, index=data.feature_names).sort_values(ascending=False)[:10]
feat_series.plot(kind='barh', ax=axes[1], color='steelblue')
axes[1].invert_yaxis()
axes[1].set_title('Top 10 Important Features', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('cancer_model_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ── STEP 11: Predict on New Sample ───────────────────────────
# Real dataset ka ek sample lekar predict karte hain
sample = X_test.iloc[[0]]
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
probability = model.predict_proba(sample_scaled)

label = "Benign (Non-cancerous)" if prediction[0] == 1 else "Malignant (Cancerous)"
print(f"\n🔬 Sample Prediction: {label}")
print(f"   Malignant Probability : {probability[0][0]*100:.1f}%")
print(f"   Benign Probability    : {probability[0][1]*100:.1f}%")