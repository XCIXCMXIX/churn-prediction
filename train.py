import json
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------- 1. Load + clean ----------
df = pd.read_csv("data/Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.drop_duplicates().dropna(subset=["TotalCharges"])
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# ---------- 2. Quick EDA ----------
print("Shape:", df.shape)
print("Churn rate:", round(df["Churn"].mean() * 100, 1), "%")
print(df.groupby("Contract")["Churn"].mean().round(3))

# ---------- 3. Features ----------
NUM = ["tenure", "MonthlyCharges"]
CAT = ["Contract", "InternetService", "PaymentMethod", "TechSupport", "PaperlessBilling"]
X, y = df[NUM + CAT], df["Churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def make(clf):
    pre = ColumnTransformer([("n", StandardScaler(), NUM), ("c", OneHotEncoder(), CAT)])
    return Pipeline([("pre", pre), ("clf", clf)])

models = {
    "Logistic Regression": make(LogisticRegression(max_iter=1000, class_weight="balanced")),
    "Decision Tree": make(DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)),
    "Random Forest": GridSearchCV(
        make(RandomForestClassifier(class_weight="balanced", random_state=42)),
        {"clf__n_estimators": [100, 200], "clf__max_depth": [6, 10]},
        cv=3, scoring="recall"),
}

# ---------- 4. Train + evaluate ----------
metrics = {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    p = m.predict(X_te)
    cv = cross_val_score(m, X_tr, y_tr, cv=5, scoring="f1").mean()
    metrics[name] = {
        "accuracy": round(accuracy_score(y_te, p), 3),
        "precision": round(precision_score(y_te, p), 3),
        "recall": round(recall_score(y_te, p), 3),
        "f1": round(f1_score(y_te, p), 3),
        "cv_f1": round(cv, 3),
    }
    print(name, metrics[name])

# ---------- 5. Export Logistic Regression to JSON ----------
lr = models["Logistic Regression"]
pre, clf = lr.named_steps["pre"], lr.named_steps["clf"]
w, b = clf.coef_[0], float(clf.intercept_[0])
sc, ohe = pre.named_transformers_["n"], pre.named_transformers_["c"]

num = {}
for i, c in enumerate(NUM):          # scaling ko weights mein fold kar diya
    num[c] = float(w[i] / sc.scale_[i])
    b -= float(w[i] * sc.mean_[i] / sc.scale_[i])

cat, k = {}, len(NUM)
for c, cats in zip(CAT, ohe.categories_):
    cat[c] = {}
    for v in cats:
        cat[c][str(v)] = float(w[k]); k += 1

json.dump({"b": b, "num": num, "cat": cat, "metrics": metrics},
          open("model.json", "w"), indent=2)
print("model.json saved")