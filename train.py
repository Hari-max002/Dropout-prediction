import json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from generate_data import generate, FEATURES

df = generate(); df.to_csv("students.csv", index=False)
X, y = df[FEATURES], df.dropped_out
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=7)
models = {
  "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
  "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample", random_state=7),
}
metrics = {}
for name, m in models.items():
    m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]; pred = (p >= .5).astype(int)
    metrics[name] = {"recall": round(recall_score(yte, pred), 3), "precision": round(precision_score(yte, pred), 3),
                     "roc_auc": round(roc_auc_score(yte, p), 3)}
    print(name, metrics[name])
best = max(metrics, key=lambda k: metrics[k]["roc_auc"])
model = models[best]
if best == "Random Forest": imp = dict(zip(FEATURES, model.feature_importances_.round(3).tolist()))
else: imp = dict(zip(FEATURES, np.abs(model[-1].coef_[0]).round(3).tolist()))
joblib.dump({"model": model, "medians": X.median().to_dict()}, "model.joblib")
json.dump({"selected": best, "test_rows": len(yte), "dropout_rate": round(float(y.mean()), 3),
           "models": metrics, "feature_importance": imp, "data": "synthetic"}, open("metrics.json", "w"), indent=1)
print("selected:", best)
