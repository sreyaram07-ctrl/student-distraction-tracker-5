import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rng = np.random.default_rng(42)
n = 500
df = pd.DataFrame({
    "study_hours": rng.uniform(0, 10, n).round(1),
    "social_media_hours": rng.uniform(0, 8, n).round(1),
    "sleep_hours": rng.uniform(4, 10, n).round(1),
    "breaks": rng.integers(0, 10, n),
})
score = (df.study_hours*8 + df.sleep_hours*5
         - df.social_media_hours*7 - abs(df.breaks-4)*2
         + rng.normal(0, 5, n))
df["productivity"] = pd.cut(score, [-100, 30, 55, 200],
                            labels=["Low", "Medium", "High"])
df.to_csv("data.csv", index=False)

X = df.drop(columns="productivity")
y = df["productivity"].astype(str)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(Xtr, ytr)
print("Accuracy:", accuracy_score(yte, model.predict(Xte)))
print(dict(zip(X.columns, model.feature_importances_.round(2))))
joblib.dump(model, "model.pkl")