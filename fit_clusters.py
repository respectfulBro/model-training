from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent

SKILLS = [
    "skill_programming",
    "skill_data_stats",
    "skill_frontend_design",
    "skill_business_pitch",
]

csv_path = ROOT / "class_dataset.csv"
if not csv_path.exists():
    csv_path = ROOT / "class dataset.csv"
df = pd.read_csv(csv_path)
X = df[SKILLS]

model = KMeans(n_clusters=3, n_init="auto", random_state=0)
labels = list(model.fit_predict(X))
df["cluster"] = labels

for i, centroid in enumerate(model.cluster_centers_):
    n = labels.count(i)
    vals = ", ".join(f"{v:.2f}" for v in centroid)
    print(f"cluster {i}: n={n} centroid=[{vals}]")

print(df[["netid", "cluster"]].to_string(index=False))
