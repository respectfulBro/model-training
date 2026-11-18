from pathlib import Path

import matplotlib.pyplot as plt
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

ks = range(1, 7)
inertias = []
for k in ks:
    model = KMeans(n_clusters=k, n_init="auto", random_state=0)
    model.fit(X)
    inertias.append(model.inertia_)
    print(f"students={len(df)} k={k} inertia={model.inertia_:.2f}")

plt.plot(list(ks), inertias, marker="o")
plt.xlabel("k")
plt.ylabel("inertia")
plt.savefig(ROOT / "elbow.png")
