import csv
from pathlib import Path

from cluster_model import predict_cluster

SKILLS = (
    "skill_programming",
    "skill_data_stats",
    "skill_frontend_design",
    "skill_business_pitch",
)


def assign_groups(students, n_groups=8):
    labeled = [
        {"id": s["id"], "cluster": predict_cluster([s[k] for k in SKILLS])}
        for s in students
    ]
    by_cluster = {}
    for s in labeled:
        by_cluster.setdefault(s["cluster"], []).append(s)
    for members in by_cluster.values():
        members.sort(key=lambda s: s["id"])
    # ponytail: equal-size clusters break ties by cluster number
    order = sorted(by_cluster, key=lambda c: (-len(by_cluster[c]), c))
    for i, s in enumerate(s for c in order for s in by_cluster[c]):
        s["group"] = (i % n_groups) + 1
    return labeled


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    path = root / "class_dataset.csv"
    if not path.exists():
        path = root / "class dataset.csv"
    with path.open(newline="") as f:
        students = [
            {"id": row["netid"], **{k: int(row[k]) for k in SKILLS}}
            for row in csv.DictReader(f)
        ]
    n_groups = 8
    assigned = assign_groups(students, n_groups)
    clusters = sorted({s["cluster"] for s in assigned})
    missing = 0
    for g in range(1, n_groups + 1):
        members = [s for s in assigned if s["group"] == g]
        counts = {c: sum(s["cluster"] == c for s in members) for c in clusters}
        missing += any(counts[c] == 0 for c in clusters)
        parts = " ".join(f"c{c}={counts[c]}" for c in clusters)
        print(f"group {g}: n={len(members)} {parts}")
    print(f"groups_missing_a_cluster={missing}")
