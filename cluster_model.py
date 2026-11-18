CENTROIDS = (
    (3.5, 3.4, 2.3, 2.3),
    (3.6, 3.9, 4.0, 4.7),
    (1.8571428571428568, 2.285714285714286, 2.2142857142857144, 4.0),
)


def predict_cluster(skills):
    return min(
        range(len(CENTROIDS)),
        key=lambda i: sum((a - b) ** 2 for a, b in zip(skills, CENTROIDS[i])),
    )
