"""Đọc MovieLens 100K và chia train/test theo thời gian."""
import os
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ml-100k")

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def load_ratings(data_dir=DATA_DIR):
    """Trả về list (user, item, rating, timestamp)."""
    rows = []
    with open(os.path.join(data_dir, "u.data"), encoding="latin-1") as f:
        for line in f:
            u, i, r, t = line.split("\t")
            rows.append((int(u), int(i), float(r), int(t)))
    return rows


def load_items(data_dir=DATA_DIR):
    """Trả về (titles[item], genres[item] = list 0/1 độ dài 19)."""
    titles, genres = {}, {}
    with open(os.path.join(data_dir, "u.item"), encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            item = int(parts[0])
            titles[item] = parts[1]
            genres[item] = [int(x) for x in parts[5:24]]
    return titles, genres


def temporal_split(rows, n_test=10):
    """Giữ lại n_test rating cuối (theo timestamp) của mỗi user làm test."""
    by_user = defaultdict(list)
    for row in rows:
        by_user[row[0]].append(row)
    train, test = [], []
    for u, user_rows in by_user.items():
        user_rows.sort(key=lambda x: x[3])
        if len(user_rows) <= n_test:  # quá ít dữ liệu: để hết vào train
            train.extend(user_rows)
            continue
        train.extend(user_rows[:-n_test])
        test.extend(user_rows[-n_test:])
    return train, test


def to_user_dict(rows):
    """ratings[user][item] = rating."""
    ratings = defaultdict(dict)
    for u, i, r, _ in rows:
        ratings[u][i] = r
    return dict(ratings)


def invert(ratings):
    """item_users[item] = {user: rating}."""
    item_users = defaultdict(dict)
    for u, items in ratings.items():
        for i, r in items.items():
            item_users[i][u] = r
    return dict(item_users)
