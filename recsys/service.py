"""Lớp dịch vụ: nạp model đã fit và trả kết quả dạng dict, dùng chung cho API và Streamlit."""
import os
import pickle

import numpy as np

from .data import load_items, load_ratings, to_user_dict
from .fast import FastItemCF

ARTIFACT = os.path.join(os.path.dirname(__file__), "..", "data", "model.pkl")
# Tham số chọn bằng tune.py trên validation
PARAMS = dict(k=50, min_common=3)


class BadRequest(ValueError):
    """Đầu vào không hợp lệ (id phim lạ, rating ngoài 1-5, ...)."""


class RecService:
    def __init__(self, model, titles):
        self.model, self.titles = model, titles
        self._lower = {i: t.lower() for i, t in titles.items()}

    # ---- tạo / nạp
    @classmethod
    def build(cls, path=ARTIFACT):
        titles, genres = load_items()
        model = FastItemCF(**PARAMS).fit(to_user_dict(load_ratings()), genres)
        with open(path, "wb") as f:
            pickle.dump({"model": model, "titles": titles}, f)
        return cls(model, titles)

    @classmethod
    def load(cls, path=ARTIFACT):
        if not os.path.exists(path):
            return cls.build(path)
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["model"], d["titles"])

    # ---- tiện ích
    def movie(self, item):
        return {"id": item, "title": self.titles[item], "num_ratings": self.model.item_count.get(item, 0)}

    def search(self, q, limit=20):
        q = q.lower().strip()
        hits = [i for i, t in self._lower.items() if q in t]
        hits.sort(key=lambda i: -self.model.item_count.get(i, 0))
        return [self.movie(i) for i in hits[:limit]]

    def _validate(self, ratings):
        for i, r in ratings.items():
            if i not in self.titles:
                raise BadRequest(f"Không có phim id={i}")
            if not 1 <= r <= 5:
                raise BadRequest(f"Rating phải trong khoảng 1-5 (phim {i}: {r})")

    # ---- gợi ý
    def recommend(self, ratings, k=10, explain=True):
        """Gợi ý cho một tập rating {item: rating}. Rỗng -> phim phổ biến nhất (cold-start)."""
        self._validate(ratings)
        m = self.model
        items = m.recommend_for(ratings, k)
        out = []
        for i in items:
            entry = self.movie(i)
            if explain and ratings:
                entry["because"] = self._because(ratings, i)
            out.append(entry)
        return {"strategy": "item_cf" if ratings else "popularity", "recommendations": out}

    def _because(self, ratings, item, n=3):
        m = self.model
        row = m.S[m.iidx[item]]
        contrib = [(row[m.iidx[j]] * r, j, row[m.iidx[j]]) for j, r in ratings.items()
                   if j in m.iidx and row[m.iidx[j]] > 0]
        contrib.sort(reverse=True)
        return [{"id": j, "title": self.titles[j], "similarity": round(float(s), 3), "your_rating": ratings[j]}
                for _, j, s in contrib[:n]]

    def recommend_user(self, user_id, k=10):
        seen = self.model.ratings.get(user_id)
        if seen is None:
            raise BadRequest(f"Không có user id={user_id}")
        return self.recommend(dict(seen), k)

    def similar(self, item, k=10):
        if item not in self.titles:
            raise BadRequest(f"Không có phim id={item}")
        m = self.model
        row = m.S[m.iidx[item]]
        order = np.argsort(-row)[:k]
        return [dict(self.movie(m.all_items[n]), similarity=round(float(row[n]), 3))
                for n in order if row[n] > 0]
