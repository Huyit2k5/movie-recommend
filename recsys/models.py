"""Các mô hình gợi ý, đi từ đơn giản đến phức tạp.

Mọi mô hình có chung giao diện:
    fit(ratings, genres)       ratings[user][item] = rating, genres[item] = vector 0/1
    predict(user, item)        dự đoán rating (dùng cho RMSE/MAE)
    recommend(user, k)         top-K item user chưa xem (dùng cho Precision/Recall/NDCG)
"""
import math
import random
from collections import defaultdict

import numpy as np

from .data import invert


class Recommender:
    name = "base"

    def fit(self, ratings, genres):
        self.ratings = ratings
        self.item_users = invert(ratings)
        self.all_items = sorted(genres)
        n = sum(len(v) for v in ratings.values())
        self.global_mean = sum(r for v in ratings.values() for r in v.values()) / n
        self.user_mean = {u: sum(v.values()) / len(v) for u, v in ratings.items()}
        # Trung bình có "giảm chấn" (damped mean): item ít rating bị kéo về global mean,
        # tránh việc phim chỉ có 1 rating 5 sao đứng đầu bảng.
        damping = 5
        self.item_mean = {
            i: (sum(us.values()) + damping * self.global_mean) / (len(us) + damping)
            for i, us in self.item_users.items()
        }
        self.item_count = {i: len(us) for i, us in self.item_users.items()}
        self.popular = sorted(self.item_users, key=lambda i: -self.item_count[i])
        return self

    def fallback_rating(self, user, item):
        if item in self.item_mean:
            return self.item_mean[item]
        if user in self.user_mean:
            return self.user_mean[user]
        return self.global_mean

    def clip(self, r):
        return min(5.0, max(1.0, r))

    def predict(self, user, item):
        return self.fallback_rating(user, item)

    def score(self, user, candidates):
        """Trả về dict {item: điểm} để xếp hạng. Mặc định: độ phổ biến."""
        return {i: self.item_count.get(i, 0) for i in candidates}

    def recommend(self, user, k=10):
        seen = self.ratings.get(user, {})
        if not seen:  # cold-start user: fallback về popularity
            return [i for i in self.popular if i not in seen][:k]
        candidates = [i for i in self.all_items if i not in seen]  # không gợi ý lại
        scores = self.score(user, candidates)
        # tie-break bằng độ phổ biến rồi item id để kết quả ổn định
        candidates.sort(key=lambda i: (-scores[i], -self.item_count.get(i, 0), i))
        return candidates[:k]


# ---------------------------------------------------------------- Bước 1
class Popularity(Recommender):
    """Gợi ý phim có nhiều rating nhất; dự đoán rating = damped mean của item."""
    name = "Popularity"


# ---------------------------------------------------------------- Bước 2
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class ContentBased(Recommender):
    """Dựa trên thể loại phim.

    - Hồ sơ user = tổng vector thể loại các phim đã xem, trọng số (rating - mean của user).
      Phim thích hơn mức trung bình kéo hồ sơ về phía thể loại đó, phim ghét đẩy ra xa.
    - Xếp hạng: cosine(hồ sơ user, vector thể loại phim).
    - Dự đoán rating: trung bình có trọng số rating của user cho k phim giống nhất về thể loại.
    """
    name = "Content-based"

    def __init__(self, k=30):
        self.k = k

    def fit(self, ratings, genres):
        super().fit(ratings, genres)
        self.genres = genres
        self.profile = {}
        for u, items in ratings.items():
            mu = self.user_mean[u]
            p = [0.0] * len(next(iter(genres.values())))
            for i, r in items.items():
                for g, flag in enumerate(genres[i]):
                    p[g] += (r - mu) * flag
            self.profile[u] = p
        return self

    def predict(self, user, item):
        seen = self.ratings.get(user)
        if not seen or item not in self.genres:
            return self.fallback_rating(user, item)
        sims = sorted(((cosine(self.genres[item], self.genres[j]), r) for j, r in seen.items()),
                      reverse=True)[: self.k]
        den = sum(s for s, _ in sims)
        if den == 0:
            return self.fallback_rating(user, item)
        return self.clip(sum(s * r for s, r in sims) / den)

    def score(self, user, candidates):
        p = self.profile[user]
        return {i: cosine(p, self.genres[i]) for i in candidates}


# ---------------------------------------------------------------- Bước 3
class ItemCF(Recommender):
    """Item-based collaborative filtering với cosine similarity (Python thuần).

    sim(i, j) = sum_u r_ui * r_uj / (||r_i|| * ||r_j||)
      - tử số chỉ tính trên user rate cả hai phim; không có user chung -> sim = 0
      - mẫu số dùng toàn bộ rating của mỗi phim, nên cặp ít user chung tự bị phạt
      - bỏ cặp có ít hơn min_common user chung (tránh similarity = 1 giả)
    """
    name = "Item-based CF"

    def __init__(self, k=30, min_common=5, mean_center=False):
        self.k = k
        self.min_common = min_common
        self.mean_center = mean_center
        if mean_center:
            self.name = "Item-based CF (adjusted cosine)"

    def fit(self, ratings, genres):
        super().fit(ratings, genres)
        # adjusted cosine: trừ mean của user trước khi tính, loại bỏ việc user "dễ tính"/"khó tính"
        val = (lambda u, r: r - self.user_mean[u]) if self.mean_center else (lambda u, r: r)

        norm = {i: math.sqrt(sum(val(u, r) ** 2 for u, r in us.items()))
                for i, us in self.item_users.items()}
        dot = defaultdict(float)
        common = defaultdict(int)
        # duyệt theo user: mỗi user đóng góp vào mọi cặp phim họ đã rate
        for u, items in ratings.items():
            vals = sorted((i, val(u, r)) for i, r in items.items())
            for a in range(len(vals)):
                i, ri = vals[a]
                for b in range(a + 1, len(vals)):
                    j, rj = vals[b]
                    dot[i, j] += ri * rj
                    common[i, j] += 1

        self.sim = defaultdict(dict)
        for (i, j), d in dot.items():
            if common[i, j] < self.min_common or not norm[i] or not norm[j]:
                continue
            s = d / (norm[i] * norm[j])
            self.sim[i][j] = s
            self.sim[j][i] = s
        return self

    def neighbors(self, user, item):
        """k phim user đã rate giống `item` nhất (chỉ lấy similarity dương)."""
        sims = self.sim.get(item, {})
        seen = self.ratings[user]
        nb = [(sims[j], r) for j, r in seen.items() if sims.get(j, 0) > 0]
        nb.sort(reverse=True)
        return nb[: self.k]

    def predict(self, user, item):
        if user not in self.ratings or item not in self.sim:  # cold-start
            return self.fallback_rating(user, item)
        nb = self.neighbors(user, item)
        den = sum(s for s, _ in nb)
        if den == 0:  # mẫu số bằng 0 -> điểm trung bình
            return self.fallback_rating(user, item)
        if self.mean_center:
            mu = self.user_mean[user]
            return self.clip(mu + sum(s * (r - mu) for s, r in nb) / den)
        return self.clip(sum(s * r for s, r in nb) / den)

    def score(self, user, candidates):
        # Xếp hạng bằng tổng sim * rating (không chia mẫu số): ưu tiên phim giống
        # NHIỀU phim user thích. Nếu chia mẫu số, phim chỉ có 1 hàng xóm rating 5
        # sẽ được điểm 5 tuyệt đối và đứng đầu -> top-K rất tệ.
        out = {}
        for i in candidates:
            nb = self.neighbors(user, i) if i in self.sim else []
            out[i] = sum(s * r for s, r in nb)
        return out


# ---------------------------------------------------------------- Bước 4
class MatrixFactorization(Recommender):
    """Biased MF học bằng SGD:  r_ui ≈ mu + b_u + b_i + p_u · q_i."""
    name = "Matrix Factorization (SGD)"

    def __init__(self, factors=50, epochs=30, lr=0.01, reg=0.05, seed=42):
        self.factors, self.epochs, self.lr, self.reg, self.seed = factors, epochs, lr, reg, seed

    def fit(self, ratings, genres):
        super().fit(ratings, genres)
        rng = np.random.default_rng(self.seed)
        self.uidx = {u: n for n, u in enumerate(ratings)}
        self.iidx = {i: n for n, i in enumerate(self.item_users)}
        nu, ni, f = len(self.uidx), len(self.iidx), self.factors
        self.P = rng.normal(0, 0.1, (nu, f))
        self.Q = rng.normal(0, 0.1, (ni, f))
        self.bu = np.zeros(nu)
        self.bi = np.zeros(ni)
        mu = self.global_mean

        data = [(self.uidx[u], self.iidx[i], r) for u, its in ratings.items() for i, r in its.items()]
        random.seed(self.seed)
        lr, reg = self.lr, self.reg
        for _ in range(self.epochs):
            random.shuffle(data)
            for u, i, r in data:
                pu, qi = self.P[u], self.Q[i]
                err = r - (mu + self.bu[u] + self.bi[i] + pu @ qi)
                self.bu[u] += lr * (err - reg * self.bu[u])
                self.bi[i] += lr * (err - reg * self.bi[i])
                self.P[u] = pu + lr * (err * qi - reg * pu)
                self.Q[i] = qi + lr * (err * pu - reg * qi)
        return self

    def _raw(self, user, item):
        u, i = self.uidx.get(user), self.iidx.get(item)
        r = self.global_mean
        if u is not None:
            r += self.bu[u]
        if i is not None:
            r += self.bi[i]
        if u is not None and i is not None:
            r += self.P[u] @ self.Q[i]
        return r

    def predict(self, user, item):
        return self.clip(self._raw(user, item))

    def score(self, user, candidates):
        u = self.uidx[user]
        known = [i for i in candidates if i in self.iidx]
        vals = self.bi[[self.iidx[i] for i in known]] + self.Q[[self.iidx[i] for i in known]] @ self.P[u]
        out = dict.fromkeys(candidates, -1e9)  # item chưa có rating nào: xếp cuối
        out.update(zip(known, vals.tolist()))
        return out
