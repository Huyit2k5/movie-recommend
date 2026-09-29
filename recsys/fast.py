"""Bản nhanh (numpy/scipy) của Item-CF và mô hình Hybrid, dùng cho dịch vụ thật.

Kết quả của FastItemCF khớp ItemCF trong models.py (cùng công thức cosine, cùng min_common,
cùng top-k hàng xóm), chỉ khác cách tính: nhân ma trận thưa thay vì vòng lặp Python.
"""
import numpy as np
from scipy import sparse

from .models import Recommender


class FastItemCF(Recommender):
    name = "Item-based CF (fast)"

    def __init__(self, k=30, min_common=5):
        self.k, self.min_common = k, min_common

    def fit(self, ratings, genres):
        super().fit(ratings, genres)
        self.iidx = {i: n for n, i in enumerate(self.all_items)}
        users = {u: n for n, u in enumerate(ratings)}
        rows, cols, vals = [], [], []
        for u, items in ratings.items():
            for i, r in items.items():
                rows.append(users[u]); cols.append(self.iidx[i]); vals.append(r)
        ni = len(self.all_items)
        R = sparse.csr_matrix((vals, (rows, cols)), shape=(len(users), ni), dtype=np.float64)
        B = R.copy()
        B.data[:] = 1
        dot = (R.T @ R).toarray()
        common = (B.T @ B).toarray()
        norm = np.sqrt(np.diag(dot))
        denom = np.outer(norm, norm)
        S = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
        S[common < self.min_common] = 0
        np.fill_diagonal(S, 0)
        S[S < 0] = 0  # rating không âm nên cosine luôn >= 0; giữ cho chắc
        self.S = S.astype(np.float32)
        self.count_vec = np.array([self.item_count.get(i, 0) for i in self.all_items], dtype=np.float32)
        return self

    # -- lõi: nhận thẳng dict {item: rating}, nên dùng được cho user mới ngoài dữ liệu
    def _neighbor_matrix(self, my_ratings):
        idx = [self.iidx[j] for j in my_ratings if j in self.iidx]
        r = np.array([my_ratings[j] for j in my_ratings if j in self.iidx], dtype=np.float32)
        if not idx:
            return None, r, idx
        sub = self.S[:, idx].copy()
        m = len(idx)
        if m > self.k:  # chỉ giữ k hàng xóm giống nhất cho mỗi phim
            thresh = np.partition(sub, m - self.k, axis=1)[:, m - self.k]
            sub[sub < thresh[:, None]] = 0
        return sub, r, idx

    def scores_for(self, my_ratings):
        """(Σ sim·rating, Σ sim) cho mọi phim. Phim đã xem bị gán -inf ở score tổng hợp."""
        sub, r, _ = self._neighbor_matrix(my_ratings)
        n = len(self.all_items)
        if sub is None:
            return np.zeros(n, np.float32), np.zeros(n, np.float32)
        return sub @ r, sub.sum(axis=1)

    def seen_mask(self, my_ratings):
        mask = np.zeros(len(self.all_items), bool)
        for j in my_ratings:
            if j in self.iidx:
                mask[self.iidx[j]] = True
        return mask

    def recommend_for(self, my_ratings, k=10):
        if not my_ratings:
            return [i for i in self.popular][:k]
        num, den = self.scores_for(my_ratings)
        score = np.where(self.seen_mask(my_ratings), -np.inf, num)
        return self._top(score, k)

    def _top(self, score, k):
        # tie-break theo độ phổ biến rồi id, giống Recommender.recommend
        order = np.lexsort((np.arange(len(score)), -self.count_vec, -score))
        return [self.all_items[n] for n in order[:k] if np.isfinite(score[n])]

    def recommend(self, user, k=10):
        return self.recommend_for(self.ratings.get(user, {}), k)

    def predict(self, user, item):
        seen = self.ratings.get(user)
        if not seen or item not in self.iidx:
            return self.fallback_rating(user, item)
        num, den = self.scores_for(seen)
        n = self.iidx[item]
        if den[n] <= 0:
            return self.fallback_rating(user, item)
        return self.clip(float(num[n] / den[n]))


class Hybrid(FastItemCF):
    """Điểm = w_cf·CF + w_pop·Popularity + w_ct·Content, mỗi thành phần chuẩn hoá về [0, 1].

    - CF cho gợi ý cá nhân hoá.
    - Popularity kéo về phim được nhiều người xem (mốc mạnh trong top-K).
    - Content (thể loại) là đường duy nhất cho phim mới chưa có rating nào.
    """
    name = "Hybrid (CF + Pop + Content)"

    def __init__(self, w_cf=1.0, w_pop=0.3, w_ct=0.3, k=30, min_common=5):
        super().__init__(k=k, min_common=min_common)
        self.w_cf, self.w_pop, self.w_ct = w_cf, w_pop, w_ct

    def fit(self, ratings, genres):
        super().fit(ratings, genres)
        G = np.array([genres[i] for i in self.all_items], dtype=np.float32)
        n = np.linalg.norm(G, axis=1, keepdims=True)
        self.G = G                                    # vector thể loại thô
        self.Gn = np.divide(G, n, out=np.zeros_like(G), where=n > 0)
        pop = np.log1p(self.count_vec)
        self.pop_vec = pop / pop.max()
        return self

    def _content(self, my_ratings):
        idx = [self.iidx[j] for j in my_ratings if j in self.iidx]
        if not idx:
            return np.zeros(len(self.all_items), np.float32)
        r = np.array([my_ratings[j] for j in my_ratings if j in self.iidx], np.float32)
        profile = ((r - r.mean())[:, None] * self.G[idx]).sum(axis=0)
        norm = np.linalg.norm(profile)
        if norm == 0:
            return np.zeros(len(self.all_items), np.float32)
        return np.clip(self.Gn @ (profile / norm), 0, None)

    def combined(self, my_ratings):
        num, _ = self.scores_for(my_ratings)
        cf = num / num.max() if num.max() > 0 else num
        return self.w_cf * cf + self.w_pop * self.pop_vec + self.w_ct * self._content(my_ratings)

    def recommend_for(self, my_ratings, k=10):
        if not my_ratings:
            return list(self.popular[:k])
        score = np.where(self.seen_mask(my_ratings), -np.inf, self.combined(my_ratings))
        return self._top(score, k)
