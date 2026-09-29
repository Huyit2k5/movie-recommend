"""Metric đánh giá: RMSE/MAE cho dự đoán rating, Precision/Recall/NDCG@K cho top-K."""
import math
from collections import defaultdict


def rmse_mae(model, test):
    se = ae = 0.0
    for u, i, r, _ in test:
        e = model.predict(u, i) - r
        se += e * e
        ae += abs(e)
    return math.sqrt(se / len(test)), ae / len(test)


def ranking_metrics(model, test, k=10, threshold=4):
    """Item "liên quan" = phim trong test được user chấm >= threshold.

    Chỉ tính trên user có ít nhất 1 item liên quan trong test.
    Coverage = tỉ lệ phim trong kho từng xuất hiện trong top-K của ai đó (đo popularity bias).
    """
    relevant = defaultdict(set)
    for u, i, r, _ in test:
        if r >= threshold:
            relevant[u].add(i)

    prec = rec = ndcg = 0.0
    recommended = set()
    for u, rel in relevant.items():
        top = model.recommend(u, k)
        recommended.update(top)
        hits = [1 if i in rel else 0 for i in top]
        prec += sum(hits) / k
        rec += sum(hits) / len(rel)
        dcg = sum(h / math.log2(pos + 2) for pos, h in enumerate(hits))
        idcg = sum(1 / math.log2(pos + 2) for pos in range(min(len(rel), k)))
        ndcg += dcg / idcg
    n = len(relevant)
    return {
        f"P@{k}": prec / n,
        f"R@{k}": rec / n,
        f"NDCG@{k}": ndcg / n,
        "Coverage": len(recommended) / len(model.all_items),
    }
