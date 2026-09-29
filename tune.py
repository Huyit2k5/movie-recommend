"""Tinh chỉnh Hybrid trên tập validation, đo cuối cùng trên test (chỉ đo một lần).

    python tune.py
"""
import itertools
import sys
import time

from recsys.data import load_items, load_ratings, temporal_split, to_user_dict
from recsys.fast import FastItemCF, Hybrid
from recsys.metrics import ranking_metrics
from recsys.models import ItemCF, Popularity


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows = load_ratings()
    _, genres = load_items()
    train_rows, test = temporal_split(rows, n_test=10)
    # validation = 5 rating cuối của mỗi user TRONG train; test không bị chạm tới
    fit_rows, val = temporal_split(train_rows, n_test=5)
    fit, train = to_user_dict(fit_rows), to_user_dict(train_rows)

    # 1) Kiểm tra bản nhanh khớp bản chậm và nhanh hơn bao nhiêu
    t = time.time(); slow = ItemCF(k=30, min_common=5).fit(train, genres); ts = time.time() - t
    t = time.time(); fast = FastItemCF(k=30, min_common=5).fit(train, genres); tf = time.time() - t
    a, b = ranking_metrics(slow, test), ranking_metrics(fast, test)
    print(f"Fit: chậm {ts:.1f}s -> nhanh {tf:.1f}s | NDCG@10 chậm {a['NDCG@10']:.3f} = nhanh {b['NDCG@10']:.3f}\n")

    # 2) Grid search trên validation
    grid = itertools.product([20, 30, 50], [3, 5, 10], [0.0, 0.2, 0.4, 0.8], [0.0, 0.2, 0.4])
    best, results = None, []
    cache = {}
    for k, mc, w_pop, w_ct in grid:
        if (k, mc) not in cache:  # similarity chỉ phụ thuộc k, min_common: fit một lần
            cache[k, mc] = Hybrid(k=k, min_common=mc).fit(fit, genres)
        m = cache[k, mc]
        m.w_pop, m.w_ct = w_pop, w_ct
        ndcg = ranking_metrics(m, val)["NDCG@10"]
        results.append((ndcg, k, mc, w_pop, w_ct))
        if best is None or ndcg > best[0]:
            best = (ndcg, k, mc, w_pop, w_ct)
    results.sort(reverse=True)
    print("Top 5 trên validation (NDCG@10, k, min_common, w_pop, w_ct):")
    for r in results[:5]:
        print("  ", tuple(round(x, 3) for x in r))

    # 3) Đo cuối trên test với tham số tốt nhất, fit lại trên toàn bộ train
    _, k, mc, w_pop, w_ct = best
    final = Hybrid(k=k, min_common=mc, w_pop=w_pop, w_ct=w_ct).fit(train, genres)
    print(f"\nTest (tham số: k={k}, min_common={mc}, w_pop={w_pop}, w_ct={w_ct})")
    for m in [Popularity().fit(train, genres), fast, final]:
        r = ranking_metrics(m, test)
        print(f"  {m.name:<30} P@10 {r['P@10']:.3f}  R@10 {r['R@10']:.3f}  "
              f"NDCG@10 {r['NDCG@10']:.3f}  Coverage {r['Coverage']:.1%}")


if __name__ == "__main__":
    main()
