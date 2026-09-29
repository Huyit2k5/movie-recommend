"""Chạy toàn bộ lộ trình: Popularity -> Content-based -> Item-based CF -> Matrix Factorization.

    python main.py            # đánh giá tất cả mô hình
    python main.py --user 1   # thêm demo gợi ý cho user 1
"""
import argparse
import sys
import time

from recsys.data import load_items, load_ratings, temporal_split, to_user_dict
from recsys.metrics import ranking_metrics, rmse_mae
from recsys.models import ContentBased, ItemCF, MatrixFactorization, Popularity


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, default=None, help="in top-10 gợi ý cho user này")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    rows = load_ratings()
    titles, genres = load_items()
    train_rows, test = temporal_split(rows, n_test=10)
    train = to_user_dict(train_rows)

    n_users, n_items = len({r[0] for r in rows}), len(titles)
    sparsity = 1 - len(rows) / (n_users * n_items)
    print(f"MovieLens 100K: {len(rows)} rating, {n_users} user, {n_items} phim, sparsity = {sparsity:.1%}")
    print(f"Train: {len(train_rows)} | Test: {len(test)} (10 rating cuối của mỗi user)\n")

    models = [
        Popularity(),
        ContentBased(k=30),
        ItemCF(k=30, min_common=5),
        ItemCF(k=30, min_common=5, mean_center=True),
        MatrixFactorization(factors=50, epochs=30, lr=0.01, reg=0.05),
    ]

    k = args.k
    header = f"{'Mô hình':<32}{'RMSE':>7}{'MAE':>7}{f'P@{k}':>8}{f'R@{k}':>8}{f'NDCG@{k}':>9}{'Cover':>7}{'Time':>7}"
    print(header)
    print("-" * len(header))
    for m in models:
        t0 = time.time()
        m.fit(train, genres)
        rmse, mae = rmse_mae(m, test)
        rk = ranking_metrics(m, test, k=k)
        dt = time.time() - t0
        print(f"{m.name:<32}{rmse:>7.3f}{mae:>7.3f}{rk[f'P@{k}']:>8.3f}{rk[f'R@{k}']:>8.3f}"
              f"{rk[f'NDCG@{k}']:>9.3f}{rk['Coverage']:>7.1%}{dt:>6.0f}s")

    if args.user is not None:
        u = args.user
        liked = sorted(train.get(u, {}).items(), key=lambda x: -x[1])[:5]
        print(f"\nUser {u} đã thích (train): " + "; ".join(titles[i] for i, _ in liked))
        for m in models:
            print(f"\n[{m.name}]")
            for rank, i in enumerate(m.recommend(u, k), 1):
                print(f"  {rank:>2}. {titles[i]}  (dự đoán {m.predict(u, i):.2f})")


if __name__ == "__main__":
    main()
