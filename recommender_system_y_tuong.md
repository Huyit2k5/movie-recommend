# Ý tưởng làm thử Recommender System

Nguyên tắc: đi từ đơn giản đến phức tạp, mỗi bước có một bản chạy được rồi mới nâng cấp.

## 1. Chọn bài toán và dữ liệu

- **MovieLens 100K**: nhỏ, sạch, có rating 1-5 của user cho phim (file `u.data`: `user_id, item_id, rating, timestamp`; `u.item`: thông tin phim và thể loại).
- Hoặc **tự tạo dữ liệu nhỏ** (khoảng 10 user × 10 món ăn/sách) để dễ kiểm tra kết quả bằng tay.

## 2. Lộ trình mô hình

| Bước | Mô hình | Ý tưởng |
|---|---|---|
| 1 | Popularity baseline | Gợi ý item có nhiều rating hoặc điểm trung bình cao nhất. Dùng làm mốc so sánh. |
| 2 | Content-based | Dựa vào đặc trưng item (thể loại, tag), gợi ý item giống những gì user đã thích. |
| 3 | Collaborative filtering (user-based / item-based) | Dùng cosine similarity tìm user/item giống nhau, rồi dự đoán rating. |
| 4 | Matrix Factorization | Phân rã ma trận user-item thành vector ẩn (SGD hoặc ALS). |
| 5 | Nâng cao (tùy chọn) | Neural CF, hybrid, embedding. |

Mới bắt đầu thì làm đến bước 3 là đủ hiểu bản chất. **Item-based CF** dễ hiểu và dễ code nhất.

## 3. Ý tưởng Item-based CF (Python thuần)

**Approach**
1. Đọc dữ liệu, xây dict `ratings[user][item] = rating`.
2. Đảo thành `item_users[item] = {user: rating}`.
3. Tính cosine similarity giữa hai item dựa trên các user đã rate cả hai.
4. Với user `u` và item chưa xem `i`: điểm dự đoán = trung bình có trọng số rating của `u` cho các item giống `i` (trọng số là độ tương đồng).
5. Sắp xếp giảm dần, lấy top-K item chưa xem làm gợi ý.

**Edge cases**
- User mới (cold-start): không có lịch sử, fallback về popularity.
- Item mới: chưa có rating, không tính được similarity, fallback về content-based hoặc bỏ qua.
- Hai item không có user chung: similarity = 0.
- Mẫu số bằng 0 (tổng similarity = 0): trả về điểm trung bình của item hoặc của user.
- Item ít rating (ví dụ chỉ 1 user) dễ cho similarity = 1 giả: đặt ngưỡng tối thiểu số user chung.
- Không gợi ý lại item user đã xem.

## 4. Cách đánh giá

- Chia train/test theo user hoặc theo thời gian (giữ lại vài rating cuối của mỗi user làm test).
- **Dự đoán rating**: RMSE, MAE.
- **Xếp hạng top-K** (quan trọng hơn trong thực tế): Precision@K, Recall@K, NDCG@K.
- So mọi mô hình với popularity baseline, vì baseline này thường mạnh hơn dự đoán.

## 5. Các vấn đề nên chủ động gặp

- **Cold-start**: user/item mới chưa có dữ liệu. Thường dùng content-based hoặc popularity.
- **Sparsity**: ma trận user-item phần lớn là ô trống.
- **Popularity bias**: hệ thống chỉ gợi ý toàn item hot.
- **Implicit vs explicit feedback**: rating là explicit, còn click/xem/mua là implicit.

## 6. Các bước bắt đầu

1. Load MovieLens 100K, xây ma trận user-item.
2. Làm popularity baseline, đo Precision@10.
3. Làm item-based CF với cosine similarity, đo lại và so sánh.
4. Cuối cùng thử matrix factorization.
