"""Kiểm tra edge case trên dữ liệu nhỏ tự tạo (dễ tính bằng tay).

    python -m unittest discover tests
"""
import math
import unittest

from recsys.models import ContentBased, ItemCF, MatrixFactorization, Popularity

# 4 user x 5 phim. Phim 1 và 2 luôn được chấm giống nhau; phim 5 không ai xem (item mới).
RATINGS = {
    "an":   {1: 5, 2: 5, 3: 1},
    "binh": {1: 4, 2: 4, 3: 2, 4: 5},
    "chi":  {1: 1, 2: 1, 3: 5},
    "dung": {3: 4, 4: 2},
}
GENRES = {1: [1, 0], 2: [1, 0], 3: [0, 1], 4: [0, 1], 5: [1, 1]}


class ToyTest(unittest.TestCase):
    def setUp(self):
        self.cf = ItemCF(k=10, min_common=1).fit(RATINGS, GENRES)

    def test_cosine_by_hand(self):
        # phim 1 và 2 có vector rating giống hệt nhau -> sim = 1
        self.assertAlmostEqual(self.cf.sim[1][2], 1.0)
        # sim(1,3) = (5*1 + 4*2 + 1*5) / (sqrt(25+16+1) * sqrt(1+4+25+16))
        self.assertAlmostEqual(self.cf.sim[1][3], 18 / (math.sqrt(42) * math.sqrt(46)))

    def test_no_common_users(self):
        # phim 5 không có ai rate -> không có similarity với phim nào
        self.assertEqual(self.cf.sim.get(5, {}), {})

    def test_min_common_threshold(self):
        # phim 2 và 4 chỉ có 1 user chung (binh) -> bị loại khi min_common = 2
        strict = ItemCF(k=10, min_common=2).fit(RATINGS, GENRES)
        self.assertIn(4, self.cf.sim[2])
        self.assertNotIn(4, strict.sim[2])

    def test_predict_weighted_average(self):
        # dung chưa xem phim 1; hàng xóm là phim 3 (r=4) và phim 4 (r=2)
        s3, s4 = self.cf.sim[1][3], self.cf.sim[1][4]
        self.assertAlmostEqual(self.cf.predict("dung", 1), (s3 * 4 + s4 * 2) / (s3 + s4))

    def test_new_item_falls_back(self):
        # item mới: không tính được similarity -> trả về trung bình của user
        self.assertAlmostEqual(self.cf.predict("an", 5), (5 + 5 + 1) / 3)

    def test_never_recommend_seen(self):
        for model in [Popularity(), ContentBased(), self.cf, MatrixFactorization(epochs=5)]:
            model.fit(RATINGS, GENRES)
            for u, seen in RATINGS.items():
                self.assertTrue(set(model.recommend(u, 10)).isdisjoint(seen), model.name)

    def test_cold_start_user_gets_popular(self):
        # user mới -> popularity: phim 3 (4 rating) rồi phim 1, 2 (3 rating)
        self.assertEqual(self.cf.recommend("moi", 3), [3, 1, 2])

    def test_content_based_follows_taste(self):
        # an thích thể loại 0 (phim 1, 2), ghét thể loại 1 (phim 3) -> phim 5 (có cả hai) > phim 4
        cb = ContentBased().fit(RATINGS, GENRES)
        self.assertEqual(cb.recommend("an", 2), [5, 4])


if __name__ == "__main__":
    unittest.main()
