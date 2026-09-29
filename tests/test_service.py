"""Test bản nhanh khớp bản chậm, và các endpoint API.

    python -m unittest discover tests
"""
import unittest

from fastapi.testclient import TestClient

from recsys.fast import FastItemCF
from recsys.models import ItemCF
from tests.test_toy import GENRES, RATINGS


class FastMatchesSlow(unittest.TestCase):
    def test_similarity_and_predict_identical(self):
        slow = ItemCF(k=10, min_common=1).fit(RATINGS, GENRES)
        fast = FastItemCF(k=10, min_common=1).fit(RATINGS, GENRES)
        for i in slow.sim:
            for j, s in slow.sim[i].items():
                self.assertAlmostEqual(float(fast.S[fast.iidx[i], fast.iidx[j]]), s, places=5)
        for u in RATINGS:
            for i in GENRES:
                if i not in RATINGS[u]:
                    self.assertAlmostEqual(fast.predict(u, i), slow.predict(u, i), places=4)

    def test_recommend_excludes_seen_and_handles_cold_start(self):
        fast = FastItemCF(k=10, min_common=1).fit(RATINGS, GENRES)
        for u, seen in RATINGS.items():
            self.assertTrue(set(fast.recommend(u, 10)).isdisjoint(seen))
        self.assertEqual(fast.recommend("moi", 3), [3, 1, 2])


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from api import app
        cls.ctx = TestClient(app)
        cls.client = cls.ctx.__enter__()  # chạy lifespan để nạp model

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    def test_health(self):
        r = self.client.get("/health").json()
        self.assertEqual((r["movies"], r["users"]), (1682, 943))

    def test_search(self):
        r = self.client.get("/movies/search", params={"q": "star wars"}).json()
        self.assertTrue(r and "Star Wars" in r[0]["title"])

    def test_new_user_recommendations_with_reasons(self):
        r = self.client.post("/recommendations", json={"ratings": {"50": 5, "1": 5, "181": 4}, "k": 5}).json()
        self.assertEqual(r["strategy"], "item_cf")
        self.assertEqual(len(r["recommendations"]), 5)
        seen = {50, 1, 181}
        for rec in r["recommendations"]:
            self.assertNotIn(rec["id"], seen)
            self.assertTrue(rec["because"])

    def test_empty_ratings_falls_back_to_popularity(self):
        r = self.client.post("/recommendations", json={"ratings": {}, "k": 3}).json()
        self.assertEqual(r["strategy"], "popularity")
        self.assertEqual(len(r["recommendations"]), 3)

    def test_existing_user_and_similar(self):
        self.assertEqual(len(self.client.get("/users/1/recommendations?k=7").json()["recommendations"]), 7)
        sim = self.client.get("/movies/50/similar?k=5").json()
        self.assertEqual(len(sim), 5)
        self.assertNotIn(50, [m["id"] for m in sim])

    def test_bad_input_returns_422(self):
        self.assertEqual(self.client.get("/users/99999/recommendations").status_code, 422)
        self.assertEqual(self.client.get("/movies/99999/similar").status_code, 422)
        self.assertEqual(self.client.post("/recommendations", json={"ratings": {"50": 9}}).status_code, 422)
        self.assertEqual(self.client.post("/recommendations", json={"ratings": {"99999": 5}}).status_code, 422)
        self.assertEqual(self.client.post("/recommendations", json={"ratings": {}, "k": 0}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
