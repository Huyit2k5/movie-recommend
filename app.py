"""Demo gợi ý phim.   streamlit run app.py"""
import streamlit as st

from recsys.service import RecService

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Đang nạp model (lần đầu mất ~1 phút)...")
def get_model():
    return RecService.load()


svc = get_model()
cf, titles = svc.model, svc.titles
by_title = {t: i for i, t in titles.items()}

# Gợi ý sẵn các phim nổi tiếng để chọn nhanh
STARTER = [cf.popular[n] for n in range(40)]

st.title("🎬 Movie Recommender")
st.caption("Item-based collaborative filtering trên MovieLens 100K. "
           "Chấm điểm vài phim bạn đã xem, hệ thống sẽ gợi ý phim tiếp theo.")

if "ratings" not in st.session_state:
    st.session_state.ratings = {}
ratings = st.session_state.ratings

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("1. Phim bạn đã xem")
    choice = st.selectbox(
        "Tìm phim (gõ để lọc)",
        options=[titles[i] for i in sorted(titles, key=lambda i: -cf.item_count.get(i, 0))],
        index=None, placeholder="Ví dụ: Star Wars",
    )
    stars = st.slider("Bạn chấm mấy sao?", 1, 5, 5)
    if st.button("Thêm", disabled=choice is None, type="primary"):
        ratings[by_title[choice]] = stars

    with st.expander("Hoặc chọn nhanh trong 40 phim phổ biến nhất"):
        pick = st.multiselect("Phim bạn thích (sẽ tính là 5 sao)", [titles[i] for i in STARTER])
        if st.button("Thêm các phim này"):
            for t in pick:
                ratings[by_title[t]] = 5

    if ratings:
        st.markdown("**Đã chấm:**")
        for i, r in list(ratings.items()):
            c1, c2 = st.columns([5, 1])
            c1.write(f"{'⭐' * int(r)}  {titles[i]}")
            if c2.button("✕", key=f"del{i}"):
                del ratings[i]
                st.rerun()
        if st.button("Xoá hết"):
            ratings.clear()
            st.rerun()

with right:
    st.subheader("2. Gợi ý cho bạn")
    k = st.slider("Số gợi ý", 5, 20, 10)
    if len(ratings) < 3:
        st.info("Hãy chấm ít nhất **3 phim** để có gợi ý (user mới chưa có lịch sử = cold-start). "
                "Trong lúc chờ, đây là phim phổ biến nhất:")
        for n, i in enumerate(cf.popular[:k], 1):
            st.write(f"{n}. {titles[i]}")
    else:
        recs = svc.recommend(ratings, k)["recommendations"]
        col_cf, col_pop = st.columns([3, 2], gap="large")
        with col_cf:
            st.markdown("**Item-based CF**")
            if not recs:
                st.warning("Chưa tìm được phim giống. Thử thêm phim phổ biến hơn.")
            for n, rec in enumerate(recs, 1):
                st.markdown(f"**{n}. {rec['title']}**")
                why = ", ".join(f"*{b['title']}* ({b['your_rating']:.0f}⭐, giống {b['similarity']:.2f})"
                                for b in rec["because"])
                st.caption(f"Vì bạn đã xem: {why}")
        with col_pop:
            st.markdown("**Popularity (để so sánh)**")
            cf_set = {r['id'] for r in recs}
            for n, i in enumerate([i for i in cf.popular if i not in ratings][:k], 1):
                st.write(f"{n}. {titles[i]}" + (" ✅" if i in cf_set else ""))
            st.caption("✅ = cũng nằm trong gợi ý của CF. Phần còn lại của CF là gợi ý cá nhân hoá.")
