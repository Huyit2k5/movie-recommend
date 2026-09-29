"""API gợi ý phim.   uvicorn api:app --port 8000   (docs tại /docs)"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from recsys.service import BadRequest, RecService

state = {}


@asynccontextmanager
async def lifespan(app):
    state["svc"] = RecService.load()  # nạp một lần khi khởi động
    yield


app = FastAPI(title="Movie Recommender", version="1.0", lifespan=lifespan)


class RatingsIn(BaseModel):
    ratings: dict[int, float] = Field(
        default_factory=dict, description="{movie_id: rating 1-5}. Rỗng = user mới, trả phim phổ biến.")
    k: int = Field(10, ge=1, le=50)


def call(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except BadRequest as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/health")
def health():
    m = state["svc"].model
    return {"status": "ok", "movies": len(m.all_items), "users": len(m.ratings)}


@app.get("/movies/search")
def search(q: str = Query(min_length=1), limit: int = Query(20, ge=1, le=100)):
    return state["svc"].search(q, limit)


@app.get("/movies/{movie_id}/similar")
def similar(movie_id: int, k: int = Query(10, ge=1, le=50)):
    return call(state["svc"].similar, movie_id, k)


@app.get("/users/{user_id}/recommendations")
def user_recommendations(user_id: int, k: int = Query(10, ge=1, le=50)):
    """Gợi ý cho user có sẵn trong MovieLens (id 1-943)."""
    return call(state["svc"].recommend_user, user_id, k)


@app.post("/recommendations")
def recommendations(body: RatingsIn):
    """Gợi ý cho user bất kỳ từ danh sách rating gửi lên, kèm lý do."""
    return call(state["svc"].recommend, body.ratings, body.k)
