from datetime import datetime,timezone
import pytest
from src.database import Database
from src.point_in_time import get_news
from tests.helpers import make_news

def test_future_article_excluded(tmp_path):
    db=Database(tmp_path/"t.sqlite"); db.initialize()
    db.upsert_news(make_news("old","2025-10-01T14:00:00Z"))
    db.upsert_news(make_news("future","2025-10-01T16:00:00Z"))
    as_of=datetime(2025,10,1,15,0,tzinfo=timezone.utc)
    assert [x["id"] for x in get_news(db,"AAPL",as_of)]==["old"]

def test_later_updated_article_excluded(tmp_path):
    db=Database(tmp_path/"t.sqlite"); db.initialize()
    db.upsert_news(make_news("edited","2025-10-01T14:00:00Z","2025-10-01T16:00:00Z"))
    as_of=datetime(2025,10,1,15,0,tzinfo=timezone.utc)
    assert get_news(db,"AAPL",as_of)==[]

def test_multi_ticker_and_ticker_filter(tmp_path):
    db=Database(tmp_path/"t.sqlite"); db.initialize()
    db.upsert_news(make_news("both","2025-10-01T14:00:00Z",tickers=["AAPL","MSFT"]))
    as_of=datetime(2025,10,1,15,0,tzinfo=timezone.utc)
    assert len(get_news(db,"AAPL",as_of))==1 and len(get_news(db,"MSFT",as_of))==1
    assert get_news(db,"NVDA",as_of)==[]

def test_naive_datetime_rejected(tmp_path):
    db=Database(tmp_path/"t.sqlite"); db.initialize()
    with pytest.raises(ValueError): get_news(db,"AAPL",datetime(2025,10,1,15,0))
