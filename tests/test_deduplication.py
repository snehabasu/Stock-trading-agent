from src.database import Database
from tests.helpers import make_news

def test_dedupes_by_article_id(tmp_path):
    db=Database(tmp_path/"t.sqlite"); db.initialize()
    item=make_news("dup","2025-10-01T14:00:00Z")
    assert db.upsert_news(item) is True
    assert db.upsert_news(item) is False
    with db.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]==1

def test_stores_observed_article_versions(tmp_path):
    db=Database(tmp_path/"t.sqlite"); db.initialize()
    db.upsert_news(make_news("id","2025-10-01T14:00:00Z",headline="Original"))
    db.upsert_news(make_news("id","2025-10-01T14:00:00Z",headline="Edited"))
    with db.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM news_article_versions").fetchone()[0]==2
