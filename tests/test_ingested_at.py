from src.database import Database
from tests.helpers import make_news


def test_reingest_keeps_first_ingested_at(tmp_path):
    db = Database(tmp_path / "t.sqlite"); db.initialize()
    first = make_news("id", "2025-10-01T14:00:00Z"); first["ingested_at"] = "2025-10-01T14:01:00Z"
    later = make_news("id", "2025-10-01T14:00:00Z", headline="Edited"); later["ingested_at"] = "2025-11-01T00:00:00Z"
    db.upsert_news(first); db.upsert_news(later)
    with db.connect() as c:
        row = c.execute("SELECT ingested_at, headline FROM news_articles").fetchone()
        observed = [r[0] for r in c.execute("SELECT observed_at FROM news_article_versions ORDER BY 1")]
    assert tuple(row) == ("2025-10-01T14:01:00Z", "Edited")
    assert observed == ["2025-10-01T14:01:00Z", "2025-11-01T00:00:00Z"]
