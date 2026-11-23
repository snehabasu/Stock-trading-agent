import csv
from src.call_estimate import estimate, split_days
from src.database import Database
from src.labeling import read_labels, sample_for_labeling, score, write_label_csv
from tests.helpers import make_news


def test_score_precision_recall():
    labels = {("a", "T"): True, ("b", "T"): True, ("c", "T"): False, ("d", "T"): False}
    preds = {("a", "T"): True, ("b", "T"): False, ("c", "T"): True, ("d", "T"): False}
    r = score(labels, preds)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 1, 1, 1)
    assert r["precision"] == 0.5 and r["recall"] == 0.5 and len(r["disagreements"]) == 2


def test_sample_one_pair_per_article_and_roundtrip(tmp_path):
    db = Database(tmp_path / "m.sqlite"); db.initialize()
    for i in range(30):
        db.upsert_news(make_news(f"n{i}", f"2025-03-{1 + i % 28:02d}T12:00:00Z", tickers=["MSFT", "NVDA"]))
    rows = sample_for_labeling(db, "2025-03-01T00:00:00Z", "2025-07-01T00:00:00Z", n=20)
    assert len(rows) == 20 and len({r["article_id"] for r in rows}) == 20
    assert {r["ticker"] for r in rows} == {"MSFT", "NVDA"}
    path = write_label_csv(rows, tmp_path / "labels.csv")
    with open(path, newline="") as f:
        data = list(csv.DictReader(f))
    data[0]["relevant"], data[1]["relevant"] = "yes", "0"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    assert list(read_labels(path).values()) == [True, False]


def test_split_days_and_naive_formula():
    assert split_days() == {"development": 92, "validation": 61, "test": 51}
    r = estimate(None, articles_per_day=25)
    assert r["naive_calls"] == 204 * 6 * 2
    assert r["cached_calls"] < r["naive_calls"]
