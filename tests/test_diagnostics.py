"""Fast tests for the rubric's plot data helpers; no retraining required."""
import csv

from diagnostics.plots import moving_average, read_csv


def test_trailing_mean():
    assert moving_average([1.0, 2.0, 3.0, 4.0], window=2) == [1.0, 1.5, 2.5, 3.5]


def test_read_csv(tmp_path):
    file = tmp_path / "updates.csv"
    with file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["update", "loss"])
        writer.writeheader()
        writer.writerow({"update": 1, "loss": 0.5})
    assert read_csv(file) == [{"update": "1", "loss": "0.5"}]
