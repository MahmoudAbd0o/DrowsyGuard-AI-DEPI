"""Unit tests for session logging."""
import csv
import os
import tempfile
import unittest

from data.logger import MEASUREMENT_FIELDS, EventLog, MeasurementLogger


class TestMeasurementLogger(unittest.TestCase):
    def test_writes_header_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "session_log.csv")
            with MeasurementLogger(path=path, buffer_size=2) as logger:
                logger.log(timestamp=0.0, state="ALERT", level="0", perclos=10.0)
                logger.log(timestamp=0.1, state="DROWSY", level="2", perclos=55.5)

            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, MEASUREMENT_FIELDS)
                rows = list(reader)

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["state"], "DROWSY")
            self.assertEqual(rows[1]["level"], "2")
            self.assertEqual(rows[1]["perclos"], "55.50")

    def test_missing_values_are_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "log.csv")
            with MeasurementLogger(path=path, buffer_size=1) as logger:
                logger.log(timestamp=1.0, state="UNKNOWN", level="N")
            with open(path, newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["perclos"], "")
            self.assertEqual(row["ear"], "")
            self.assertEqual(row["attention"], "")

    def test_boolean_columns_are_ints(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "log.csv")
            with MeasurementLogger(path=path, buffer_size=1) as logger:
                logger.log(timestamp=1.0, state="DROWSY", level="2",
                           head_down=True, distracted=False, face_present=True)
            with open(path, newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["head_down"], "1")
            self.assertEqual(row["distracted"], "0")


class TestEventLog(unittest.TestCase):
    def test_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "session_report.csv")
            log = EventLog(path=path)
            log.add("DROWSINESS ALERT #1", timestamp="12:00:00")
            log.add("Session autosaved.", timestamp="12:01:00")
            self.assertTrue(log.save())

            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["time", "event"])
            self.assertEqual(rows[1], ["12:00:00", "DROWSINESS ALERT #1"])
            self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
