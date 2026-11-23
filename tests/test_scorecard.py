import unittest

from csv_shootout import ordinary_scores, run_csv_shootout


class ScorecardTests(unittest.TestCase):
    def test_partial_and_empty_queries(self):
        scores = ordinary_scores([[True], [True, False], []])
        self.assertEqual(scores["precision_at_k"], 2 / 30)
        self.assertEqual(scores["precision_returned"], 2 / 3)
        self.assertEqual(scores["empty"], 1)

    def test_all_refused(self):
        scores = ordinary_scores([[]])
        self.assertEqual(scores["precision_at_k"], 0)
        self.assertIsNone(scores["precision_returned"])

    def test_colliding_exports_refused_before_training(self):
        with self.assertRaisesRegex(ValueError, "different paths"):
            run_csv_shootout("unused.csv", "rid", export_bundle="demo.pt",
                             evaluation_bundle="./demo.pt")
