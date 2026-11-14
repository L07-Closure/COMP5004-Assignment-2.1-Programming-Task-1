import unittest

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier


class StreamTrainerTests(unittest.TestCase):
    def make_trainer(self, buffer_size=2):
        pipeline = Pipeline([
            ("model", DecisionTreeClassifier(buffer_size=buffer_size)),
        ])
        return StreamTrainer(pipeline)

    def test_fit_score_and_log(self):
        trainer = self.make_trainer()
        self.assertIs(trainer.fit_chunk([[0], [1]], [0, 1]), trainer)

        self.assertEqual(trainer.score_chunk([[0], [1]], [0, 1]), 1.0)
        log = trainer.logs[0]
        self.assertEqual(log["chunk"], 1)
        self.assertEqual(log["n_samples"], 2)
        self.assertEqual(log["accuracy"], 1.0)
        self.assertEqual(log["cumulative_accuracy"], 1.0)
        self.assertGreater(log["model_memory_bytes"], 0)

    def test_cumulative_accuracy_across_chunks(self):
        trainer = self.make_trainer()
        trainer.fit_chunk([[0], [1]], [0, 1])

        self.assertEqual(trainer.score_chunk([[0], [1]], [0, 0]), 0.5)
        self.assertEqual(trainer.score_chunk([[1], [1]], [1, 1]), 1.0)
        self.assertEqual(len(trainer.logs), 2)
        self.assertEqual(trainer.logs[1]["cumulative_accuracy"], 0.75)

    def test_score_waits_for_first_retraining(self):
        trainer = self.make_trainer(buffer_size=3)
        trainer.fit_chunk([[0], [1]], [0, 1])

        with self.assertRaises(RuntimeError):
            trainer.score_chunk([[0]], [0])
        self.assertEqual(trainer.logs, [])

        trainer.fit_chunk([[2]], [1])
        self.assertEqual(trainer.score_chunk([[0], [2]], [0, 1]), 1.0)

    def test_invalid_score_does_not_add_log(self):
        trainer = self.make_trainer()
        trainer.fit_chunk([[0], [1]], [0, 1])

        with self.assertRaises(ValueError):
            trainer.score_chunk([[0], [1]], [0])
        self.assertEqual(trainer.logs, [])

    def test_requires_pipeline(self):
        with self.assertRaises(TypeError):
            StreamTrainer(object())


if __name__ == "__main__":
    unittest.main()