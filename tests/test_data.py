from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mfh.contracts import Question
from mfh.data.benchmarks import load_aa_csv, load_simpleqa_csv
from mfh.data.contamination import find_overlaps
from mfh.data.io import read_questions, write_question_bundle, write_questions
from mfh.data.normalization import normalize_question
from mfh.data.splits import (
    ResearchSplit,
    SplitPlan,
    assert_disjoint,
    exclude_exact_duplicate_groups,
    make_research_splits,
    semantic_group_ids,
)
from mfh.errors import DataValidationError
from mfh.provenance import stable_hash


def question(index: int, *, text: str | None = None, alias: str | None = None) -> Question:
    return Question(
        question_id=f"q{index}",
        benchmark="toy",
        text=text or f"Who is person number {index}?",
        aliases=(alias or f"Person {index}",),
        entities=(f"Entity {index}",),
    )


class DataIOTests(unittest.TestCase):
    def test_question_jsonl_round_trip(self) -> None:
        values = [question(1), question(2)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            self.assertEqual(write_questions(path, values), 2)
            self.assertEqual(list(read_questions(path)), values)
            with self.assertRaises(FileExistsError):
                write_questions(path, values)

    def test_question_bundle_collision_cannot_publish_partial_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "splits"
            destination.mkdir()
            sentinel = destination / "existing.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_question_bundle(
                    destination,
                    {"one.jsonl": [question(1)], "two.jsonl": [question(2)]},
                )
            self.assertEqual([path.name for path in destination.iterdir()], ["existing.txt"])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_question_bundle_rejects_ids_duplicated_across_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "splits"
            with self.assertRaises(DataValidationError):
                write_question_bundle(
                    destination,
                    {
                        "one.jsonl": [question(1)],
                        "two.jsonl": [question(1, text="Different text")],
                    },
                )
            self.assertFalse(destination.exists())

    def test_question_bundle_publishes_metadata_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "splits"
            write_question_bundle(
                destination,
                {"one.jsonl": [question(1)]},
                metadata_files={"curation-report.json": {"policy": "conservative"}},
            )
            self.assertEqual(
                (destination / "curation-report.json").read_text(encoding="utf-8"),
                '{\n  "policy": "conservative"\n}\n',
            )

    def test_csv_adapters_match_official_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            simpleqa_path = Path(directory) / "simpleqa.csv"
            simpleqa_path.write_text(
                "original_index,problem,answer,topic\n5,Capital of France?,Paris,Geography\n",
                encoding="utf-8",
            )
            aa_path = Path(directory) / "aa.csv"
            aa_path.write_text(
                "domain,topic,question_id,question,answer\n"
                "Science,Physics,1,Unit of force?,Newton\n",
                encoding="utf-8",
            )
            simpleqa = list(load_simpleqa_csv(simpleqa_path))
            aa = list(load_aa_csv(aa_path))
        self.assertEqual(simpleqa[0].question_id, "simpleqa:5")
        self.assertEqual(simpleqa[0].aliases, ("Paris",))
        self.assertEqual(aa[0].metadata["domain"], "Science")


class SplitTests(unittest.TestCase):
    def test_exact_disjoint_split_and_reserved_remainder(self) -> None:
        result = make_research_splits(
            [question(index) for index in range(12)],
            SplitPlan(steer=4, controller=2, dev=2, test=2, seed=3),
        )
        assert_disjoint(result)
        self.assertEqual(len(result.splits[ResearchSplit.T_STEER]), 4)
        self.assertEqual(len(result.splits[ResearchSplit.T_CONTROLLER]), 2)
        self.assertEqual(len(result.splits[ResearchSplit.T_DEV]), 2)
        self.assertEqual(len(result.splits[ResearchSplit.T_TEST]), 2)
        self.assertEqual(len(result.splits[ResearchSplit.RESERVED]), 2)

    def test_conflicting_exact_duplicate_is_rejected(self) -> None:
        first = question(1, text="What is X?", alias="One")
        second = question(2, text="what is x", alias="Two")
        with self.assertRaises(DataValidationError):
            make_research_splits(
                [first, second],
                SplitPlan(steer=1, controller=0, dev=0, test=0),
            )

    def test_conservative_curation_excludes_every_collision_group(self) -> None:
        connected_a = question(1, text="What is X?", alias="One")
        connected_b = question(2, text="what is x", alias="One")
        disconnected_a = question(3, text="What is Y?", alias="Two")
        disconnected_b = question(4, text="what is y", alias="Three")
        retained = question(5)

        result = exclude_exact_duplicate_groups(
            [connected_a, connected_b, disconnected_a, disconnected_b, retained]
        )

        self.assertEqual(result.questions, (retained,))
        self.assertEqual(result.report.input_count, 5)
        self.assertEqual(result.report.retained_count, 1)
        self.assertEqual(result.report.excluded_question_count, 4)
        self.assertEqual(result.report.duplicate_group_count, 2)
        self.assertEqual(result.report.alias_connected_group_count, 1)
        self.assertEqual(result.report.alias_disconnected_group_count, 1)
        report = result.report.to_dict()
        digest = report.pop("report_digest")
        self.assertEqual(digest, stable_hash(report))
        self.assertEqual(
            report["excluded_question_ids_sha256"], stable_hash(["q1", "q2", "q3", "q4"])
        )

    def test_duplicate_question_ids_are_rejected_before_assignment(self) -> None:
        with self.assertRaises(DataValidationError):
            make_research_splits(
                [question(1), question(1, text="A different question?")],
                SplitPlan(steer=1, controller=1, dev=0, test=0),
            )

    def test_shared_alias_never_crosses_non_reserved_splits(self) -> None:
        values = [question(index) for index in range(8)]
        values[1] = question(1, alias="Shared Person")
        values[2] = question(2, alias="Shared Person")
        result = make_research_splits(
            values,
            SplitPlan(steer=3, controller=2, dev=1, test=1),
            require_exact_sizes=False,
        )
        assert_disjoint(result)

    def test_semantic_group_ids_reuse_the_exact_split_leakage_policy(self) -> None:
        values = [question(index) for index in range(4)]
        values[1] = question(1, alias="Shared Person")
        values[2] = question(2, alias="Shared Person")
        groups = semantic_group_ids(values)
        self.assertEqual(groups["q1"], groups["q2"])
        self.assertNotEqual(groups["q0"], groups["q1"])
        with self.assertRaises(TypeError):
            groups["q0"] = "forged"  # type: ignore[index]
        with self.assertRaises(DataValidationError):
            semantic_group_ids((question(1), question(1)))

    def test_duplicate_features_are_merged_before_leakage_grouping(self) -> None:
        duplicate_a = Question("q1", "toy", "What is the answer?", ("One",), entities=("E1",))
        duplicate_b = Question("q2", "toy", "what is answer", ("One", "Shared"), entities=("E2",))
        linked = Question("q3", "toy", "A different question?", ("Shared",))
        fillers = [question(index) for index in range(4, 10)]
        result = make_research_splits(
            [duplicate_a, duplicate_b, linked, *fillers],
            SplitPlan(steer=3, controller=2, dev=1, test=1),
            require_exact_sizes=False,
        )
        locations = {
            item.question_id: split for split, items in result.splits.items() for item in items
        }
        self.assertEqual(locations["q1"], locations["q3"])
        merged = next(
            item for items in result.splits.values() for item in items if item.question_id == "q1"
        )
        self.assertIn("Shared", merged.aliases)
        self.assertIn("E2", merged.entities)


class ContaminationTests(unittest.TestCase):
    def test_exact_normalized_overlap_is_reported_for_removal(self) -> None:
        source = [question(1, text="Who discovered penicillin?"), question(2)]
        target = [question(3, text="WHO discovered penicillin!")]
        report = find_overlaps(source, target)
        self.assertEqual(report.source_ids_to_remove, ("q1",))
        self.assertTrue(report.matches[0].exact)

    def test_semantic_comparator_sees_lexically_distant_pairs(self) -> None:
        source = [question(1, text="Who wrote Hamlet?")]
        target = [question(2, text="Name the playwright responsible for the Danish prince tragedy")]
        report = find_overlaps(
            source,
            target,
            ngram_threshold=1.0,
            embedding_similarity=lambda left, right: 0.95,
            embedding_threshold=0.9,
        )
        self.assertEqual(report.source_ids_to_remove, ("q1",))
        self.assertFalse(report.matches[0].exact)

    def test_question_normalization_removes_articles(self) -> None:
        self.assertEqual(normalize_question("Who is The Doctor?"), "who is doctor")


if __name__ == "__main__":
    unittest.main()
