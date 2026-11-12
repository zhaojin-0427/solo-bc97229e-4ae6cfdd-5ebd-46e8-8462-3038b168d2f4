"""Core service tests — runnable with the Python stdlib only:

    python3 -I tests/test_core.py

They exercise the real SQLite-backed services (no mocks), including
concurrency, crash-style rollback semantics and every reconciliation rule.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import traceback
import unittest
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.csv_parser import CsvFormatError, parse_flow_csv  # noqa: E402
from app.db import Database  # noqa: E402
from app.ledger import upload  # noqa: E402
from app.matching import compute_candidates, validate_combination  # noqa: E402
from app.matching_service import (  # noqa: E402
    confirm_candidate,
    list_candidates,
    list_flows,
    list_groups,
    list_unmatched,
)
from app.models import (  # noqa: E402
    BANK,
    INTERNAL,
    MATCHED,
    UNMATCHED,
    CapacityExceeded,
    FlowRow,
    NotFound,
    ParsedRow,
    StaleCandidateError,
)
from app.recompute import current_generation, recompute  # noqa: E402
from app.repositories import load_flows  # noqa: E402


def rows(side, data: list[tuple]) -> list[ParsedRow]:
    return [ParsedRow(serial=s, name=n, txn_date=d, amount=a) for s, n, d, a in data]


BANK_CSV = (
    "流水号,户名,日期,金额(分)\n"
    "B1,甲公司,2026-11-10,10000\n"
    "B2,乙公司,2026-11-11,5000\n"
    "B3,丙公司,2026-11-12,9000\n"
)
INTERNAL_CSV = (
    "流水号,户名,日期,金额(分)\n"
    "I1,甲公司,2026-11-09,4000\n"
    "I2,甲公司,2026-11-11,6000\n"
    "I3,甲公司,2026-11-10,10000\n"
    "I4,乙公司,2026-11-10,2500\n"
    "I5,乙公司,2026-11-12,2500\n"
)


class ReconcileTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.tmp.name, "test.db"))

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def seed(self, bank_csv: str = BANK_CSV, internal_csv: str = INTERNAL_CSV) -> None:
        upload(self.db, BANK, parse_flow_csv(bank_csv))
        upload(self.db, INTERNAL, parse_flow_csv(internal_csv))
        recompute(self.db)

    def assert_invariant_no_half_match(self) -> None:
        """Every matched flow points at an existing complete group."""
        with self.db.read_tx() as conn:
            for table in ("bank_flows", "internal_flows"):
                for r in conn.execute(
                    f"SELECT serial, status, group_id FROM {table}"
                ).fetchall():
                    if r["status"] == MATCHED:
                        self.assertIsNotNone(r["group_id"])
                        g = conn.execute(
                            "SELECT 1 FROM match_groups WHERE id = ?",
                            (r["group_id"],),
                        ).fetchone()
                        self.assertIsNotNone(g, f"{table}.{r['serial']} 单边匹配!")
            groups = conn.execute("SELECT id FROM match_groups").fetchall()
            for g in groups:
                bn = conn.execute(
                    "SELECT COUNT(*) n FROM bank_flows WHERE group_id=? AND status=?",
                    (g["id"], MATCHED),
                ).fetchone()["n"]
                ins = conn.execute(
                    "SELECT COUNT(*) n FROM internal_flows WHERE group_id=? AND status=?",
                    (g["id"], MATCHED),
                ).fetchone()["n"]
                self.assertGreaterEqual(bn, 1)
                self.assertGreaterEqual(ins, 1)

    # ---- CSV parsing -----------------------------------------------------
    def test_csv_parsing_and_aliases(self) -> None:
        parsed = parse_flow_csv(BANK_CSV)
        self.assertEqual([r.serial for r in parsed], ["B1", "B2", "B3"])
        self.assertEqual(parsed[0].amount, 10000)
        en = parse_flow_csv("serial,name,date,amount\nX1,丁,2026/11/10,100\n")
        self.assertEqual(en[0].txn_date, "2026-11-10")

    def test_csv_rejects_bad_amount_and_duplicates(self) -> None:
        with self.assertRaises(CsvFormatError):
            parse_flow_csv("流水号,户名,日期,金额(分)\nX1,丁,2026-11-10,0\n")
        with self.assertRaises(CsvFormatError):
            parse_flow_csv("流水号,户名,日期,金额(分)\nX1,丁,2026-11-10,-5\n")
        with self.assertRaises(CsvFormatError):
            parse_flow_csv(
                "流水号,户名,日期,金额(分)\nX1,丁,2026-11-10,5\nX1,丁,2026-11-10,5\n"
            )
        with self.assertRaises(CsvFormatError):
            parse_flow_csv("流水号,户名,日期\nX1,丁,2026-11-10\n")

    def test_csv_capacity_limit(self) -> None:
        big = "流水号,户名,日期,金额(分)\n" + "\n".join(
            f"B{i:03d},丁,2026-11-10,100" for i in range(101)
        )
        with self.assertRaises(CsvFormatError):
            parse_flow_csv(big)

    # ---- candidate rules -------------------------------------------------
    def test_candidate_rules(self) -> None:
        self.seed()
        cands = list_candidates(self.db)
        by_bank: dict[str, list] = {}
        for c in cands:
            by_bank.setdefault(c["bank"]["serial"], []).append(c)
        # B1=10000: [I3] and [I1(4000)+I2(6000)]
        self.assertEqual(len(by_bank["B1"]), 2)
        for c in by_bank["B1"]:
            self.assertEqual(c["total_amount"], 10000)
            self.assertEqual(len(c["internals"]), 1) if c["internals"][0]["serial"] == "I3" \
                else self.assertEqual(len(c["internals"]), 2)
        # B2=5000: [I4+I5], dates 11-10 and 11-12 both within 2 days of 11-11
        self.assertEqual(len(by_bank["B2"]), 1)
        self.assertEqual(
            sorted(f["serial"] for f in by_bank["B2"][0]["internals"]),
            ["I4", "I5"],
        )
        # B3 丙公司 has no internal flows
        self.assertNotIn("B3", by_bank)

    def test_candidate_engine_pure_rules(self) -> None:
        with self.db.read_tx() as conn:
            # empty DB -> no candidates, no error
            self.assertEqual(compute_candidates(load_flows(conn, BANK),
                                                load_flows(conn, INTERNAL)), [])
        self.seed()
        with self.db.read_tx() as conn:
            specs = compute_candidates(
                load_flows(conn, BANK), load_flows(conn, INTERNAL)
            )
        self.assertTrue(all(1 <= len(s.internals) <= 3 for s in specs))

    def test_date_window_boundary_and_name(self) -> None:
        bank = rows(BANK, [("B1", "甲", "2026-11-10", 100)])
        internal = rows(
            INTERNAL,
            [
                ("I_OK_MINUS2", "甲", "2026-11-08", 100),
                ("I_OK_PLUS2", "甲", "2026-11-12", 100),
                ("I_BAD_DATE", "甲", "2026-11-13", 100),
                ("I_BAD_NAME", "乙", "2026-11-10", 100),
            ],
        )
        upload(self.db, BANK, bank)
        upload(self.db, INTERNAL, internal)
        recompute(self.db)
        serials = {
            f["serial"]
            for c in list_candidates(self.db)
            for f in c["internals"]
        }
        self.assertIn("I_OK_MINUS2", serials)
        self.assertIn("I_OK_PLUS2", serials)
        self.assertNotIn("I_BAD_DATE", serials)
        self.assertNotIn("I_BAD_NAME", serials)

    def test_max_three_internals(self) -> None:
        bank = rows(BANK, [("B1", "甲", "2026-11-10", 300)])
        internal = rows(
            INTERNAL,
            [
                ("I1", "甲", "2026-11-10", 100),
                ("I2", "甲", "2026-11-10", 100),
                ("I3", "甲", "2026-11-10", 100),
                ("I4", "甲", "2026-11-10", 100),
            ],
        )
        upload(self.db, BANK, bank)
        upload(self.db, INTERNAL, internal)
        recompute(self.db)
        cands = list_candidates(self.db)
        # four different 3-item subsets sum to 300; no 4-item combo allowed
        self.assertEqual(len(cands), 4)
        self.assertTrue(all(len(c["internals"]) <= 3 for c in cands))

    # ---- confirm ---------------------------------------------------------
    def test_confirm_happy_path_and_queries(self) -> None:
        self.seed()
        cand = next(c for c in list_candidates(self.db)
                    if c["bank"]["serial"] == "B2")
        group = confirm_candidate(self.db, cand["candidate_id"],
                                  cand["expected_versions"])
        self.assertEqual(group["group_id"], group["group_id"])
        self.assertEqual([f["serial"] for f in group["bank"]], ["B2"])
        self.assertEqual(
            sorted(f["serial"] for f in group["internals"]), ["I4", "I5"]
        )
        # all members matched, candidate gone from the confirmable list
        unmatched = list_unmatched(self.db)
        self.assertNotIn("B2", [f["serial"] for f in unmatched["bank"]])
        self.assertFalse(
            any(f["serial"] in ("I4", "I5") for f in unmatched["internal"])
        )
        ids = {g["group_id"] for g in list_groups(self.db)}
        self.assertIn(group["group_id"], ids)
        # candidates sharing B2/I4/I5 are now stale
        live = list_candidates(self.db)
        self.assertFalse(
            any(c["candidate_id"] == cand["candidate_id"] for c in live)
        )
        self.assert_invariant_no_half_match()

    def test_concurrent_confirm_only_one_wins(self) -> None:
        self.seed()
        cands = list_candidates(self.db)
        # B1 has two candidates sharing the bank flow B1.
        b1_cands = [c for c in cands if c["bank"]["serial"] == "B1"]
        self.assertEqual(len(b1_cands), 2)
        results: list = []
        errors: list = []

        def worker(cand_id: int, versions: dict) -> None:
            try:
                results.append(confirm_candidate(self.db, cand_id, versions))
            except StaleCandidateError as exc:
                errors.append(exc)
            except Exception:  # pragma: no cover
                errors.append(traceback.format_exc())

        t1 = threading.Thread(
            target=worker,
            args=(b1_cands[0]["candidate_id"], b1_cands[0]["expected_versions"]),
        )
        t2 = threading.Thread(
            target=worker,
            args=(b1_cands[1]["candidate_id"], b1_cands[1]["expected_versions"]),
        )
        t1.start(); t2.start(); t1.join(); t2.join()
        self.assertEqual(len(results), 1)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], StaleCandidateError)
        self.assert_invariant_no_half_match()
        # exactly one group involving B1 exists
        groups = list_groups(self.db)
        self.assertEqual(len(groups), 1)

    def test_stale_version_rejected_with_latest_payload(self) -> None:
        self.seed()
        cand = next(c for c in list_candidates(self.db)
                    if c["bank"]["serial"] == "B2")
        old_versions = dict(cand["expected_versions"])
        # tamper the page's view of one version
        key = next(iter(old_versions))
        old_versions[key] -= 1
        with self.assertRaises(StaleCandidateError) as ctx:
            confirm_candidate(self.db, cand["candidate_id"], old_versions)
        self.assertEqual(ctx.exception.status, 409)
        self.assertTrue(ctx.exception.latest["flows"])

    def test_late_recompute_cannot_confirm_outdated_combo(self) -> None:
        self.seed()
        cands_before = list_candidates(self.db)
        cand_b2 = next(c for c in cands_before if c["bank"]["serial"] == "B2")
        # Correct an internal flow after the page loaded the candidate.
        upload(
            self.db,
            INTERNAL,
            rows(INTERNAL, [("I4", "乙公司", "2026-11-10", 3000)]),
        )
        # A late recomputation happens after the correction.
        recompute(self.db)
        # The correction deleted the stale candidate row; either way the old
        # page can never confirm the outdated combination.
        with self.assertRaises((StaleCandidateError, NotFound)):
            confirm_candidate(
                self.db, cand_b2["candidate_id"], cand_b2["expected_versions"]
            )
        self.assertFalse(
            any(c["candidate_id"] == cand_b2["candidate_id"]
                for c in list_candidates(self.db))
        )
        self.assert_invariant_no_half_match()

    # ---- correction / disband -------------------------------------------
    def test_correction_disbands_whole_group(self) -> None:
        self.seed()
        cand = next(c for c in list_candidates(self.db)
                    if c["bank"]["serial"] == "B2")
        group = confirm_candidate(self.db, cand["candidate_id"],
                                  cand["expected_versions"])
        gid = group["group_id"]
        # Correct the bank flow of the confirmed group.
        report = upload(
            self.db, BANK, rows(BANK, [("B2", "乙公司", "2026-11-11", 5500)])
        )
        self.assertEqual(report.disbanded_groups, [gid])
        # every member is back to unmatched, group gone
        unmatched = list_unmatched(self.db)
        serials_b = {f["serial"] for f in unmatched["bank"]}
        serials_i = {f["serial"] for f in unmatched["internal"]}
        self.assertIn("B2", serials_b)
        self.assertIn("I4", serials_i)
        self.assertIn("I5", serials_i)
        self.assertEqual(list_groups(self.db), [])
        for side in (BANK, INTERNAL):
            for f in list_flows(self.db, side):
                self.assertIsNone(f["group_id"])
                self.assertEqual(f["status"], UNMATCHED)
        self.assert_invariant_no_half_match()
        # new raw value no longer matches the old amount; no candidate for B2
        recompute(self.db)
        self.assertFalse(
            any(c["bank"]["serial"] == "B2" for c in list_candidates(self.db))
        )

    def test_identical_reupload_is_noop(self) -> None:
        self.seed()
        before = {f["serial"]: f for f in list_flows(self.db, BANK)}
        report = upload(self.db, BANK, parse_flow_csv(BANK_CSV))
        self.assertEqual(report.upserted, 0)
        self.assertEqual(report.unchanged, 3)
        after = {f["serial"]: f for f in list_flows(self.db, BANK)}
        self.assertEqual(before, after)

    def test_same_serial_overwrite_and_version_bump(self) -> None:
        self.seed()
        old = next(f for f in list_flows(self.db, BANK) if f["serial"] == "B1")
        upload(self.db, BANK, rows(BANK, [("B1", "甲公司", "2026-11-10", 9999)]))
        new = next(f for f in list_flows(self.db, BANK) if f["serial"] == "B1")
        self.assertEqual(new["amount"], 9999)
        self.assertEqual(new["version"], old["version"] + 1)
        self.assertEqual(new["status"], UNMATCHED)

    # ---- capacity --------------------------------------------------------
    def test_side_capacity_100(self) -> None:
        first = rows(BANK, [(f"B{i}", "甲", "2026-11-10", 100) for i in range(60)])
        upload(self.db, BANK, first)
        more = rows(BANK, [(f"C{i}", "甲", "2026-11-10", 100) for i in range(40)])
        upload(self.db, BANK, more)  # exactly 100
        overflow = rows(BANK, [("X1", "甲", "2026-11-10", 100)])
        with self.assertRaises(CapacityExceeded):
            upload(self.db, BANK, overflow)
        # overwriting an existing serial still works at the cap
        upload(self.db, BANK, rows(BANK, [("B0", "甲", "2026-11-10", 200)]))

    # ---- durability / restart -------------------------------------------
    def test_restart_preserves_confirmed_matches(self) -> None:
        self.seed()
        cand = next(c for c in list_candidates(self.db)
                    if c["bank"]["serial"] == "B2")
        group = confirm_candidate(self.db, cand["candidate_id"],
                                  cand["expected_versions"])
        path = self.db.path
        self.db.close()
        db2 = Database(path)  # simulate process restart
        try:
            groups = list_groups(db2)
            self.assertEqual([g["group_id"] for g in groups], [group["group_id"]])
            g = groups[0]
            self.assertEqual([f["serial"] for f in g["bank"]], ["B2"])
            self.assertEqual(
                sorted(f["serial"] for f in g["internals"]), ["I4", "I5"]
            )
            self.assertTrue(all(f["status"] == MATCHED for f in g["bank"]))
            self.assert_invariant_no_half_match_on(db2)
        finally:
            db2.close()

    def assert_invariant_no_half_match_on(self, db: Database) -> None:
        with db.read_tx() as conn:
            for table in ("bank_flows", "internal_flows"):
                for r in conn.execute(
                    f"SELECT * FROM {table} WHERE status = ?", (MATCHED,)
                ).fetchall():
                    self.assertIsNotNone(r["group_id"])

    # ---- background recompute never blocks writers ----------------------
    def test_background_worker_lifecycle(self) -> None:
        from app.recompute import RecomputeWorker

        worker = RecomputeWorker(self.db, debounce_seconds=0.05)
        worker.start()
        self.seed()
        worker.schedule()
        # wait until the background refresh produces candidates
        deadline = 0
        while not list_candidates(self.db) and deadline < 50:
            deadline += 1
            threading.Event().wait(0.05)
        self.assertTrue(list_candidates(self.db))
        # signal stop without closing the shared Database object
        worker._stop.set()
        worker._wake.set()
        worker._thread.join(timeout=5)
        self.assertFalse(worker._thread.is_alive())

    def test_new_insert_starts_at_version_one(self) -> None:
        upload(self.db, BANK, rows(BANK, [("N1", "甲", "2026-11-10", 100)]))
        flow = next(f for f in list_flows(self.db, BANK) if f["serial"] == "N1")
        self.assertEqual(flow["version"], 1)

    def test_recompute_concurrent_with_writes(self) -> None:
        self.seed()
        stop = threading.Event()

        def writer() -> None:
            i = 0
            while not stop.is_set():
                try:
                    upload(
                        self.db,
                        INTERNAL,
                        rows(INTERNAL, [("W%04d" % i, "甲公司", "2026-11-10", 1)]),
                    )
                except CapacityExceeded:
                    # 100-row side cap reached; writes still work via overwrite
                    upload(
                        self.db,
                        INTERNAL,
                        rows(INTERNAL, [("W0000", "甲公司", "2026-11-10", 2)]),
                    )
                i += 1
                if i > 200:
                    break

        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = [pool.submit(writer) for _ in range(2)]
            for _ in range(5):
                recompute(self.db)
            stop.set()
            for f in futs:
                f.result(timeout=30)
        # writers kept succeeding past any single-side cap? cap=100 -> stop earlier
        self.assert_invariant_no_half_match()

    # ---- recompute publish concurrency (epoch-guarded) -------------------
    def _combos(self) -> set:
        """(bank, internals) tuples of the default (confirmable) list."""
        return {
            (
                c["bank"]["serial"],
                tuple(sorted(f["serial"] for f in c["internals"])),
            )
            for c in list_candidates(self.db)
        }

    @staticmethod
    def _flowrow(d: dict) -> FlowRow:
        return FlowRow(
            serial=d["serial"],
            name=d["name"],
            txn_date=d["txn_date"],
            amount=d["amount"],
            status=d["status"],
            group_id=d["group_id"],
            version=d["version"],
            content_hash=d["content_hash"],
            updated_at=d["updated_at"],
        )

    def _data_epoch(self) -> int:
        with self.db.read_tx() as conn:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'data_epoch'"
            ).fetchone()
            return int(row["value"]) if row else 0

    def assert_default_list_sound(self) -> None:
        """Every default-list candidate matches current values & claim state."""
        for c in list_candidates(self.db):  # one read tx -> consistent view
            bank = self._flowrow(c["bank"])
            internals = [self._flowrow(f) for f in c["internals"]]
            self.assertEqual(bank.status, UNMATCHED, c)
            self.assertTrue(
                all(f.status == UNMATCHED for f in internals),
                [f["serial"] for f in c["internals"]],
            )
            expected = c["expected_versions"]
            self.assertEqual(bank.version, expected[bank.serial], c)
            for f in internals:
                self.assertEqual(f.version, expected[f.serial], c)
            validate_combination(bank, internals)  # business rules still hold

    def assert_default_list_complete(self) -> None:
        """Default list == every candidate valid for the settled inputs."""
        with self.db.read_tx() as conn:
            specs = compute_candidates(
                load_flows(conn, BANK), load_flows(conn, INTERNAL)
            )
        expected = {s.candidate_key for s in specs}
        actual = {c["candidate_key"] for c in list_candidates(self.db)}
        self.assertEqual(actual, expected)

    def test_data_epoch_tracks_inputs_not_derived_writes(self) -> None:
        e0 = self._data_epoch()
        upload(self.db, BANK, rows(BANK, [("B1", "甲", "2026-11-10", 100)]))
        upload(self.db, INTERNAL, rows(INTERNAL, [("I1", "甲", "2026-11-10", 100)]))
        e1 = self._data_epoch()
        self.assertGreater(e1, e0)
        # identical re-upload changes nothing -> epoch must not move
        upload(self.db, BANK, rows(BANK, [("B1", "甲", "2026-11-10", 100)]))
        self.assertEqual(self._data_epoch(), e1)
        # publishing candidates is a derived write -> epoch must not move
        recompute(self.db)
        self.assertEqual(self._data_epoch(), e1)
        # a raw correction moves the epoch
        upload(self.db, INTERNAL, rows(INTERNAL, [("I1", "甲", "2026-11-10", 200)]))
        e2 = self._data_epoch()
        self.assertGreater(e2, e1)
        upload(self.db, INTERNAL, rows(INTERNAL, [("I1", "甲", "2026-11-10", 100)]))
        e3 = self._data_epoch()
        self.assertGreater(e3, e2)
        # a confirmation (claim-status change) moves the epoch
        recompute(self.db)
        cand = list_candidates(self.db)[0]
        confirm_candidate(self.db, cand["candidate_id"], cand["expected_versions"])
        self.assertGreater(self._data_epoch(), e3)

    def test_late_recompute_cannot_overwrite_newer_results(self) -> None:
        """The exact reported interleaving: B1=100, I1=100, I2=200. The old
        computation snapshots, then corrections swap I1->200 / I2->100, a new
        recompute publishes B1/I2, and only then does the old computation
        finish. It must neither re-list B1/I1 nor hide B1/I2."""
        upload(self.db, BANK, rows(BANK, [("B1", "甲公司", "2026-11-10", 100)]))
        upload(
            self.db,
            INTERNAL,
            rows(
                INTERNAL,
                [
                    ("I1", "甲公司", "2026-11-10", 100),
                    ("I2", "甲公司", "2026-11-10", 200),
                ],
            ),
        )
        gen1 = recompute(self.db)
        self.assertEqual(self._combos(), {("B1", ("I1",))})

        snap = threading.Event()
        release = threading.Event()
        outcome: dict = {}

        def stale_run() -> None:
            # max_attempts=1: the old computation gets one late publish attempt
            outcome["gen"] = recompute(
                self.db,
                max_attempts=1,
                after_snapshot=lambda epoch: snap.set(),
                before_publish=lambda epoch: release.wait(5),
            )

        t = threading.Thread(target=stale_run)
        t.start()
        self.assertTrue(snap.wait(5), "old recompute never took its snapshot")
        # corrections land while the old computation is still running
        upload(self.db, INTERNAL, rows(INTERNAL, [("I1", "甲公司", "2026-11-10", 200)]))
        upload(self.db, INTERNAL, rows(INTERNAL, [("I2", "甲公司", "2026-11-10", 100)]))
        gen2 = recompute(self.db)  # the new computation publishes B1/I2
        self.assertGreater(gen2, gen1)
        self.assertEqual(self._combos(), {("B1", ("I2",))})
        release.set()
        t.join(5)
        self.assertFalse(t.is_alive())
        # the late publish was discarded: old combo not resurrected, new kept
        self.assertIsNone(outcome["gen"])
        self.assertEqual(self._combos(), {("B1", ("I2",))})
        self.assertEqual(current_generation(self.db), gen2)
        self.assert_default_list_sound()
        self.assert_default_list_complete()

    def test_confirm_during_recompute_is_not_resurrected(self) -> None:
        """A combination confirmed while a recompute is in flight must not
        reappear as confirmable when that recompute finishes late."""
        upload(
            self.db,
            BANK,
            rows(
                BANK,
                [
                    ("B1", "甲公司", "2026-11-10", 100),
                    ("B2", "乙公司", "2026-11-10", 200),
                ],
            ),
        )
        upload(
            self.db,
            INTERNAL,
            rows(
                INTERNAL,
                [
                    ("I1", "甲公司", "2026-11-10", 100),
                    ("I2", "乙公司", "2026-11-10", 200),
                ],
            ),
        )
        recompute(self.db)
        target = next(
            c for c in list_candidates(self.db) if c["bank"]["serial"] == "B1"
        )

        snap = threading.Event()
        release = threading.Event()
        outcome: dict = {}

        def stale_run() -> None:
            outcome["gen"] = recompute(
                self.db,
                max_attempts=1,
                after_snapshot=lambda epoch: snap.set(),
                before_publish=lambda epoch: release.wait(5),
            )

        t = threading.Thread(target=stale_run)
        t.start()
        self.assertTrue(snap.wait(5))
        # the confirmation commits while the old computation is suspended
        group = confirm_candidate(
            self.db, target["candidate_id"], target["expected_versions"]
        )
        release.set()
        t.join(5)
        self.assertIsNone(outcome["gen"])
        # the confirmed combination stays confirmed and is not re-listed
        self.assertEqual(self._combos(), {("B2", ("I2",))})
        self.assertEqual(
            [g["group_id"] for g in list_groups(self.db)], [group["group_id"]]
        )
        # once settled, a completed recompute lists exactly the remaining set
        recompute(self.db)
        self.assertEqual(self._combos(), {("B2", ("I2",))})
        self.assert_default_list_sound()
        self.assert_default_list_complete()
        self.assert_invariant_no_half_match()

    def test_recompute_retries_and_converges_when_inputs_settle(self) -> None:
        """A recompute that loses the race re-snapshots and publishes the
        complete valid set once inputs stop changing."""
        upload(self.db, BANK, rows(BANK, [("B1", "甲公司", "2026-11-10", 100)]))
        upload(
            self.db,
            INTERNAL,
            rows(
                INTERNAL,
                [
                    ("I1", "甲公司", "2026-11-10", 100),
                    ("I2", "甲公司", "2026-11-10", 200),
                ],
            ),
        )
        recompute(self.db)

        first_attempt = threading.Event()
        release = threading.Event()
        attempts = {"n": 0}

        def gate(epoch: int) -> None:
            attempts["n"] += 1
            if attempts["n"] == 1:
                first_attempt.set()
                release.wait(5)

        outcome: dict = {}

        def run() -> None:
            # default max_attempts -> the call itself retries after the race
            outcome["gen"] = recompute(self.db, after_snapshot=gate)

        t = threading.Thread(target=run)
        t.start()
        self.assertTrue(first_attempt.wait(5))
        upload(self.db, INTERNAL, rows(INTERNAL, [("I1", "甲公司", "2026-11-10", 200)]))
        upload(self.db, INTERNAL, rows(INTERNAL, [("I2", "甲公司", "2026-11-10", 100)]))
        release.set()
        t.join(5)
        self.assertFalse(t.is_alive())
        self.assertGreaterEqual(attempts["n"], 2, "must have re-snapshotted")
        self.assertIsNotNone(outcome["gen"])
        self.assertEqual(self._combos(), {("B1", ("I2",))})
        self.assert_default_list_sound()
        self.assert_default_list_complete()

    def test_interleaved_writes_recomputes_sound_and_complete(self) -> None:
        """Uploads/corrections/confirmations interleaved with a continuous
        recompute loop: the default list stays sound at every instant, and
        once inputs settle a completed recompute lists every valid candidate.
        """
        upload(
            self.db,
            BANK,
            rows(
                BANK,
                [
                    ("B1", "甲公司", "2026-11-10", 100),
                    ("B2", "甲公司", "2026-11-11", 200),
                    ("B3", "乙公司", "2026-11-10", 300),
                ],
            ),
        )
        upload(
            self.db,
            INTERNAL,
            rows(
                INTERNAL,
                [
                    ("I1", "甲公司", "2026-11-10", 100),
                    ("I2", "甲公司", "2026-11-11", 200),
                    ("I3", "乙公司", "2026-11-10", 300),
                    ("I4", "乙公司", "2026-11-11", 150),
                ],
            ),
        )
        recompute(self.db)

        stop = threading.Event()

        def recompute_loop() -> None:
            while not stop.is_set():
                recompute(self.db)

        rt = threading.Thread(target=recompute_loop)
        rt.start()
        try:
            upload(self.db, INTERNAL,
                   rows(INTERNAL, [("I1", "甲公司", "2026-11-10", 150)]))
            self.assert_default_list_sound()
            upload(self.db, BANK,
                   rows(BANK, [("B2", "甲公司", "2026-11-11", 250)]))
            self.assert_default_list_sound()
            for _ in range(3):
                cands = list_candidates(self.db)
                if not cands:
                    break
                cand = cands[0]
                try:
                    confirm_candidate(
                        self.db, cand["candidate_id"], cand["expected_versions"]
                    )
                except (StaleCandidateError, NotFound):
                    pass  # lost a race against the recompute loop; still fine
                self.assert_default_list_sound()
            upload(self.db, INTERNAL,
                   rows(INTERNAL, [("I4", "乙公司", "2026-11-11", 100)]))
            self.assert_default_list_sound()
        finally:
            stop.set()
            rt.join(5)
        self.assertFalse(rt.is_alive())
        # inputs settled: one completed recompute -> exactly the valid set
        recompute(self.db)
        self.assert_default_list_sound()
        self.assert_default_list_complete()
        self.assert_invariant_no_half_match()

    def test_worker_retries_when_publish_discarded(self) -> None:
        """The background worker converges even when a publish is discarded
        because inputs changed mid-computation."""
        import app.recompute as rc

        worker = rc.RecomputeWorker(self.db, debounce_seconds=0.02)
        real = rc.recompute
        calls = {"n": 0}

        def flaky(db, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return None  # simulate a discarded (lost-race) publish
            return real(db, **kwargs)

        rc.recompute = flaky
        try:
            worker.start()
            upload(self.db, BANK, rows(BANK, [("B1", "甲", "2026-11-10", 100)]))
            upload(self.db, INTERNAL, rows(INTERNAL, [("I1", "甲", "2026-11-10", 100)]))
            worker.schedule()
            for _ in range(100):
                if list_candidates(self.db):
                    break
                threading.Event().wait(0.05)
            self.assertTrue(list_candidates(self.db))
            self.assertGreaterEqual(calls["n"], 2)
            self.assert_default_list_sound()
            self.assert_default_list_complete()
        finally:
            rc.recompute = real
            worker._stop.set()
            worker._wake.set()
            worker._thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
