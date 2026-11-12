"""Candidate recomputation and its background scheduler.

Recomputation never holds a write transaction while the subset-sum search
runs: it snapshots the unmatched flows in a short read transaction, computes
on plain Python objects, then replaces the candidate table in one short write
transaction.

Publishing is *epoch-guarded*. Every input change (upload, correction,
confirmation, group disband) advances ``meta['data_epoch']`` via triggers.
A recompute records the epoch together with its snapshot and, inside the
short publish transaction, re-checks it:

* epoch unchanged  -> the snapshot still reflects the current raw values and
  claim status, so replacing the candidate table is safe and atomic;
* epoch changed    -> the snapshot is stale. The computed set is **discarded,
  never published**, and the attempt is retried from a fresh snapshot. A
  late computation can therefore never overwrite newer results, resurrect a
  combination that was confirmed or corrected in the meantime, or hide a
  newer combination.

Once inputs stop changing, the next attempt observes a stable epoch and
publishes the complete set of currently valid candidates. Uploads and
confirmations keep their own short write transactions and are never blocked
by recomputation; the confirm step still re-validates every version, so a
stale page can never confirm an outdated combination either.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Callable, Optional

from .db import Database
from .matching import compute_candidates
from .models import BANK, INTERNAL, CandidateSpec
from .repositories import load_flows

logger = logging.getLogger("reconcile.recompute")

GEN_KEY = "candidate_generation"
EPOCH_KEY = "data_epoch"

# How often a single recompute() call re-snapshots when inputs keep changing
# underneath it. Inputs eventually settle (uploads are debounced), after
# which the first quiet attempt publishes.
MAX_ATTEMPTS = 4


def _get_generation(conn) -> int:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (GEN_KEY,)).fetchone()
    return int(row["value"]) if row else 0


def _set_generation(conn, value: int) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (GEN_KEY, str(value)),
    )


def _get_epoch(conn) -> int:
    """Monotonic input epoch; advanced by triggers on every flow change."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (EPOCH_KEY,)).fetchone()
    return int(row["value"]) if row else 0


def _upsert_candidate(conn, spec: CandidateSpec, generation: int) -> None:
    conn.execute(
        """
        INSERT INTO candidates(candidate_key, bank_serial, total_amount,
                               bank_version, internal_versions, stale,
                               generation)
        VALUES (?, ?, ?, ?, ?, 0, ?)
        ON CONFLICT(candidate_key) DO UPDATE SET
            total_amount      = excluded.total_amount,
            bank_version      = excluded.bank_version,
            internal_versions = excluded.internal_versions,
            stale             = 0,
            generation        = excluded.generation
        """,
        (
            spec.candidate_key,
            spec.bank.serial,
            spec.total_amount,
            spec.bank.version,
            json.dumps(
                {f.serial: f.version for f in spec.internals},
                ensure_ascii=False,
                sort_keys=True,
            ),
            generation,
        ),
    )
    cid = conn.execute(
        "SELECT id FROM candidates WHERE candidate_key = ?", (spec.candidate_key,)
    ).fetchone()["id"]
    conn.execute("DELETE FROM candidate_items WHERE candidate_id = ?", (cid,))
    conn.execute(
        "INSERT INTO candidate_items(candidate_id, side, flow_serial, ord) "
        "VALUES (?, ?, ?, ?)",
        (cid, BANK, spec.bank.serial, 0),
    )
    for ordinal, flow in enumerate(spec.internals, start=1):
        conn.execute(
            "INSERT INTO candidate_items(candidate_id, side, flow_serial, ord) "
            "VALUES (?, ?, ?, ?)",
            (cid, INTERNAL, flow.serial, ordinal),
        )


def recompute(
    db: Database,
    *,
    max_attempts: int = MAX_ATTEMPTS,
    after_snapshot: Optional[Callable[[int], None]] = None,
    before_publish: Optional[Callable[[int], None]] = None,
) -> Optional[int]:
    """Recompute all candidates; returns the published generation.

    Returns ``None`` when inputs kept changing for ``max_attempts`` straight
    attempts and nothing was published (a later scheduled run converges).

    ``after_snapshot`` / ``before_publish`` are deterministic test seams
    invoked (outside any transaction) right after the snapshot is taken and
    right before the publish transaction opens; they let tests reproduce
    exact interleavings of recomputation with uploads and confirmations.
    """
    for _ in range(max(1, max_attempts)):
        # Snapshot (flows + input epoch) outside the write transaction.
        with db.read_tx() as conn:
            epoch = _get_epoch(conn)
            bank_snapshot = load_flows(conn, BANK)
            internal_snapshot = load_flows(conn, INTERNAL)
        if after_snapshot is not None:
            after_snapshot(epoch)
        specs = compute_candidates(bank_snapshot, internal_snapshot)
        if before_publish is not None:
            before_publish(epoch)

        with db.write_tx() as conn:
            if _get_epoch(conn) != epoch:
                # Inputs moved while computing: this snapshot is stale.
                # Discard it instead of overwriting newer results; retry
                # from a fresh snapshot below.
                continue
            generation = _get_generation(conn) + 1
            # Every candidate not produced by this snapshot becomes stale.
            conn.execute("UPDATE candidates SET stale = 1")
            for spec in specs:
                _upsert_candidate(conn, spec, generation)
            _set_generation(conn, generation)
        logger.info(
            "recomputed candidates: %d (generation %d)", len(specs), generation
        )
        return generation
    logger.info(
        "recompute deferred: inputs changed during %d attempts", max_attempts
    )
    return None


def current_generation(db: Database) -> int:
    """The generation of the candidate set currently published (0 if none)."""
    with db.read_tx() as conn:
        return _get_generation(conn)


class RecomputeWorker:
    """Debounced single-worker background recomputation."""

    def __init__(self, db: Database, debounce_seconds: float = 0.15) -> None:
        self.db = db
        self.debounce = debounce_seconds
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def schedule(self) -> None:
        self._wake.set()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name="recompute-worker", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self.db.close()

    def run_once_now(self) -> int:
        """Synchronous recomputation (used by the refresh endpoint / tests).

        Returns the generation published by this call, or the generation
        already current if inputs kept changing through every attempt.
        """
        self._wake.clear()
        published = recompute(self.db)
        return published if published is not None else current_generation(self.db)

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._wake.wait(timeout=1.0):
                continue
            # Debounce: coalesce bursts of uploads.
            while self._wake.wait(timeout=self.debounce):
                self._wake.clear()
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                published = recompute(self.db)
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("background recompute failed")
                continue
            if published is None and not self._stop.is_set():
                # Inputs moved while computing; converge on the next pass.
                self._wake.set()
