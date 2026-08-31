"""Turn a folder of raw camera exports into a named dataset (BUILD_SPEC Phase 6b).

Scans arrive from a phone with names that carry no information —
``WhatsApp Image 2026-06-01 at 11.43.41 AM (1).jpeg``,
``039d23c3-60ef-4158-a48a-6fd7aae48d4d.JPG``. Two scripts downstream need to
know which side of the document they are looking at, and both read it out of the
filename: ``label_ocr.detect_side`` picks the field set from it, and
``evaluate_ocr`` picks the template from it. So the raw set has to be classified
and renamed before any of it is usable.

This script does that in two steps, on purpose:

1. **Report** (default) — reads every scan, works out the document side, looks
   for duplicates, and writes a plan to ``datasets/ingest_plan.csv``. Nothing on
   disk is touched. Rows the classifier is not sure about are marked
   ``confidence=low`` so you know exactly which ones to eyeball.
2. **Apply** (``--apply``) — replays that CSV. Because the plan is a file you can
   edit, correcting a misclassified scan means fixing one cell, not patching the
   classifier.

Renaming never re-encodes: the JPEG bytes are moved untouched, because the fraud
detectors read compression artifacts straight out of them.

How the side is decided
-----------------------
**Proposal** — the back page has a two-column table and the front does not, so
the normalized mass of interior vertical rules separates them outright. Measured
over 137 scans: every front is exactly ``0.000``, the lowest back is ``0.170``.

**NIC** — harder, because a card is photographed lying on a table at an
arbitrary angle and sometimes fills a tenth of the frame. The front carries a
face photo and the back a barcode (PDF417 on older cards, QR on the current
one), so the signals are a Haar face detector run at four quarter turns and the
peak local ink density of a barcode block. Measured over 38 scans: 27 land in
the high-confidence branches and all 27 are right; the remaining 11 are flagged
for review, and exactly one of those (a laminated card behind heavy glare) is
guessed wrong. Treat the NIC side column as a draft, not an answer.

Usage::

    python -m ml_training.ingest_raw                  # report + write the plan
    python -m ml_training.ingest_raw --doc-kind nic    # one kind only
    python -m ml_training.ingest_raw --apply           # replay the plan
    python -m ml_training.ingest_raw --undo            # put the old names back
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import itertools
import json
import re
from pathlib import Path

import cv2
import numpy as np

from . import config

PLAN_PATH = config.DATASETS_DIR / "ingest_plan.csv"
MANIFEST_PATH = config.DATASETS_DIR / "ingest_manifest.csv"
# Rejected scans are moved here rather than deleted. config.iter_raw_images only
# yields files, so a subfolder drops out of the dataset without being lost.
REJECTED_DIRNAME = "_rejected"

PLAN_COLUMNS = (
    "action",
    "original",
    "proposed",
    "doc_kind",
    "side",
    "confidence",
    "evidence",
    "group",
    "rotate",
    "note",
)

# csv parks values past the last column under this key instead of ``None``, so
# read_plan can name the offending line rather than fail later in DictWriter.
_OVERFLOW = "_overflow"

# The name this script produces, and recognizes as "already done" on a re-run.
CANONICAL_RE = re.compile(r"^(nic|proposal)-(\d{3})-(front|back)$")

# --- Classifier thresholds --------------------------------------------------
# Interior vertical-rule mass. Fronts measured 0.000 (all 68 of them), backs
# 0.170-0.844, so anything in between is a wide no-man's land.
PROPOSAL_VMID_SPLIT = 0.10
# Peak local ink fraction over a 90x90 window. A PDF417 block sustains ~0.50;
# dense trilingual print on a card front peaks around 0.46.
NIC_INK_BARCODE = 0.48
# Above this, a detected face is no longer enough on its own to call it a front.
NIC_INK_FRONT_CEILING = 0.45
# dHash distance (out of 256 bits) below which two scans are the same document.
# Measured: 8 for a recompressed copy, 21 for a second photo of the same card,
# and nothing else in the set below 40.
DHASH_NEAR = 24
# Mean absolute pixel difference below which two same-sized scans are the same
# capture saved twice, not two photographs. Measured: 0.3 for a recompressed
# copy against its original, 12.8 for an unrelated pair.
REENCODE_MAD = 2.0

_face_cascade_cache = None


# --- I/O --------------------------------------------------------------------
def _read(path: Path) -> np.ndarray:
    """Read an image via bytes, so non-ASCII Windows paths work."""
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    """Path relative to the raw scan root, as posix — the plan's row key."""
    return path.relative_to(config.RAW_DIR).as_posix()


# --- Fingerprints -----------------------------------------------------------
def dhash(bgr: np.ndarray, size: int = 16) -> np.ndarray:
    """Difference hash of the upright grayscale image, as a bit array."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] > gray.shape[0]:
        gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    gray = cv2.resize(gray, (size + 1, size))
    return (gray[:, 1:] > gray[:, :-1]).flatten()


def _hamming(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero(left ^ right))


def _mean_abs_diff(left: Path, right: Path) -> float | None:
    """Mean absolute pixel difference, or ``None`` if the shapes differ."""
    a = _read(left).astype(np.float32)
    b = _read(right).astype(np.float32)
    if a.shape != b.shape:
        return None
    return float(np.abs(a - b).mean())


# --- Side classification: proposal ------------------------------------------
def proposal_rule_mass(bgr: np.ndarray) -> float:
    """Normalized mass of *interior* vertical rules on an upright page.

    The back page of the proposal form is a two-column table; the front is a
    single column of fields. Isolating long vertical strokes with a tall thin
    structuring element and measuring how much of that mass sits in the middle
    third of the width therefore separates the two layouts outright.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] > gray.shape[0]:
        gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    gray = cv2.resize(gray, (600, 800))

    binary = cv2.adaptiveThreshold(
        cv2.bitwise_not(gray), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical = cv2.dilate(cv2.erode(binary, kernel), kernel)

    profile = vertical.sum(axis=0).astype(np.float64)
    total = profile.sum()
    if total <= 0:
        return 0.0
    return float(profile[210:450].sum() / total)


def proposal_side(bgr: np.ndarray) -> tuple[str, str, str]:
    """Return ``(side, confidence, evidence)`` for a proposal page."""
    mass = proposal_rule_mass(bgr)
    side = "back" if mass >= PROPOSAL_VMID_SPLIT else "front"
    # The measured gap is 0.000 vs 0.170, so only a value actually inside the
    # gap is worth a human's time.
    margin = abs(mass - PROPOSAL_VMID_SPLIT)
    confidence = "high" if margin >= 0.06 else "low"
    return side, confidence, f"rule_mass={mass:.3f}"


# --- Side classification: NIC -----------------------------------------------
def _face_cascade():
    global _face_cascade_cache
    if _face_cascade_cache is None:
        _face_cascade_cache = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade_cache


def face_hits(gray: np.ndarray) -> int:
    """Count Haar face detections across all four quarter turns.

    A card lies on the table at whatever angle the photographer held it, so the
    portrait photo on the front can be upright, sideways, or upside down.
    """
    cascade = _face_cascade()
    small = cv2.resize(gray, None, fx=0.6, fy=0.6)
    hits = 0
    for turn in range(4):
        candidate = small if turn == 0 else np.rot90(small, turn).copy()
        hits += len(cascade.detectMultiScale(candidate, 1.1, 5, minSize=(24, 24)))
    return hits


def barcode_ink_peak(gray: np.ndarray, window: int = 90, stride: int = 15) -> float:
    """Highest ink fraction found in any ``window``-sized square.

    A PDF417 block is close to half black over a large area, which no run of
    printed text sustains. Computed from an integral image so the sliding window
    costs one pass.
    """
    binary = cv2.adaptiveThreshold(
        cv2.bitwise_not(gray), 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -8
    )
    integral = cv2.integral(binary.astype(np.float32))
    rows = np.arange(0, max(binary.shape[0] - window, 1), stride)
    cols = np.arange(0, max(binary.shape[1] - window, 1), stride)
    if not rows.size or not cols.size:
        return 0.0

    bottom, right = rows + window, cols + window
    sums = (
        integral[np.ix_(bottom, right)]
        - integral[np.ix_(rows, right)]
        - integral[np.ix_(bottom, cols)]
        + integral[np.ix_(rows, cols)]
    )
    return float(sums.max() / (window * window))


def nic_side(bgr: np.ndarray) -> tuple[str, str, str]:
    """Return ``(side, confidence, evidence)`` for one side of an NIC card."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    scale = 900 / max(gray.shape[:2])
    gray = cv2.resize(gray, None, fx=scale, fy=scale)

    faces = face_hits(gray)
    ink = barcode_ink_peak(gray)
    evidence = f"faces={faces} ink={ink:.3f}"

    # A face and no barcode-grade ink: a front, and nothing else looks like this.
    if faces and ink < NIC_INK_FRONT_CEILING:
        return "front", "high", evidence
    # A barcode block and no face: a back.
    if not faces and ink >= NIC_INK_BARCODE:
        return "back", "high", evidence
    # Both signals firing, or neither. Guess, but say so.
    if faces:
        return ("back" if ink >= NIC_INK_BARCODE else "front"), "low", evidence
    return "back", "low", evidence


SIDE_CLASSIFIERS = {"nic": nic_side, "proposal": proposal_side}


# --- Survey -----------------------------------------------------------------
def survey(doc_kinds) -> list[dict]:
    """Measure every raw scan once. Each scan is decoded a single time here."""
    records: list[dict] = []
    for kind in doc_kinds:
        folder = config.RAW_DIR / kind
        if not folder.is_dir():
            print(f"  (no folder {folder})")
            continue

        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in config.IMAGE_EXTENSIONS:
                continue

            canonical = CANONICAL_RE.match(path.stem)
            bgr = _read(path)
            height, width = bgr.shape[:2]

            if canonical:
                # Already named by an earlier run: trust the name, do not
                # re-classify, and reserve the number.
                side, confidence, evidence = canonical.group(3), "named", ""
                number = int(canonical.group(2))
            else:
                side, confidence, evidence = SIDE_CLASSIFIERS[kind](bgr)
                number = None

            records.append(
                {
                    "kind": kind,
                    "path": path,
                    "number": number,
                    "side": side,
                    "confidence": confidence,
                    "evidence": evidence,
                    "landscape": width > height,
                    "size": path.stat().st_size,
                    "digest": _sha256(path),
                    "bits": dhash(bgr),
                }
            )
    return records


# --- Duplicate detection ----------------------------------------------------
def find_duplicates(records: list[dict]) -> None:
    """Annotate records in place with ``group`` and ``duplicate_of``.

    Three different relationships, deliberately handled differently:

    * identical bytes — one file, copied. Reject the copies.
    * same pixels, different bytes — a scan re-saved by a chat app or an editor.
      Reject the re-encodes: recompression is exactly what the ELA and CNN
      detectors look for, so a re-saved genuine document trains them to call
      genuine documents fake. Keep the largest file, which is the least
      recompressed.
    * similar but not identical — two photographs of the same document. Keep
      both (they are genuinely different captures) but group them, so the
      train/validation split can hold them on the same side.

    Best-effort by design: a second photo taken at a very different angle or
    exposure will not be caught here. The grouping is a starting point for the
    review step, not a guarantee.
    """
    for record in records:
        record.setdefault("group", "")
        record.setdefault("duplicate_of", None)

    by_kind: dict[str, list[dict]] = {}
    for record in records:
        by_kind.setdefault(record["kind"], []).append(record)

    for kind, group in by_kind.items():
        # Union-find over "same document" edges, so a cluster of three joins up.
        parent = {id(r): id(r) for r in group}
        lookup = {id(r): r for r in group}

        def find(key):
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for left, right in itertools.combinations(group, 2):
            if left["digest"] == right["digest"]:
                left["_exact"] = True
                right["_exact"] = True
                union(id(left), id(right))
                continue
            if _hamming(left["bits"], right["bits"]) > DHASH_NEAR:
                continue
            union(id(left), id(right))
            difference = _mean_abs_diff(left["path"], right["path"])
            if difference is not None and difference < REENCODE_MAD:
                # Same capture, saved twice. Reject whichever has fewer bytes.
                loser = left if left["size"] < right["size"] else right
                winner = right if loser is left else left
                loser["duplicate_of"] = winner
                loser["reason"] = (
                    f"re-encode of {_relative(winner['path'])} "
                    f"(mean pixel diff {difference:.2f})"
                )

        clusters: dict[int, list[dict]] = {}
        for record in group:
            clusters.setdefault(find(id(record)), []).append(record)
        index = 0
        for members in clusters.values():
            if len(members) < 2:
                continue
            index += 1
            label = f"{kind}-dup{index:02d}"
            for record in members:
                record["group"] = label

        # An exact byte copy has no reason to survive at all.
        for members in clusters.values():
            exact: dict[str, dict] = {}
            for record in sorted(members, key=lambda r: r["path"].name):
                first = exact.setdefault(record["digest"], record)
                if first is not record and record.get("duplicate_of") is None:
                    record["duplicate_of"] = first
                    record["reason"] = f"identical bytes to {_relative(first['path'])}"


# --- Plan -------------------------------------------------------------------
def build_plan(records: list[dict]) -> list[dict]:
    """Assign canonical names and turn the survey into plan rows."""
    find_duplicates(records)

    # Numbers already taken by a previous run stay taken, so re-running after
    # adding a few scans does not renumber (and so rewrite) the whole folder.
    next_number = {}
    for kind in config.DOC_KINDS:
        used = [r["number"] for r in records if r["kind"] == kind and r["number"]]
        next_number[kind] = (max(used) + 1) if used else 1

    rows: list[dict] = []
    for record in records:
        path, kind = record["path"], record["kind"]
        original = _relative(path)

        if record["duplicate_of"] is not None:
            rows.append(
                {
                    "action": "reject",
                    "original": original,
                    "proposed": f"{kind}/{REJECTED_DIRNAME}/{path.name}",
                    "doc_kind": kind,
                    "side": record["side"],
                    "confidence": record["confidence"],
                    "evidence": record["evidence"],
                    "group": record["group"],
                    "rotate": "",
                    "note": record.get("reason", "duplicate"),
                }
            )
            continue

        if record["number"] is not None:
            rows.append(
                {
                    "action": "skip",
                    "original": original,
                    "proposed": original,
                    "doc_kind": kind,
                    "side": record["side"],
                    "confidence": record["confidence"],
                    "evidence": "",
                    "group": record["group"],
                    "rotate": "",
                    "note": "already named",
                }
            )
            continue

        number = next_number[kind]
        next_number[kind] += 1
        name = f"{kind}-{number:03d}-{record['side']}{path.suffix.lower()}"

        notes = []
        if record["landscape"]:
            # Nothing downstream reads a per-file rotation, so this is a warning
            # for a human, not an instruction: an upright template will not line
            # up with a sideways page.
            notes.append("landscape - quarter turn unknown, not upright")
        rows.append(
            {
                "action": "rename",
                "original": original,
                "proposed": f"{kind}/{name}",
                "doc_kind": kind,
                "side": record["side"],
                "confidence": record["confidence"],
                "evidence": record["evidence"],
                "group": record["group"],
                "rotate": "" if record["landscape"] else "0",
                "note": "; ".join(notes),
            }
        )
    return rows


def write_plan(rows: list[dict], path: Path = PLAN_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_plan(path: Path = PLAN_PATH) -> list[dict]:
    """Load a plan or manifest CSV, refusing any row csv could not read cleanly.

    The plan exists to be hand-edited, and the likeliest edit mistake is a bare
    comma inside ``note`` or ``evidence``: csv then parks the overflow under a
    ``None`` key, and the row silently carries a field ``DictWriter`` cannot
    emit. Validating here costs nothing; discovering it after ``apply_plan`` has
    moved 175 files costs the undo manifest, which is exactly what happened once.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, restkey=_OVERFLOW, restval="")
        if reader.fieldnames != list(PLAN_COLUMNS):
            raise ValueError(
                f"{path.name}: header is {reader.fieldnames}, "
                f"expected {list(PLAN_COLUMNS)}"
            )

        rows: list[dict] = []
        # start=2 because line 1 is the header. Off by one if a quoted cell holds
        # a newline, which nothing here writes.
        for line, row in enumerate(reader, start=2):
            overflow = row.pop(_OVERFLOW, None)
            if overflow:
                raise ValueError(
                    f"{path.name} line {line}: {len(PLAN_COLUMNS) + len(overflow)} "
                    f"values for {len(PLAN_COLUMNS)} columns. A cell almost "
                    f"certainly contains an unquoted comma; the text that spilled "
                    f"past the last column is {overflow}. Wrap that cell in double "
                    "quotes, or drop the comma."
                )
            if row["action"] not in ("rename", "reject", "skip"):
                raise ValueError(
                    f"{path.name} line {line}: action is {row['action']!r}, "
                    "expected rename, reject or skip"
                )
            rows.append(row)
    return rows


def report(rows: list[dict]) -> None:
    """Print what the plan says, loudest thing first."""
    renames = [r for r in rows if r["action"] == "rename"]
    rejects = [r for r in rows if r["action"] == "reject"]
    skips = [r for r in rows if r["action"] == "skip"]

    print(f"\n{len(rows)} scan(s): {len(renames)} to rename, "
          f"{len(rejects)} to reject, {len(skips)} already named")

    for kind in config.DOC_KINDS:
        kind_rows = [r for r in renames if r["doc_kind"] == kind]
        if not kind_rows:
            continue
        front = sum(1 for r in kind_rows if r["side"] == "front")
        back = len(kind_rows) - front
        low = sum(1 for r in kind_rows if r["confidence"] == "low")
        print(f"  {kind:<9} {len(kind_rows):>4}  front={front:<4} back={back:<4} "
              f"needs review={low}")

    if rejects:
        print("\nREJECT (moved aside, not deleted):")
        for row in rejects:
            print(f"  {row['original']}\n      {row['note']}")

    low_rows = [r for r in rows if r["confidence"] == "low" and r["action"] == "rename"]
    if low_rows:
        print(f"\nNEEDS REVIEW - {len(low_rows)} row(s) the classifier is unsure of.")
        print("Open the plan, check the scan, and fix the 'side' cell if it is wrong")
        print("(the proposed filename is regenerated from it on --apply).")
        for row in low_rows:
            print(f"  {row['side']:<5} {row['evidence']:<22} {row['original']}")

    landscape = [r for r in renames if not r["rotate"]]
    if landscape:
        print(f"\nNOT UPRIGHT - {len(landscape)} scan(s) are landscape. Fine for fraud")
        print("training; an upright OCR template will not align to them.")

    groups = sorted({r["group"] for r in rows if r["group"]})
    if groups:
        print(f"\nSAME-DOCUMENT GROUPS - {len(groups)}:")
        for label in groups:
            members = [r["original"] for r in rows if r["group"] == label]
            print(f"  {label}: " + ", ".join(members))
        print("Best-effort: a second photo at a very different angle is not caught.")

    print(f"\nPlan written to {PLAN_PATH.relative_to(config.ROOT_DIR).as_posix()}")
    print("Review it, then:  python -m ml_training.ingest_raw --apply")


# --- Apply ------------------------------------------------------------------
def _resolve_targets(rows: list[dict]) -> list[dict]:
    """Regenerate ``proposed`` from an edited ``side`` cell and validate.

    The whole point of the review step is that you fix ``side``; the filename
    has to follow, or a corrected row would keep the wrong name.
    """
    resolved = []
    for row in rows:
        row = dict(row)
        if row["action"] == "rename":
            stem, suffix = row["proposed"].rsplit(".", 1)
            kind, number, _ = stem.split("/", 1)[1].split("-", 2)
            row["proposed"] = f"{kind}/{kind}-{number}-{row['side']}.{suffix}"
        resolved.append(row)

    seen: dict[str, str] = {}
    for row in resolved:
        if row["action"] == "skip":
            continue
        if row["side"] not in ("front", "back"):
            raise ValueError(f"{row['original']}: side must be front or back")
        clash = seen.get(row["proposed"])
        if clash:
            raise ValueError(
                f"two scans want the same name {row['proposed']}: {clash}, {row['original']}"
            )
        seen[row["proposed"]] = row["original"]
    return resolved


def apply_plan(rows: list[dict]) -> dict[str, str]:
    """Move files per the plan. Returns the old -> new mapping that was applied."""
    rows = _resolve_targets(rows)

    moves: list[tuple[dict, Path, Path]] = []
    for row in rows:
        if row["action"] == "skip":
            continue
        source = config.RAW_DIR / row["original"]
        target = config.RAW_DIR / row["proposed"]
        if not source.is_file():
            raise FileNotFoundError(f"plan lists a scan that is gone: {source}")
        if target.exists() and target.resolve() != source.resolve():
            raise FileExistsError(f"target already exists: {target}")
        moves.append((row, source, target))

    applied: dict[str, str] = {}
    with _manifest_writer() as record:
        for row, source, target in moves:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            record(row)
            applied[_relative(source)] = _relative(target)
            print(f"  {_relative(source)}\n    -> {_relative(target)}")
    return applied


@contextlib.contextmanager
def _manifest_writer():
    """Yield a single-row append that is flushed the moment it is called.

    Recording each move as it happens, instead of the whole plan once the loop
    finishes, keeps the manifest true to what is on disk even if the run dies
    halfway through: a batch write at the end can leave every file renamed and
    no record of where any of it came from. ``extrasaction="ignore"`` is the
    second belt — no stray key can strand the manifest behind completed moves.
    """
    exists = MANIFEST_PATH.is_file()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=PLAN_COLUMNS, extrasaction="ignore"
        )
        if not exists:
            writer.writeheader()

        def record(row: dict) -> None:
            writer.writerow(row)
            handle.flush()

        yield record


def undo() -> dict[str, str]:
    """Reverse every recorded move, most recent first."""
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"nothing to undo: no {MANIFEST_PATH}")

    rows = read_plan(MANIFEST_PATH)

    # A manifest that does not account for every canonically named scan cannot
    # restore the folder, and silently reversing the part it does cover would
    # look like success. Say so before touching anything.
    recorded = {row["proposed"] for row in rows}
    orphans = [
        _relative(path)
        for _, path in config.iter_raw_images()
        if CANONICAL_RE.match(path.stem) and _relative(path) not in recorded
    ]
    if orphans:
        print(f"WARNING: {len(orphans)} scan(s) carry a canonical name that this")
        print("manifest does not record, so their original filenames are not")
        print("recoverable from it. They are left exactly as they are:")
        for rel in orphans[:5]:
            print(f"  {rel}")
        if len(orphans) > 5:
            print(f"  ... and {len(orphans) - 5} more")
        print()

    reverted: dict[str, str] = {}
    for row in reversed(rows):
        source = config.RAW_DIR / row["proposed"]
        target = config.RAW_DIR / row["original"]
        if not source.is_file():
            print(f"  skip (already moved): {row['proposed']}")
            continue
        if target.exists():
            print(f"  skip (original name taken): {row['original']}")
            continue
        source.rename(target)
        reverted[row["proposed"]] = row["original"]
        print(f"  {row['proposed']}\n    -> {row['original']}")

    MANIFEST_PATH.unlink()
    print(f"\nRemoved {MANIFEST_PATH.name}. Re-run without --apply to plan again.")
    return reverted


# --- Keep templates and labels pointing at the right scan -------------------
def _image_targets(rows: list[dict]) -> dict[str, str]:
    """Map every moved scan to the repo-relative path that should replace it.

    A rejected re-encode maps to the twin that survived, not to its own new
    location: whatever referenced the duplicate meant the *document*, and the
    surviving file is the same document with better bytes.
    """
    prefix = config.RAW_DIR.relative_to(config.ROOT_DIR).as_posix()
    kept_by_group: dict[str, str] = {}
    for row in rows:
        if row["action"] == "rename" and row["group"]:
            kept_by_group.setdefault(row["group"], row["proposed"])

    mapping: dict[str, str] = {}
    for row in rows:
        if row["action"] == "skip":
            continue
        destination = row["proposed"]
        if row["action"] == "reject":
            destination = kept_by_group.get(row["group"], row["proposed"])
        mapping[f"{prefix}/{row['original']}"] = f"{prefix}/{destination}"
    return mapping


def repair_references(mapping: dict[str, str]) -> None:
    """Rewrite the scan paths recorded in templates and OCR label files.

    Both point at a raw scan by path, and a rename breaks them silently — the
    template's ``reference_image`` is what alignment warps against, and a label
    file names the scan it describes. The mapping is exact, so this is
    mechanical rather than a guess.
    """
    for path in sorted(config.TEMPLATES_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        current = data.get("reference_image")
        if current in mapping:
            data["reference_image"] = mapping[current]
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(f"  template {path.name}: reference_image -> {mapping[current]}")

    for path in sorted(config.OCR_LABELS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        current = data.get("image")
        if current not in mapping:
            continue
        data["image"] = mapping[current]
        data["side"] = "back" if "back" in Path(mapping[current]).stem else "front"
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        # The label's own filename embeds the scan stem, so it moves too.
        target = path.with_name(
            f"{data.get('doc_kind', 'unknown')}__{Path(mapping[current]).stem}.json"
        )
        if target == path:
            print(f"  label {path.name}: image -> {mapping[current]}")
        elif target.exists():
            print(f"  label {path.name}: image updated, but {target.name} exists - "
                  "merge these two by hand")
        else:
            path.rename(target)
            print(f"  label {path.name} -> {target.name}")


# --- CLI --------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--doc-kind",
        action="append",
        dest="doc_kinds",
        choices=config.DOC_KINDS,
        help="Limit to one document kind (repeatable).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Replay the plan CSV: rename the files and write the manifest.",
    )
    parser.add_argument(
        "--undo",
        action="store_true",
        help="Reverse the recorded moves and put the original filenames back.",
    )
    args = parser.parse_args()

    config.ensure_dirs()

    if args.undo:
        mapping = undo()
        if mapping:
            repair_references(mapping)
        return

    if args.apply:
        if not PLAN_PATH.is_file():
            parser.error(
                f"no plan at {PLAN_PATH}. Run without --apply first, review it, "
                "then re-run with --apply."
            )
        rows = read_plan()
        # Build the reference mapping before the files move: it is derived from
        # the plan, not from what is on disk.
        mapping = _image_targets(_resolve_targets(rows))
        apply_plan(rows)
        repair_references(mapping)
        print(f"\nManifest: {MANIFEST_PATH.relative_to(config.ROOT_DIR).as_posix()}")
        print("Next:  python -m ml_training.label_ocr --status")
        return

    kinds = args.doc_kinds or list(config.DOC_KINDS)
    print(f"Reading scans under {config.RAW_DIR} ...")
    records = survey(kinds)
    if not records:
        print("No scans found.")
        return
    rows = build_plan(records)
    write_plan(rows)
    report(rows)


if __name__ == "__main__":
    main()
