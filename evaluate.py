"""Deterministic grader for the Ridgeline RFQ extraction pipeline.

Grades only the first three RFQs (the ones with answer keys) by comparing the
pipeline's JSON outputs in outputs/ against the answer keys in answer_key/, and
writes a results summary to evaluation/results.json.

Robustness contract:
- Never crashes on missing or malformed input files; an RFQ whose answer key is
  unreadable is skipped with a warning while the remaining RFQs still grade.
- Reads only answer_key/, outputs/, and evaluation/latency.json.
- Deterministic: the evaluated RFQs are hardcoded (no on-disk discovery) and no
  randomness is used.
"""

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ANSWER_KEY_DIR = REPO_ROOT / "answer_key"
OUTPUTS_DIR = REPO_ROOT / "outputs"
EVALUATION_DIR = REPO_ROOT / "evaluation"

# Hardcoded on purpose — do NOT discover this list from disk, so the evaluation
# surface stays fixed and deterministic.
EVALUATED = ["rfq_01_email", "rfq_02_clearwater", "rfq_03_tricounty"]

# Decision threshold for the confidence-based needs_review fallback.
# Rationale: precision/recall values observed during evaluation mostly fall in
# the 0.7-0.9 band, so 0.8 is a safe cut-off; refine it later once more
# evaluation sets exist.
THRESHOLD = 0.8


def _load_json(path):
    """Return the parsed JSON value at `path`, or None if unreadable/invalid."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return None


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_int(value):
    """Coerce a line number to int, or None when it is not an integer."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _strip(value):
    return value.strip() if isinstance(value, str) else value


def _string_equal(a, b, case_insensitive=False):
    """Null-aware string comparison with surrounding-whitespace stripping."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    sa = _strip(a)
    sb = _strip(b)
    if case_insensitive:
        sa = sa.lower() if isinstance(sa, str) else sa
        sb = sb.lower() if isinstance(sb, str) else sb
    return sa == sb


def _quantity_equal(a, b):
    """Numeric equality with 1e-9 tolerance; null-null counts as equal."""
    if a is None and b is None:
        return True
    if not _is_number(a) or not _is_number(b):
        return False
    return abs(a - b) <= 1e-9


def _as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _flags_equal(a, b):
    la = _as_list(a)
    lb = _as_list(b)
    if not la and not lb:
        return True
    return sorted(la, key=repr) == sorted(lb, key=repr)


def _needs_review(output_line):
    """Resolve an output line's needs_review flag (output_line may be None)."""
    if output_line is None:
        return True
    if "needs_review" in output_line:
        return bool(output_line["needs_review"])
    confidence = output_line.get("confidence")
    if not _is_number(confidence):
        return True
    return confidence <= THRESHOLD


def _fmt(value):
    return "null" if value is None else repr(value)


def _score_extraction(rfq_id, answer_line, output_line):
    """Score one aligned answer-key line for extraction accuracy.

    Returns (expected_fields, correct_fields, mismatch_strings). A missing
    output line scores every expected field as incorrect.
    """
    missing = output_line is None

    def out(field):
        return None if missing else output_line.get(field)

    expected = 0
    correct = 0
    mismatches = []
    line_no = answer_line.get("line")

    checks = [
        ("as_written",
         answer_line.get("as_written"), out("as_written"),
         _string_equal(answer_line.get("as_written"), out("as_written"))),
        ("quantity",
         answer_line.get("quantity"), out("quantity"),
         _quantity_equal(answer_line.get("quantity"), out("quantity"))),
        ("unit",
         answer_line.get("unit"), out("unit"),
         _string_equal(answer_line.get("unit"), out("unit"), case_insensitive=True)),
        ("expected_part_number",
         answer_line.get("expected_part_number"), out("ridgeline_part_number"),
         _string_equal(answer_line.get("expected_part_number"), out("ridgeline_part_number"))),
        ("match_method",
         answer_line.get("match_method"), out("match_method"),
         _string_equal(answer_line.get("match_method"), out("match_method"))),
        ("flags",
         answer_line.get("flags"), out("flags"),
         _flags_equal(answer_line.get("flags"), out("flags"))),
    ]

    for field, av, ov, ok in checks:
        expected += 1
        if not missing and ok:
            correct += 1
        else:
            mismatches.append(f"{rfq_id}:{line_no} {field} expected={_fmt(av)} got={_fmt(ov)}")

    if "note" in answer_line:
        expected += 1
        note = out("note")
        if not missing and isinstance(note, str) and note.strip() != "":
            correct += 1
        else:
            mismatches.append(
                f"{rfq_id}:{line_no} note expected={_fmt(answer_line.get('note'))} got={_fmt(note)}"
            )

    return expected, correct, mismatches


def _score_rfq(rfq_id):
    """Grade one RFQ. Raises ValueError only when its answer key is unusable."""
    key_data = _load_json(ANSWER_KEY_DIR / f"{rfq_id}.json")
    if not isinstance(key_data, dict) or not isinstance(key_data.get("lines"), list):
        raise ValueError(f"answer key for {rfq_id} is missing or malformed")
    key_lines = key_data["lines"]

    out_data = _load_json(OUTPUTS_DIR / f"{rfq_id}.json")
    if isinstance(out_data, dict) and isinstance(out_data.get("lines"), list):
        out_lines = out_data["lines"]
    else:
        print(
            f"warning: outputs/{rfq_id}.json missing or malformed — treating as no output lines",
            file=sys.stderr,
        )
        out_lines = []

    out_by_line = {}
    matched_lines = 0
    cross_reference_lines = 0
    for ol in out_lines:
        if not isinstance(ol, dict):
            continue
        if ol.get("ridgeline_part_number") is not None:
            matched_lines += 1
            if _string_equal(ol.get("match_method"), "cross_reference"):
                cross_reference_lines += 1
        ln = _as_int(ol.get("line"))
        if ln is not None and ln not in out_by_line:
            out_by_line[ln] = ol

    answer_line_nums = [
        _as_int(al.get("line")) if isinstance(al, dict) else None for al in key_lines
    ]
    answer_line_set = {ln for ln in answer_line_nums if ln is not None}

    expected_total = 0
    correct_total = 0
    mismatches = []
    tp = fp = fn = 0

    for idx, al in enumerate(key_lines):
        if not isinstance(al, dict):
            continue
        ln = answer_line_nums[idx]
        output_line = out_by_line.get(ln) if ln is not None else None

        exp, corr, ms = _score_extraction(rfq_id, al, output_line)
        expected_total += exp
        correct_total += corr
        mismatches.extend(ms)

        expected_part = al.get("expected_part_number")
        predicted = output_line.get("ridgeline_part_number") if output_line is not None else None
        nr = _needs_review(output_line)
        committed = predicted is not None and nr is False
        part_match = _string_equal(expected_part, predicted)
        committed_correct = committed and part_match

        if committed_correct:
            tp += 1
        if committed and not part_match:
            fp += 1
        if expected_part is not None and not committed_correct:
            fn += 1

    # Extra output lines (no answer-key counterpart) contribute FP when committed.
    for ol in out_lines:
        if not isinstance(ol, dict):
            continue
        ln = _as_int(ol.get("line"))
        if ln is None or ln not in answer_line_set:
            if ol.get("ridgeline_part_number") is not None and not _needs_review(ol):
                fp += 1

    return {
        "rfq_id": rfq_id,
        "answer_lines": sum(1 for al in key_lines if isinstance(al, dict)),
        "output_lines": sum(1 for ol in out_lines if isinstance(ol, dict)),
        "expected_fields": expected_total,
        "correct_fields": correct_total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched_lines": matched_lines,
        "cross_reference_lines": cross_reference_lines,
        "mismatches": mismatches,
    }


def _read_latency():
    """Read evaluation/latency.json, keeping numeric entries for EVALUATED ids."""
    data = _load_json(EVALUATION_DIR / "latency.json")
    if data is None:
        print(
            "warning: evaluation/latency.json missing or unreadable — latency set to empty",
            file=sys.stderr,
        )
        return {}
    if not isinstance(data, dict):
        print(
            "warning: evaluation/latency.json is not an object — latency set to empty",
            file=sys.stderr,
        )
        return {}
    result = {}
    for key in EVALUATED:
        value = data.get(key)
        if _is_number(value) and math.isfinite(value):
            result[key] = float(value)
    return result


def _build_results():
    extraction_correct = 0
    extraction_expected = 0
    tp = fp = fn = 0
    matched_lines = 0
    cross_reference_lines = 0
    all_mismatches = []
    per_rfq = []

    for rfq_id in EVALUATED:
        try:
            r = _score_rfq(rfq_id)
        except Exception as exc:
            print(f"warning: skipping {rfq_id}: {exc}", file=sys.stderr)
            continue
        extraction_expected += r["expected_fields"]
        extraction_correct += r["correct_fields"]
        tp += r["tp"]
        fp += r["fp"]
        fn += r["fn"]
        matched_lines += r["matched_lines"]
        cross_reference_lines += r["cross_reference_lines"]
        all_mismatches.extend(r["mismatches"])
        per_rfq.append(r)

    extraction_accuracy = extraction_correct / extraction_expected if extraction_expected else 0.0
    match_precision = tp / (tp + fp) if (tp + fp) else 0.0
    match_recall = tp / (tp + fn) if (tp + fn) else 0.0
    cross_reference_hit_rate = cross_reference_lines / matched_lines if matched_lines else 0.0

    latency = _read_latency()
    results = {
        "threshold": float(THRESHOLD),
        "extraction_accuracy": extraction_accuracy,
        "match_precision": match_precision,
        "match_recall": match_recall,
        "additional_metrics": {
            "cross_reference_hit_rate": cross_reference_hit_rate,
            "latency": {
                "total_seconds": float(round(sum(latency.values()), 4)),
                "per_rfq_seconds": dict(latency),
            },
        },
    }
    return results, per_rfq, all_mismatches


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Grade the Ridgeline RFQ extraction pipeline on the first three RFQs."
    )
    parser.add_argument(
        "--output",
        default=str(EVALUATION_DIR / "results.json"),
        help="Where to write the results JSON (default: evaluation/results.json)",
    )
    args = parser.parse_args(argv)

    try:
        results, per_rfq, mismatches = _build_results()
    except Exception as exc:  # defensive: never let the script crash
        print(f"error: evaluation failed: {exc}", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        print(f"error: could not write {out_path}: {exc}", file=sys.stderr)
        return 1

    am = results["additional_metrics"]
    print("Evaluation results:")
    print(f"  threshold:                {results['threshold']}")
    print(f"  extraction_accuracy:      {results['extraction_accuracy']}")
    print(f"  match_precision:          {results['match_precision']}")
    print(f"  match_recall:             {results['match_recall']}")
    print(f"  cross_reference_hit_rate: {am['cross_reference_hit_rate']}")
    print(f"  latency total_seconds:    {am['latency']['total_seconds']}")
    print()
    print("Per-RFQ line counts:")
    for r in per_rfq:
        print(
            f"  {r['rfq_id']}: {r['answer_lines']} answer-key lines, "
            f"{r['output_lines']} output lines"
        )
    if not per_rfq:
        print("  (no RFQs graded)")
    print()
    if mismatches:
        print(f"Line-level mismatches ({len(mismatches)}):")
        for m in mismatches:
            print(f"  {m}")
    else:
        print("Line-level mismatches: none")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
