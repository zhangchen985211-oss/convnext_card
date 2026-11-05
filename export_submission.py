"""Export one-domain or combined Lightbox/Sunlamp challenge CSV predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _load_domain_rows(prediction_path: Path, domain: str, test_json: Path):
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    expected_items = json.loads(test_json.read_text(encoding="utf-8"))
    if not isinstance(predictions, list) or not isinstance(expected_items, list):
        raise ValueError("prediction and test JSON files must contain arrays")

    pred_by_name = {}
    for item in predictions:
        filename = item["filename"]
        if filename in pred_by_name:
            raise ValueError(f"duplicate prediction for {domain}/{filename}")
        pred_by_name[filename] = item
    expected = {item["filename"] for item in expected_items}
    missing = expected.difference(pred_by_name)
    extra = set(pred_by_name).difference(expected)
    if missing or extra:
        raise ValueError(
            f"incomplete {domain} predictions: missing={len(missing)}, extra={len(extra)}"
        )

    rows = []
    for filename in sorted(expected):
        item = pred_by_name[filename]
        quaternion = [float(value) for value in item["q_vbs2tango_true"]]
        translation = [float(value) for value in item["r_Vo2To_vbs_true"]]
        if len(quaternion) != 4 or len(translation) != 3:
            raise ValueError(f"invalid pose vector lengths in {domain}/{filename}")
        if not all(math.isfinite(value) for value in quaternion + translation):
            raise ValueError(f"non-finite pose values in {domain}/{filename}")
        norm = math.sqrt(sum(value * value for value in quaternion))
        if norm <= 0.0:
            raise ValueError(f"zero quaternion in {domain}/{filename}")
        quaternion = [value / norm for value in quaternion]
        rows.append([domain, filename, *quaternion, *translation])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("lightbox", "sunlamp", "both"), required=True)
    parser.add_argument("--predictions", default="", help="single-domain prediction JSON")
    parser.add_argument("--test-json", default="", help="single-domain test.json")
    parser.add_argument("--lightbox-predictions", default="")
    parser.add_argument("--sunlamp-predictions", default="")
    parser.add_argument("--lightbox-test-json", default="")
    parser.add_argument("--sunlamp-test-json", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    if args.domain == "both":
        required = (
            args.lightbox_predictions,
            args.sunlamp_predictions,
            args.lightbox_test_json,
            args.sunlamp_test_json,
        )
        if not all(required):
            parser.error(
                "--domain both requires prediction and test JSON paths for both domains"
            )
        rows.extend(
            _load_domain_rows(
                Path(args.lightbox_predictions),
                "lightbox",
                Path(args.lightbox_test_json),
            )
        )
        rows.extend(
            _load_domain_rows(
                Path(args.sunlamp_predictions),
                "sunlamp",
                Path(args.sunlamp_test_json),
            )
        )
    else:
        if not args.predictions:
            parser.error("--predictions is required for a single domain")
        prediction_path = Path(args.predictions)
        test_json = Path(args.test_json) if args.test_json else prediction_path.with_name("test.json")
        if not test_json.is_file():
            parser.error(f"test JSON not found: {test_json}")
        rows = _load_domain_rows(prediction_path, args.domain, test_json)

    rows.sort(key=lambda row: (row[0], row[1]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fp:
        csv.writer(fp, lineterminator="\n").writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
