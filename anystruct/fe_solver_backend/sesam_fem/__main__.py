"""Command line helpers for SESAM formatted FEM files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .diagnostics import SesamFemError
from .document import read_sesam_fem_document
from .exporter import write_sesam_fem_document
from .importer import import_sesam_fem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m anystruct.fe_solver_backend.sesam_fem")
    parser.add_argument("--lenient", action="store_true", help="collect diagnostics instead of failing on errors")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_parser = sub.add_parser("inspect", help="show document summary")
    inspect_parser.add_argument("input")

    validate_parser = sub.add_parser("validate", help="validate a FEM file")
    validate_parser.add_argument("input")

    roundtrip_parser = sub.add_parser("roundtrip", help="canonicalize a FEM file")
    roundtrip_parser.add_argument("input")
    roundtrip_parser.add_argument("output")
    roundtrip_parser.add_argument("--overwrite", action="store_true")

    summary_parser = sub.add_parser("import-summary", help="read and optionally build a solver model")
    summary_parser.add_argument("input")
    summary_parser.add_argument("--summary-json", dest="summary_json")
    summary_parser.add_argument("--no-build-model", action="store_true")

    args = parser.parse_args(argv)
    strict = not args.lenient
    try:
        if args.command == "inspect":
            document = read_sesam_fem_document(args.input, strict=strict)
            return _print(document.summary(), args.json)
        if args.command == "validate":
            document = read_sesam_fem_document(args.input, strict=strict)
            payload = {"ok": not any(item.severity == "error" for item in document.diagnostics), "diagnostics": [item.as_dict() for item in document.diagnostics]}
            return _print(payload, args.json)
        if args.command == "roundtrip":
            document = read_sesam_fem_document(args.input, strict=strict)
            report = write_sesam_fem_document(document, args.output, overwrite=args.overwrite)
            return _print({"output": str(report.path), "records_written": report.records_written, "bytes_written": report.bytes_written}, args.json)
        if args.command == "import-summary":
            result = import_sesam_fem(args.input, strict=strict, build_model=not args.no_build_model)
            payload = result.document.summary()
            payload["element_count_by_type"] = result.element_count_by_type
            payload["model_built"] = result.model is not None
            if args.summary_json:
                Path(args.summary_json).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            return _print(payload, args.json)
    except SesamFemError as exc:
        payload = {"ok": False, "code": exc.code, "message": str(exc), "diagnostics": [item.as_dict() for item in exc.diagnostics]}
        _print(payload, True)
        return 2
    return 1


def _print(payload: object, as_json: bool) -> int:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if isinstance(payload, dict):
            for key, value in payload.items():
                print(f"{key}: {value}")
        else:
            print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
