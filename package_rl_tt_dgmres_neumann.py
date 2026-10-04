"""Package a completed Neumann experiment and every declared development pilot.

Standard library plus the standalone evidence auditor only; no numerical package
is imported. A new ZIP is the sole output and its receipt is printed. Existing
files are never replaced. Run only after numerical stages have stopped.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode = True  # Packaging creates only its requested new ZIP.
from audit_rl_tt_dgmres_neumann import Auditor, digest, read_json, require


PACKAGES = ("rl_tt_dgmres_neumann", "rl_tt_dgmres_v1", "rl_tt_dgmres_v2",
            "rl_tt_dgmres_v3", "adaptive_tt_dgmres")
REQUIRED_PILOTS = {
    "neumann_development_pilot_20261001", "neumann_rank256_pilot_20261001",
    "neumann_rank512_pilot_20261001", "neumann_threads4_pilot_20261001",
}
HELPERS = ("audit_rl_tt_dgmres_neumann.py", "run_trained_neumann.py",
           "pilot_neumann_training.py", "package_rl_tt_dgmres_neumann.py")
MANIFEST_NAME = "DELIVERY_MANIFEST.json"
AUDIT_NAME = "INDEPENDENT_AUDIT_NEUMANN.json"


def workspace_path(root, value, *, directory=False):
    path = Path(value).resolve()
    require(root in path.parents, "Path must be strictly inside the workspace: " + str(path))
    if directory:
        require(path.is_dir(), "Required directory is missing: " + str(path))
    return path


def tree_files(root, folder, *, code=False):
    """Reject aliases; include all evidence files, not just successful rows."""
    files = set()
    for current, dirs, names in os.walk(folder, followlinks=False):
        current = Path(current)
        for name in dirs + names:
            path = current / name
            require(not path.is_symlink() and path.resolve() == path.absolute(),
                    "Symbolic links/junctions are not packaged: " + str(path))
            require(root in path.resolve().parents, "Included artifact escapes workspace: " + str(path))
        if code:
            dirs[:] = [name for name in dirs if name not in {"__pycache__", ".pytest_cache", ".git"}]
        for name in names:
            path = current / name
            require(path.is_file(), "Not an ordinary file: " + str(path))
            if not code or path.suffix not in {".pyc", ".pyo"}:
                files.add(path)
    return files


def development_receipt(path, root, manifest, splits):
    protocol_path = path / "protocol.json"
    protocol = read_json(protocol_path)
    require(protocol["role"] == "development_only_not_heldout" and
            protocol["paper_seed_excluded"] is True and protocol["no_training_or_test_selection_claim"] is True,
            "Development evidence is not labelled as development-only: " + str(path))
    specs = {spec["name"]: spec for spec in protocol["specs"]}
    actions = protocol["actions"]
    require(len(specs) == len(protocol["specs"]) and specs and actions and len(actions) == len(set(actions)) and
            all(type(action) is int and 0 <= action < 9 for action in actions), "Invalid development job grid")
    formal_seeds = {spec["seed"] for role in ("train", "validation", "test") for spec in splits[role]}
    excluded = set(manifest["split_contract"]["excluded_development_seeds"])
    require(all(spec["seed"] != 20260906 and spec["seed"] in excluded and spec["seed"] not in formal_seeds
                for spec in specs.values()), "Development RHS is not explicitly excluded from formal splits")
    expected = {f"{name}_a{action}.json": (name, action) for name in specs for action in actions}
    summaries = read_json(path / "summary.json") if (path / "summary.json").exists() else []
    require(isinstance(summaries, list), "Development summary must be a list")
    observed, status_counts, successes = {}, Counter(), 0
    for filename, (name, action) in expected.items():
        raw_path = path / filename
        if not raw_path.exists():
            continue  # Interrupted pilot evidence is retained and identified, never hidden.
        raw = read_json(raw_path)["summary"]
        require(raw["problem"] == specs[name] and raw["config"] == protocol["config"] and
                type(raw["accepted"]) is bool and all(a == action for a in raw["action_indices"]),
                "Development raw result disagrees with its saved protocol: " + filename)
        observed[filename] = {"action": action, **raw}
        status_counts[raw["status"]] += 1
        successes += int(raw["accepted"])
    summary_names = []
    for row in summaries:
        filename = f"{row['problem']['name']}_a{row['action']}.json"
        require(filename in observed and row == observed[filename], "Development summary differs from raw evidence")
        summary_names.append(filename)
    require(len(summary_names) == len(set(summary_names)), "Duplicate development summary job")
    extra_raw = {p.name for p in path.glob("pilot*_a*.json")} - set(expected)
    require(not extra_raw, "Unexplained development raw jobs: " + str(sorted(extra_raw)))
    return {"path": path.relative_to(root).as_posix(), "role": protocol["role"],
            "protocol_sha256": digest(protocol_path), "expected_jobs": len(expected),
            "raw_jobs": len(observed), "summary_jobs": len(summaries), "successes": successes,
            "failures": len(observed) - successes, "status_counts": dict(status_counts),
            "complete_expected_grid": set(observed) == set(expected) == set(summary_names),
            "missing_jobs": sorted(set(expected) - set(observed)),
            "raw_jobs_not_yet_in_summary": sorted(set(observed) - set(summary_names)),
            "all_files_included_regardless_of_success": True,
            "note": "Development source was not originally frozen; recorded protocols/raw logs are retained as development evidence, not held-out results."}


def file_record(path, root):
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def byte_record(name, value):
    return {"path": name, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def package(results, output, development_results, root=None):
    root = Path(root or Path(__file__).resolve().parent).resolve()
    results = workspace_path(root, results, directory=True)
    output = workspace_path(root, output)
    require(output.suffix.lower() == ".zip" and not output.exists() and not output.is_symlink(),
            "Output must be a new .zip file; overwrites are forbidden")
    require(output.parent.is_dir(), "Create the requested output parent directory first")
    development = [workspace_path(root, value, directory=True) for value in development_results]
    require(len(set(development)) == len(development), "Duplicate development directory")
    required_paths = {(root / "results" / name).resolve() for name in REQUIRED_PILOTS}
    require(required_paths.issubset(development), "All four named development pilots must be explicitly supplied")
    evidence_roots = [results, *development]
    require(all(left != right and left not in right.parents and right not in left.parents
                for i, left in enumerate(evidence_roots) for right in evidence_roots[i + 1:]),
            "Primary/development directories must be distinct and nonnested")
    code_roots = [workspace_path(root, root / name, directory=True) for name in PACKAGES]
    require(all(output != tree and tree not in output.parents for tree in [*evidence_roots, *code_roots]),
            "Output archive cannot be inside a source/evidence tree being packaged")
    header = read_json(results / "manifest.json")
    require(header.get("stage") == "complete" and header.get("evaluation_started") is True and
            header.get("paper_started") is True and header.get("paper_complete") is True,
            "Complete test and paper stages before packaging; no outcome files have been read")
    files = set()
    for folder in code_roots:
        files.update(tree_files(root, folder, code=True))
    for folder in evidence_roots:
        files.update(tree_files(root, folder))
    for name in HELPERS:
        path = root / name
        require(path.is_file(), "Required runnable/helper entry point missing: " + name)
        files.add(path)
        for test in (root / ("test_" + name), root / "tests" / ("test_" + name)):
            if test.is_file():
                files.add(test)
    # Root test configuration and explicitly named Neumann helper variants/docs.
    for pattern in ("audit*neumann*.py", "pilot*neumann*.py", "run*neumann*.py", "package*neumann*.py",
                    "test*neumann*.py", "*neumann*.md", "*NEUMANN*.md", "requirements*.txt"):
        files.update(p for p in root.glob(pattern) if p.is_file())
    for name in ("pytest.ini", "pyproject.toml", "conftest.py"):
        if (root / name).is_file():
            files.add(root / name)
    require((root / "rl_tt_dgmres_v2/requirements.txt") in files and
            all((root / name / "README.md") in files for name in PACKAGES), "Missing dependency list or package documentation")
    for path in files:
        require(root in path.resolve().parents and not path.is_symlink() and path.resolve() == path.absolute(),
                "A helper/input resolves outside its declared workspace path: " + str(path))
    inventory = sorted((file_record(p, root) for p in files), key=lambda item: item["path"])
    names = [item["path"] for item in inventory]
    require(len(set(name.casefold() for name in names)) == len(names) and
            not {MANIFEST_NAME, AUDIT_NAME}.intersection(names), "Duplicate/reserved archive member name")

    # Audit only after taking the evidence inventory; subsequent byte checks
    # reject changes while auditing or writing instead of packaging a moving run.
    independent_audit = Auditor(results, root).run("complete")
    require(independent_audit["audit_passed"] is True and independent_audit["effectiveness_evaluated"] is True and
            independent_audit["partial_training_audit"] is False, "Refusing to package an incomplete or failed audit")
    manifest, splits = read_json(results / "manifest.json"), read_json(results / "splits.json")
    development_records = [development_receipt(folder, root, manifest, splits) for folder in development]
    generated = {AUDIT_NAME: json_bytes(independent_audit)}
    payload_inventory = inventory + [byte_record(name, content) for name, content in generated.items()]
    receipt = {"format": "neumann_delivery_1", "primary_results": results.relative_to(root).as_posix(),
               "development_results": development_records, "files": sorted(payload_inventory, key=lambda item: item["path"]),
               "scope": "All primary and explicitly requested development files, including failures, full raw logs, models, state checkpoints and source snapshots; no selection by observed success or speedup.",
               "runnable_packages": list(PACKAGES), "required_helpers": list(HELPERS),
               "dependency_file": "rl_tt_dgmres_v2/requirements.txt",
               "inventory_self_exclusion": "DELIVERY_MANIFEST.json cannot contain its own SHA; its bytes/SHA are verified separately and printed in the external receipt.",
               "limitations": independent_audit["limitations"],
               "usage": "Extract preserving paths and run helpers from the extracted root on the recording platform (Windows); frozen source/receipt path spelling is platform-specific. Python numerical dependencies are listed, not bundled. Existing batch inference does not require the external paper Desktop path; fresh prepare currently does."}
    generated[MANIFEST_NAME] = json_bytes(receipt)
    expected = {item["path"]: item for item in inventory}
    expected.update({name: byte_record(name, content) for name, content in generated.items()})
    created = False
    try:
        with output.open("xb") as handle:
            created = True
            with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
                for item in inventory:
                    archive.write(root / item["path"], item["path"])
                for name, content in generated.items():
                    archive.writestr(name, content)
        with zipfile.ZipFile(output) as archive:
            require(len(archive.namelist()) == len(expected) and set(archive.namelist()) == set(expected), "ZIP member inventory differs")
            for info in archive.infolist():
                item = expected[info.filename]
                require(info.file_size == item["bytes"], "ZIP member byte count differs: " + info.filename)
                value, count = hashlib.sha256(), 0
                with archive.open(info) as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        value.update(block); count += len(block)
                require(count == item["bytes"] and value.hexdigest() == item["sha256"], "ZIP content SHA differs: " + info.filename)
        for item in inventory:
            path = root / item["path"]
            require(path.stat().st_size == item["bytes"] and digest(path) == item["sha256"], "Input changed during packaging: " + item["path"])
        for folder in evidence_roots:
            require(tree_files(root, folder) == {p for p in files if folder in p.parents}, "Evidence file inventory changed during packaging")
        for folder in code_roots:
            require(tree_files(root, folder, code=True) == {p for p in files if folder in p.parents}, "Source/test inventory changed during packaging")
    except Exception:
        if created:
            output.unlink(missing_ok=True)  # Only the new archive owned by this invocation.
        raise
    return {"archive": str(output), "files": len(expected), "bytes": output.stat().st_size, "sha256": digest(output),
            "delivery_manifest_bytes": len(generated[MANIFEST_NAME]),
            "delivery_manifest_sha256": expected[MANIFEST_NAME]["sha256"], "independent_audit_sha256": expected[AUDIT_NAME]["sha256"],
            "primary_results": receipt["primary_results"], "development_directories": len(development_records),
            "audit_passed": True, "observed_primary_claim_met": independent_audit["primary_observed_claim_met"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True, help="New ZIP inside workspace, outside every input tree; parent must exist")
    parser.add_argument("--development-results", nargs="+", required=True, help="Explicitly include all four named development pilot directories")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(package(args.results, args.output, args.development_results), ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        print(json.dumps({"packaged": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
