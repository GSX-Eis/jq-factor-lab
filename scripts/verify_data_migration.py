"""Read-only, resumable content verification; this command never authorizes deletion."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data_store import audit_relation, canonical, digest, parquet_file_snapshot, table_name


def verify(input_db, output_db, manifest_path, report_path, *, target_only=False):
    manifest = json.loads(manifest_path.read_text())
    sources = {Path(row["source"]).resolve() for row in manifest["tables"]}
    paths = (set() if target_only else sources) | {input_db.resolve(), output_db.resolve()}
    identity = {}
    for path in sorted(paths):
        if any(
            Path(str(path) + suffix).exists() for suffix in (".wal", "-wal", "-journal")
        ):
            raise ValueError(
                f"Active WAL requires a stable checkpoint before audit: {path}"
            )
        stat = path.stat()
        identity[str(path)] = [stat.st_size, stat.st_mtime_ns]
    identity["manifest_hash"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    identity["duckdb_version"] = duckdb.__version__
    report = {
        "identity": identity,
        "checks": {},
        "complete": False,
        "deletion_authorized": False,
        "scope": "registered_targets_and_frozen_files" if target_only else "sources_and_targets",
    }
    previous = {}
    if report_path.exists():
        previous = json.loads(report_path.read_text())
    reused_checks = 0
    last_save = 0.0

    def save(*, force=False):
        nonlocal last_save
        # ponytail: interruption can repeat at most one second of checks; final reports always flush.
        now = time.monotonic()
        if not force and now - last_save < 1.0:
            return
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(report_path)
        last_save = time.monotonic()

    def check(key, operation, dependencies=()):
        nonlocal reused_checks
        if report["checks"].get(key, {}).get("ok"):
            return
        prior_identity = previous.get("identity", {})
        dependency_keys = [
            "duckdb_version",
            *(str(Path(path).resolve()) for path in dependencies),
        ]
        prior = previous.get("checks", {}).get(key, {})
        if (
            dependencies
            and prior.get("ok")
            and all(
                prior_identity.get(name) == identity[name] for name in dependency_keys
            )
        ):
            report["checks"][key] = prior
            reused_checks += 1
            return
        try:
            report["checks"][key] = operation()
        except Exception as exc:
            report["checks"][key] = {"ok": False, "error": str(exc)}
        save()
        print(json.dumps({"check": key, **report["checks"][key]}), flush=True)

    def connect(path):
        conn = duckdb.connect(str(path), read_only=True)
        conn.execute("SET threads=2")
        conn.execute("SET memory_limit='3GB'")
        return conn

    def compare(audit, expected):
        return {
            "ok": audit == expected,
            "rows": audit["rows"],
            "different_fields": [
                key for key in audit if audit[key] != expected.get(key)
            ],
        }

    def table_contract(conn, table):
        schema, name = (table if "." in table else "main." + table).split(".")
        return {
            "columns": conn.execute(
                "SELECT column_name,column_default,is_nullable FROM duckdb_columns() "
                "WHERE schema_name=? AND table_name=? ORDER BY column_index",
                [schema, name],
            ).fetchall(),
            "constraints": conn.execute(
                "SELECT constraint_type,expression,constraint_column_names FROM duckdb_constraints() "
                "WHERE schema_name=? AND table_name=? "
                "ORDER BY constraint_type,expression,constraint_column_names",
                [schema, name],
            ).fetchall(),
        }

    with connect(input_db) as conn:
        dataset_columns = {
            row[0] for row in conn.execute("DESCRIBE meta.datasets").fetchall()
        }
        policy_column = "follow_head" if "follow_head" in dataset_columns else "TRUE"
        input_policies = conn.execute(
            f"SELECT source,target_table,{policy_column} FROM meta.datasets"
        ).fetchall()
        source_column = "coalesce(source_table,target_table)" if "source_table" in dataset_columns else "target_table"
        inputs = conn.execute(
            f"SELECT target_table,physical_table,source,version,audit_json,{source_column} FROM meta.datasets"
        ).fetchall()
        for target, physical, source, version, body, original_table in inputs:
            expected = json.loads(body)
            check(
                "input:" + version + ":" + physical,
                lambda p=physical, e=expected: compare(
                    audit_relation(conn, table_name(p)), e
                ),
                (input_db,),
            )
            if target_only:
                continue

            def source_check(s=source, t=original_table, e=expected):
                with connect(s) as old:
                    return compare(audit_relation(old, table_name(t)), e)

            check("source:" + source + ":" + target, source_check, (input_db, source))

            def contract_check(s=source, t=original_table, p=physical):
                with connect(s) as old:
                    expected_contract = table_contract(old, t)
                actual_contract = table_contract(conn, p)
                return {
                    "ok": actual_contract == expected_contract,
                    "different_fields": [
                        key
                        for key in expected_contract
                        if actual_contract[key] != expected_contract[key]
                    ],
                }

            check(
                "input-contract:" + source + ":" + target,
                contract_check,
                (input_db, source),
            )

    with connect(output_db) as conn:
        outputs = conn.execute(
            "SELECT artifact_id,target_table,coalesce(storage_table,target_table),"
            "content_hash,row_count,metadata_json,original_file,access_class FROM meta.artifacts"
        ).fetchall()
        for (
            artifact,
            _target,
            storage,
            content_hash,
            count,
            metadata,
            original,
            access,
        ) in outputs:

            def output_check(
                a=artifact,
                t=storage,
                h=content_hash,
                n=count,
                kind=access,
                file=original,
            ):
                if kind == "attachment":
                    with Path(file).open("rb") as handle:
                        actual = hashlib.file_digest(handle, "sha256").hexdigest()
                    return {"ok": actual == h, "file": file}
                if kind == "document":
                    rows = conn.execute(
                        "SELECT document FROM meta.documents WHERE artifact_id=?", [a]
                    ).fetchall()
                    return {
                        "ok": len(rows) == 1 and digest(json.loads(rows[0][0])) == h,
                        "rows": len(rows),
                    }
                relation = (
                    "(SELECT * EXCLUDE (_artifact_id) FROM "
                    + table_name(t)
                    + " WHERE _artifact_id='"
                    + a.replace("'", "''")
                    + "')"
                )
                audit = audit_relation(conn, relation)
                return {
                    "ok": digest(audit) == h and audit["rows"] == n,
                    "rows": audit["rows"],
                }

            # External files can change without changing the database checkpoint.
            check(
                "output:" + artifact,
                output_check,
                () if access == "attachment" else (output_db,),
            )
            frozen = Path(original) if original else None
            if frozen is not None and frozen.parent.parent == output_db.resolve().parent / "artifacts":

                def frozen_check(file=frozen):
                    if any(path.is_symlink() for path in (file, file.parent, file.parent.parent)):
                        return {"ok": False, "error": "Frozen artifact must not be a symlink"}
                    with file.open("rb") as handle:
                        actual = hashlib.file_digest(handle, "sha256").hexdigest()
                    return {"ok": actual == file.parent.name, "file": str(file), "sha256": actual}

                check("frozen-file:" + artifact, frozen_check)
            meta = json.loads(metadata)
            if meta.get("legacy_format") == "parquet":
                def frozen_parts_check(m=meta):
                    parts = m["legacy_files"]
                    if not parts:
                        return {"ok": False, "error": "Empty frozen partition manifest"}
                    for item in parts:
                        path = Path(item["file"])
                        if path.parent.parent != output_db.resolve().parent / "artifacts" or any(p.is_symlink() for p in (path, *path.parents)):
                            return {"ok": False, "error": "Parquet partition is outside immutable custody"}
                        with path.open("rb") as handle:
                            actual = hashlib.file_digest(handle, "sha256").hexdigest()
                        if actual != item["sha256"] or actual != path.parent.name:
                            return {"ok": False, "error": "Frozen partition hash mismatch", "file": str(path)}
                    return {"ok": True, "files": len(parts)}

                check("frozen-parts:" + artifact, frozen_parts_check)
            if not target_only and meta.get("legacy_table") and meta.get("legacy_source"):

                def legacy_check(m=meta, h=content_hash):
                    if m.get("legacy_format") == "parquet":
                        expected_files = {item["source"]: item["sha256"] for item in m["legacy_files"]}
                        if parquet_file_snapshot(m["legacy_source"]) != expected_files:
                            return {"ok": False, "error": "Legacy Parquet partition set or content changed"}
                        with duckdb.connect() as old:
                            old.execute("SET threads=2")
                            old.execute("SET memory_limit='3GB'")
                            old.read_parquet(list(expected_files), hive_partitioning=False).create_view("_legacy_parquet")
                            audit = audit_relation(old, "_legacy_parquet")
                        if parquet_file_snapshot(m["legacy_source"]) != expected_files:
                            return {"ok": False, "error": "Legacy Parquet changed during audit"}
                    elif m.get("legacy_format") == "sqlite":
                        return {
                            "ok": False,
                            "error": "SQLite legacy verification is no longer supported; migrate the artifact to DuckDB or Parquet",
                        }
                    else:
                        with connect(m["legacy_source"]) as old:
                            audit = audit_relation(old, table_name(m["legacy_table"]))
                    return {"ok": digest(audit) == h, "rows": audit["rows"]}

                check(
                    "legacy-output:" + artifact,
                    legacy_check,
                    () if meta.get("legacy_format") == "parquet" else (output_db, meta["legacy_source"]),
                )

    registered_inputs = {(str(Path(s).resolve()), t, original) for t, _, s, _, _, original in inputs}
    registered_outputs = {
        (
            str(Path(json.loads(m)["legacy_source"]).resolve()),
            json.loads(m).get("legacy_table"),
        )
        for _, _, _, _, _, m, _, _ in outputs
        if json.loads(m).get("legacy_source")
    }
    missing = []
    policy_mismatches = []
    for entry in manifest["tables"]:
        target = entry.get("target", entry["table"])
        target = target if "." in target else "main." + target
        found = (
            (str(Path(entry["source"]).resolve()), target,
             entry["table"] if "." in entry["table"] else "main." + entry["table"]) in registered_inputs
            if entry["role"] == "input"
            else (str(Path(entry["source"]).resolve()), entry["table"])
            in registered_outputs
        )
        if not found:
            missing.append(entry)
        if entry["role"] == "input":
            policies = {
                policy
                for source, table, policy in input_policies
                if str(Path(source).resolve()) == str(Path(entry["source"]).resolve())
                and table == target
            }
            expected_policy = entry.get("follow_head", True)
            if not isinstance(expected_policy, bool) or policies != {expected_policy}:
                policy_mismatches.append(
                    {
                        "source": entry["source"],
                        "table": target,
                        "expected": expected_policy,
                        "actual": list(policies),
                    }
                )
    report["manifest_entries"] = len(manifest["tables"])
    report["missing_manifest_entries"] = missing
    report["input_policy_mismatches"] = policy_mismatches
    report["input_records"] = len(inputs)
    report["output_artifacts"] = len(outputs)
    report["changed_files"] = [
        str(path)
        for path in paths
        if [path.stat().st_size, path.stat().st_mtime_ns] != identity[str(path)]
        or any(
            Path(str(path) + suffix).exists() for suffix in (".wal", "-wal", "-journal")
        )
    ]
    report["manifest_changed"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest() != identity["manifest_hash"]
    report["complete"] = True
    report["reused_checks"] = reused_checks
    report["ok"] = (
        not missing
        and not report["manifest_changed"]
        and not policy_mismatches
        and not report["changed_files"]
        and all(row["ok"] for row in report["checks"].values())
    )
    save(force=True)
    return {
        "ok": report["ok"],
        "checks": len(report["checks"]),
        "failed": sum(not row["ok"] for row in report["checks"].values()),
        "deletion_authorized": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-db", type=Path, required=True)
    parser.add_argument("--output-db", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-only", action="store_true", help="Check registered data and frozen files after source cleanup; does not assert source parity")
    args = parser.parse_args()
    result = verify(args.input_db, args.output_db, args.manifest, args.report, target_only=args.target_only)
    print(canonical(result))
    raise SystemExit(0 if result["ok"] else 1)
