"""Import validated monthly sources; preserve valid existing fields and fail on conflicts."""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
import duckdb
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.data_revision import advance_revision

from utils.data_store import audit_relation, canonical, digest
from utils.input_store import seed_input_rows

SOURCE = Path("./data/import/all_a_5m_backfill_2016_2026")
FIELDS = ["open", "close", "high", "low", "volume", "money", "factor"]
RAW_FIELDS = FIELDS[:-1]
# Original, already downloaded monthly request producer; preserve its exact identity.
LEGACY_EXTRACTOR_SHA256 = (
    "3471ca2d0752cca3f7b98b633f15da8a5f4ad1c05d1d30fdc188df3d9c0adefa"
)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def validate_partition(directory, expected_request=None):
    """Validate a complete frozen monthly download without consulting imported.json."""
    config = json.loads((directory / "config.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    if config.get("partition") != directory.name and expected_request is None:
        raise ValueError("Monthly partition directory does not match its request")
    if expected_request:
        if any(
            config.get(k) != expected_request[k]
            for k in ("partition", "entries", "plan_sha256")
        ):
            raise ValueError("Monthly source belongs to another frozen plan")
        producer = config.get("extractor_sha256")
        expected_id = (
            "all_a_5m_" + config["partition"] + "_" + config["plan_sha256"][:12]
            if producer == LEGACY_EXTRACTOR_SHA256
            else expected_request["run_id"]
        )
        if (
            producer
            not in (LEGACY_EXTRACTOR_SHA256, expected_request["extractor_sha256"])
            or config.get("run_id") != expected_id
        ):
            raise ValueError("Unknown monthly extractor/request identity")
    if any(
        result.get(k) != config.get(k) for k in ("run_id", "partition")
    ) or result.get("config_sha256") != sha(directory / "config.json"):
        raise ValueError("Monthly result identity mismatch")
    entries = config["entries"]
    if not entries or len({e["stock_code"] for e in entries}) != len(entries):
        raise ValueError("Empty or duplicate monthly stock request")
    if {e["stock_code"] for e in entries} != {
        e["stock_code"] for e in result["availability"]
    } or len(result["availability"]) != len(entries):
        raise ValueError("Monthly availability does not cover requested stocks")
    with duckdb.connect() as conn:
        conn.execute("SET threads=2")
        conn.execute("SET memory_limit='3GB'")
        conn.read_csv(
            str(directory / "prices.csv"), header=True, all_varchar=True
        ).create_view("p")
        columns = [r[0] for r in conn.execute("DESCRIBE p").fetchall()]
        if (
            columns != ["stock_code", "date", *FIELDS, "paused"]
            or columns != result["fields"]
        ):
            raise ValueError("Monthly price schema mismatch")
        count, keys = conn.execute(
            "SELECT count(*),count(DISTINCT (stock_code,date)) FROM p"
        ).fetchone()
        if (
            count != keys
            or count != result["rows"]
            or count != sum(e["expected_rows"] for e in entries)
        ):
            raise ValueError("Monthly price rows are missing or duplicated")
        actual = {
            r[0]: r[1:]
            for r in conn.execute(
                "SELECT stock_code,min(CAST(date AS DATE))::VARCHAR,max(CAST(date AS DATE))::VARCHAR,count(*) FROM p GROUP BY 1"
            ).fetchall()
        }
        if set(actual) != {e["stock_code"] for e in entries}:
            raise ValueError("Monthly price stock coverage mismatch")
        for entry in entries:
            low, high, rows = actual[entry["stock_code"]]
            if (
                not entry["start"] <= low <= high <= entry["end"]
                or rows != entry["expected_rows"]
            ):
                raise ValueError("Monthly price range/coverage mismatch")
        for column in columns[2:]:
            if conn.execute(
                f"SELECT 1 FROM p WHERE {column} IS NOT NULL AND (try_cast({column} AS DOUBLE) IS NULL OR NOT isfinite(cast({column} AS DOUBLE))) LIMIT 1"
            ).fetchone():
                raise ValueError("Invalid monthly numeric field: " + column)
        if conn.execute(
            "SELECT 1 FROM p WHERE stock_code IS NULL OR date IS NULL OR CAST(volume AS DOUBLE)<0 OR abs(CAST(volume AS DOUBLE)-round(CAST(volume AS DOUBLE)))>1e-6 OR extract(second FROM CAST(date AS TIMESTAMP))<>0 OR extract(minute FROM CAST(date AS TIMESTAMP))%5<>0 LIMIT 1"
        ).fetchone():
            raise ValueError("Invalid monthly price key, timestamp or volume")
    checked = {
        "rows": count,
        "sha256": {
            name: sha(directory / name)
            for name in ("prices.csv", "result.json", "config.json")
        },
        "partition": config["partition"],
    }
    if (directory / "manifest.json").exists() and json.loads(
        (directory / "manifest.json").read_text()
    ) != checked:
        raise ValueError("Monthly source manifest mismatch")
    return checked


def import_partition(db, directory, *, defer_factors=False):
    manifest = json.loads((directory / "manifest.json").read_text())
    if (directory / "config.json").exists():
        validate_partition(directory)
    for name, source_hash in manifest["sha256"].items():
        if (directory / name).resolve().parent != directory.resolve():
            raise ValueError("Monthly manifest references a file outside its partition")
        if sha(directory / name) != source_hash:
            raise ValueError("source integrity mismatch: " + name)
    bindings = json.loads(os.environ.get("QUANT_INPUT_PARENT_BINDINGS", "{}"))
    versioned = os.environ.get("QUANT_INPUT_BATCH_STAGING") == "1"
    if versioned != bool(bindings):
        raise ValueError(
            "Versioned price imports require a frozen parent and staging mode together"
        )
    authorization_path = directory / "replace_conflicts.json"
    authorization = (
        json.loads(authorization_path.read_text())
        if authorization_path.exists()
        else None
    )
    identity = {
        "source": str(directory.resolve()),
        "manifest": manifest,
        "defer_factors": defer_factors,
        "parent": bindings.get("inputs", {}).get("price_data_5m"),
        "authorization": authorization,
    }
    recovery = (
        Path(db).parent / "partitions" / digest(identity) if versioned else directory
    )
    recovery.mkdir(parents=True, exist_ok=True)
    receipt = recovery / "imported.json"
    previous = None
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if versioned and (
            previous.get("batch_identity") != identity or not Path(db).is_file()
        ):
            raise ValueError("Price partition receipt has no matching staging data")
        if not versioned:
            return previous
    if authorization and (
        authorization.get("partition") != directory.name
        or authorization.get("replace_fields") != RAW_FIELDS
    ):
        raise ValueError("invalid conflict replacement authorization")
    with duckdb.connect(str(db)) as c:
        c.execute("SET threads=2")
        c.execute(
            "CREATE TEMP TABLE stage AS SELECT stock_code,cast(date AS TIMESTAMP) AS date,cast(open AS DOUBLE) AS open,cast(close AS DOUBLE) AS close,cast(high AS DOUBLE) AS high,cast(low AS DOUBLE) AS low,cast(volume AS BIGINT) AS volume,cast(money AS DOUBLE) AS money,cast(factor AS DOUBLE) AS factor FROM read_csv(?,header=true,all_varchar=true)",
            [str(directory / "prices.csv")],
        )
        if previous:
            checked = audit_relation(
                c,
                "(SELECT p.* FROM price_data_5m p WHERE EXISTS (SELECT 1 FROM stage s WHERE s.stock_code=p.stock_code AND s.date=p.date))",
            )
            if checked != previous.get("staging_audit"):
                raise ValueError(
                    "Price partition staging no longer matches its receipt"
                )
            return previous
        if versioned:
            seed_input_rows(
                c, "price_data_5m", "stage", ["stock_code", "date"], bindings=bindings
            )
        c.execute("BEGIN TRANSACTION")
        try:
            # Historical source factor reference dates can differ by legacy partition.
            # A constant, well-observed overlap ratio changes no within-partition return.
            scales = (
                []
                if defer_factors
                else c.execute("""SELECT p.stock_code,count(*),min(p.factor/s.factor),max(p.factor/s.factor),avg(p.factor/s.factor)
                FROM price_data_5m p JOIN stage s USING(stock_code,date)
                WHERE p.factor>0 AND s.factor>0 AND p.close>0 AND s.close>0 AND p.open>0 AND s.open>0
                GROUP BY 1""").fetchall()
            )
            normalizations = []
            for code, n, low, high, scale in scales:
                if abs(scale - 1) > 1e-9:
                    if n < 48 or high - low > max(1e-10, abs(scale) * 1e-9):
                        raise ValueError(
                            "nonconstant/insufficient factor overlap: " + code
                        )
                    c.execute(
                        "UPDATE stage SET factor=factor*? WHERE stock_code=?",
                        [scale, code],
                    )
                    normalizations.append(
                        {
                            "stock_code": code,
                            "scale": scale,
                            "overlap_rows": n,
                            "ratio_min": low,
                            "ratio_max": high,
                        }
                    )
            # Existing non-null/positive values are immutable; source disagreement is explicit.
            fields = [k for k in FIELDS if k != "factor" or not defer_factors]
            mismatch = " OR ".join(
                f"(p.{k} IS NOT NULL AND s.{k} IS NOT NULL AND p.{k}{'>=0' if k in ('volume', 'money') else '>0'} AND abs(p.{k}-s.{k})>greatest(1e-7,abs(p.{k})*1e-9))"
                for k in fields
            )
            conflicts = c.execute(
                f"SELECT count(*) FROM price_data_5m p JOIN stage s USING(stock_code,date) WHERE {mismatch}"
            ).fetchone()[0]
            replaced = 0
            if conflicts and not authorization:
                raise ValueError(
                    f"{conflicts} existing/source conflicts; refusing mixed data"
                )
            if conflicts:
                invalid = " OR ".join(
                    f"s.{k} IS NULL OR s.{k}{'<0' if k in ('volume', 'money') else '<=0'}"
                    for k in RAW_FIELDS
                )
                if c.execute(
                    f"SELECT count(*) FROM price_data_5m p JOIN stage s USING(stock_code,date) WHERE ({mismatch}) AND ({invalid})"
                ).fetchone()[0]:
                    raise ValueError(
                        "authorized replacement contains invalid raw values"
                    )
                backup = recovery / "existing_conflicts_before_replace.parquet"
                backup_manifest = recovery / "existing_conflicts_before_replace.json"
                if not backup.exists():
                    escaped = str(backup).replace("'", "''")
                    c.execute(
                        f"COPY (SELECT p.* FROM price_data_5m p JOIN stage s USING(stock_code,date) WHERE {mismatch}) TO '{escaped}' (FORMAT PARQUET,COMPRESSION ZSTD)"
                    )
                    backup_manifest.write_text(
                        json.dumps(
                            {
                                "rows": conflicts,
                                "sha256": sha(backup),
                                "authorization_sha256": sha(authorization_path),
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                saved = json.loads(backup_manifest.read_text())
                if sha(backup) != saved["sha256"] or saved[
                    "authorization_sha256"
                ] != sha(authorization_path):
                    raise ValueError("conflict backup integrity mismatch")
                assignments = ",".join(f"{k}=s.{k}" for k in RAW_FIELDS)
                c.execute(
                    f"UPDATE price_data_5m p SET {assignments} FROM stage s WHERE p.stock_code=s.stock_code AND p.date=s.date AND ({mismatch})"
                )
                replaced = conflicts
            inserted = c.execute(
                "SELECT count(*) FROM stage s ANTI JOIN price_data_5m p USING(stock_code,date)"
            ).fetchone()[0]
            c.execute(
                "INSERT INTO price_data_5m SELECT s.* FROM stage s ANTI JOIN price_data_5m p USING(stock_code,date)"
            )
            conditions = {
                k: f"(p.{k} IS NULL"
                + (" OR p." + k + "<=0" if k not in ("volume", "money") else "")
                + ")"
                for k in fields
            }
            useful = " OR ".join(
                f"({conditions[k]} AND s.{k} IS NOT NULL"
                + (
                    f" AND s.{k}>0"
                    if k not in ("volume", "money")
                    else f" AND s.{k}>=0"
                )
                + ")"
                for k in fields
            )
            repaired = c.execute(
                f"SELECT count(*) FROM price_data_5m p JOIN stage s USING(stock_code,date) WHERE {useful}"
            ).fetchone()[0]
            assignments = ",".join(
                f"{k}=CASE WHEN {conditions[k]} AND s.{k}{'>=0' if k in ('volume', 'money') else '>0'} THEN s.{k} ELSE p.{k} END"
                for k in fields
            )
            c.execute(
                f"UPDATE price_data_5m p SET {assignments} FROM stage s WHERE p.stock_code=s.stock_code AND p.date=s.date AND ({useful})"
            )
            matched = c.execute(
                "SELECT count(*) FROM stage s JOIN price_data_5m p USING(stock_code,date)"
            ).fetchone()[0]
            assert matched == manifest["rows"]
            staged_audit = (
                audit_relation(
                    c,
                    "(SELECT p.* FROM price_data_5m p WHERE EXISTS (SELECT 1 FROM stage s WHERE s.stock_code=p.stock_code AND s.date=p.date))",
                )
                if versioned
                else None
            )
            advance_revision(c)
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise
    result = {
        "partition": directory.name,
        "rows": manifest["rows"],
        "inserted": inserted,
        "repaired": repaired,
        "replaced_conflict_rows": replaced,
        "matched_keys": matched,
        "factor_status": "pending_final_refresh"
        if defer_factors
        else "overlap_aligned",
        "factor_overlap_normalizations": normalizations,
        "source_sha256": manifest["sha256"]["prices.csv"],
    }
    if versioned:
        result["batch_identity"] = identity
        result["staging_audit"] = staged_audit
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(canonical(result) + "\n")
    temporary.replace(receipt)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--defer-factors", action="store_true")
    p.add_argument(
        "--partition", help="Import one YYYY-MM partition for recoverable publication"
    )
    a = p.parse_args()
    reports = []
    if a.partition and not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", a.partition):
        p.error("--partition must be YYYY-MM")
    paths = (
        [a.source / a.partition / "manifest.json"]
        if a.partition
        else sorted(a.source.glob("????-??/manifest.json"))
    )
    if not paths:
        raise ValueError("No validated monthly price partitions")
    for path in paths:
        # Already committed partitions are receipts, not new inputs to ingest.
        # Avoid rehashing the entire history for every subsequent monthly import.
        receipt = path.parent / "imported.json"
        if receipt.exists() and os.environ.get("QUANT_INPUT_BATCH_STAGING") != "1":
            result = json.loads(receipt.read_text())
            assert (
                result["source_sha256"]
                == json.loads(path.read_text())["sha256"]["prices.csv"]
            )
            reports.append(result)
        else:
            reports.append(
                import_partition(a.db, path.parent, defer_factors=a.defer_factors)
            )
        print(json.dumps(reports[-1]), flush=True)
    print(json.dumps({"ok": True, "validated_partitions": len(reports)}))


if __name__ == "__main__":
    main()
