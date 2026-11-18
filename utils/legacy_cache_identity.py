"""One reviewed source-equivalence mapping; future edits invalidate normally."""
import hashlib
import json
from pathlib import Path


def factor_base_digest(source_file: str) -> str:
    source = Path(source_file)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    mapping = json.loads(Path(__file__).with_name('legacy_cache_identity.json').read_text())
    if digest != mapping['equivalent_sha256']:
        return digest
    dependencies = {name: hashlib.sha256((source.parent / name).read_bytes()).hexdigest()
                    for name in mapping['dependencies']}
    if dependencies == mapping['dependencies']:
        return mapping['baseline_sha256']
    return hashlib.sha256(json.dumps([digest, dependencies], sort_keys=True).encode()).hexdigest()
