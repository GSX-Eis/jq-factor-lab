"""Identify this code export without discovering an enclosing repository."""
from importlib import metadata
import os
from pathlib import Path
import subprocess


def resolve_code_version(root: Path | None = None) -> tuple[str, str]:
    root = Path(root if root is not None else Path(__file__).resolve().parents[1]).resolve()
    # A worktree has a .git file; an export nested in another repository has neither.
    if (root / '.git').exists():
        try:
            revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                      capture_output=True, text=True, check=False)
            if revision.returncode == 0 and revision.stdout.strip():
                dirty = subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD'],
                                       capture_output=True, check=False).returncode
                return revision.stdout.strip() + ('-dirty' if dirty else ''), 'git'
        except OSError:
            pass
    explicit = os.environ.get('QUANTATION_CODE_VERSION', '').strip()
    if explicit:
        return explicit, 'env'
    try:
        return metadata.version('jq-factor-lab'), 'package'
    except metadata.PackageNotFoundError:
        return 'unknown', 'unknown'
