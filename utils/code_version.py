"""Identify this code export without discovering an enclosing repository."""
from importlib import metadata
import os
from pathlib import Path
import subprocess


_CORE_SOURCE_DIRS = (
    'backtest_and_analysis', 'factor_synthesis', 'factors', 'data_pipeline', 'utils', 'tools',
)


def resolve_code_version(root: Path | None = None) -> tuple[str, str]:
    root = Path(root if root is not None else Path(__file__).resolve().parents[1]).resolve()
    # A worktree has a .git file; an export nested in another repository has neither.
    if (root / '.git').exists():
        try:
            revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                      capture_output=True, text=True, check=False)
            if revision.returncode == 0 and revision.stdout.strip():
                dirty = subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD'],
                                       capture_output=True, check=False).returncode != 0
                if not dirty:
                    untracked = subprocess.run(
                        ['git', '-C', str(root), 'ls-files', '--others', '--exclude-standard',
                         '-z', '--', *_CORE_SOURCE_DIRS],
                        capture_output=True, check=False,
                    )
                    dirty = untracked.returncode != 0 or any(
                        path.endswith(b'.py') for path in untracked.stdout.split(b'\0')
                    )
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
