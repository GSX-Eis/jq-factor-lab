import re
from typing import Tuple, List


def rewrite_pep585_generics(source_code: str) -> str:
    """Rewrite built-in generics to typing versions for Python 3.6.
    E.g., list[str] -> List[str], dict[str, int] -> Dict[str, int].
    Conservative regex-based replacement to avoid heavy AST typing preservation.
    """
    text = source_code
    # Simple replacements; avoid touching within strings by crude heuristic (skip lines inside quotes is complex)
    text = re.sub(r"\blist\s*\[", "List[", text)
    text = re.sub(r"\bdict\s*\[", "Dict[", text)
    text = re.sub(r"\bset\s*\[", "Set[", text)
    text = re.sub(r"\btuple\s*\[", "Tuple[", text)
    # Ensure typing imports present if any typing-capitalized names appear
    needs = set()
    for name in ("List", "Dict", "Set", "Tuple"):
        if re.search(rf"\b{name}\s*\[", text):
            needs.add(name)
    if needs and 'from typing import' not in text:
        head = f"from typing import {', '.join(sorted(needs))}\n"
        text = head + text
    return text


def rewrite_pep604_union(source_code: str) -> str:
    """Rewrite X | Y to Union[X, Y] in type hints."""
    text = source_code
    # This is a naive replacement that targets common annotations like: a: X | Y, -> Union[X, Y]
    def _replace_union(match):
        left = match.group(1).strip()
        right = match.group(2).strip()
        return f"Union[{left}, {right}]"

    # only replace within annotations (after colon) or within -> return annotations
    text = re.sub(r":\s*([A-Za-z_][\w\.]*)\s*\|\s*([A-Za-z_][\w\.]*)", lambda m: ": " + _replace_union(m), text)
    text = re.sub(r"->\s*([A-Za-z_][\w\.]*)\s*\|\s*([A-Za-z_][\w\.]*)", lambda m: "-> " + _replace_union(m), text)
    if 'Union[' in text and 'from typing import' not in text:
        text = 'from typing import Union\n' + text
    return text


def detect_unsupported_syntax(source_code: str) -> List[str]:
    """Detect some Python 3.10+ constructs that won't run on 3.6."""
    issues: List[str] = []
    if re.search(r"^\s*match\s+", source_code, flags=re.MULTILINE):
        issues.append("match/case syntax detected")
    if re.search(r"->\s*list\[|->\s*dict\[|->\s*set\[|->\s*tuple\[", source_code):
        issues.append("PEP585 generics in return annotation")
    if re.search(r"\)\s*->\s*\(|\)\s*->\s*\{", source_code):
        issues.append("complex return annotation may be unsupported")
    if re.search(r"\)\s*->\s*typing\.", source_code):
        # okay, but flag for awareness
        pass
    if re.search(r"\(.*?,\s*/\s*\)", source_code):
        issues.append("positional-only parameters '/' found")
    if re.search(r"\:\=", source_code):
        issues.append("walrus operator ':=' found")
    return issues


def rewrite_sort_kind_stable(source_code: str) -> str:
    """Replace kind='stable' with kind='mergesort' for pandas 0.x compat."""
    text = source_code
    text = text.replace("kind='stable'", "kind='mergesort'")
    text = text.replace('kind="stable"', 'kind="mergesort"')
    return text


def rewrite_to_numpy(source_code: str) -> str:
    """Replace .to_numpy() with .values for pandas 0.x compat.

    pandas < 0.24 doesn't have .to_numpy(). Handles:
      .to_numpy(dtype=X)  -> .values.astype(X)
      .to_numpy(copy=True) -> .values.copy()
      .to_numpy()         -> .values
    """
    text = source_code
    text = re.sub(
        r"\.to_numpy\(\s*dtype\s*=\s*([^)]+)\)",
        r".values.astype(\1)",
        text,
    )
    text = re.sub(r"\.to_numpy\(\s*copy\s*=\s*True\s*\)", ".values.copy()", text)
    text = re.sub(r"\.to_numpy\(\s*copy\s*=\s*False\s*\)", ".values", text)
    text = re.sub(r"\.to_numpy\(\s*\)", ".values", text)
    return text


def apply_py36_fixes(source_code: str) -> Tuple[str, List[str]]:
    # Inlined modules are concatenated after imports; annotations is also unavailable on 3.6.
    text = re.sub(r"^from __future__ import annotations\s*$", "", source_code, flags=re.MULTILINE)
    text = rewrite_pep585_generics(text)
    text = rewrite_pep604_union(text)
    text = rewrite_sort_kind_stable(text)
    text = rewrite_to_numpy(text)
    issues = detect_unsupported_syntax(text)
    return text, issues
