from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

from resume_role_ai.paths import find_repository_root


ONET_VERSION = "30.3"
ONET_PATH_VERSION = ONET_VERSION.replace(".", "_")
BASE_URL = f"https://www.onetcenter.org/dl_files/database/db_{ONET_PATH_VERSION}_json"
SOURCE_FILES = ("occupation_data.json", "essential_skills.json", "software_skills.json")
REPOSITORY_ROOT = find_repository_root()
DEFAULT_SOURCE_DIR = REPOSITORY_ROOT / "datasets" / "external" / f"onet-{ONET_VERSION}"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "datasets" / "raw" / "job_positions.csv"


def download_onet(destination: str | Path = DEFAULT_SOURCE_DIR) -> Path:
    """Download the versioned O*NET occupation and skill tables atomically."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for filename in SOURCE_FILES:
        target = destination / filename
        temporary = target.with_suffix(f"{target.suffix}.part")
        try:
            urllib.request.urlretrieve(f"{BASE_URL}/{filename}", temporary)
            payload = _read_rows(temporary)
            if not payload:
                raise ValueError(f"O*NET source file is empty: {filename}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def import_onet_catalog(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    output: str | Path = DEFAULT_OUTPUT,
    *,
    preserve_custom: bool = True,
    skills_per_type: int = 5,
) -> int:
    """Transform O*NET occupations into the API's job catalog schema."""
    if skills_per_type < 1:
        raise ValueError("skills_per_type must be at least 1")
    source_dir = Path(source_dir)
    output = Path(output)
    occupations = _read_rows(source_dir / "occupation_data.json")
    essential = _essential_skills(_read_rows(source_dir / "essential_skills.json"))
    software = _software_skills(_read_rows(source_dir / "software_skills.json"))

    custom = _load_custom_rows(output) if preserve_custom and output.exists() else []
    rows: list[dict[str, str]] = custom
    for occupation in occupations:
        code = occupation["onetsoc_code"]
        skills = _unique(
            [*software.get(code, [])[:skills_per_type], *essential.get(code, [])[:skills_per_type]]
        )
        rows.append(
            {
                "id": f"onet-{code.replace('.', '-').replace('_', '-').casefold()}",
                "title": occupation["title"],
                "description": " ".join(occupation["description"].split()),
                "required_skills": "|".join(skills),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.part")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=("id", "title", "description", "required_skills")
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return len(rows)


def _read_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("row")
    if not isinstance(rows, list):
        raise ValueError(f"O*NET source has an invalid schema: {path.name}")
    return rows


def _essential_skills(rows: list[dict[str, object]]) -> dict[str, list[str]]:
    grouped: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("scale_id") == "IM"
            and row.get("recommend_suppress") != "Y"
            and row.get("not_relevant") != "Y"
        ):
            grouped[str(row["onetsoc_code"])].append(
                (float(row["data_value"]), str(row["element_name"]))
            )
    return {
        code: [name for _, name in sorted(values, key=lambda item: (-item[0], item[1]))]
        for code, values in grouped.items()
    }


def _software_skills(rows: list[dict[str, object]]) -> dict[str, list[str]]:
    grouped: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["onetsoc_code"])].append(
            (
                0 if row.get("in_demand") == "Y" else 1,
                0 if row.get("hot_technology") == "Y" else 1,
                str(row["workplace_example"]),
            )
        )
    return {
        code: _unique([name for _, _, name in sorted(values)])
        for code, values in grouped.items()
    }


def _load_custom_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return [row for row in csv.DictReader(file) if not row["id"].startswith("onet-")]


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and import the O*NET job catalog")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--replace", action="store_true", help="Do not preserve custom roles")
    args = parser.parse_args()
    if not args.skip_download:
        download_onet(args.source_dir)
    count = import_onet_catalog(
        args.source_dir, args.output, preserve_custom=not args.replace
    )
    print(f"Wrote {count:,} job profiles to {args.output}")


if __name__ == "__main__":
    main()
