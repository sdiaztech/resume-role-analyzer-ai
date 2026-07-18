import csv
import json

from resume_role_ai.onet import import_onet_catalog


def _write_table(path, rows) -> None:
    path.write_text(json.dumps({"row": rows}), encoding="utf-8")


def test_imports_onet_and_preserves_custom_roles(tmp_path) -> None:
    source = tmp_path / "onet"
    source.mkdir()
    output = tmp_path / "jobs.csv"
    output.write_text(
        "id,title,description,required_skills\ncustom,Custom Role,Custom work,Writing\n",
        encoding="utf-8",
    )
    _write_table(
        source / "occupation_data.json",
        [{"onetsoc_code": "15-0000.00", "title": "Developer", "description": "Build apps"}],
    )
    _write_table(
        source / "essential_skills.json",
        [
            {
                "onetsoc_code": "15-0000.00",
                "element_name": "Critical Thinking",
                "scale_id": "IM",
                "data_value": 4.5,
                "recommend_suppress": "N",
                "not_relevant": None,
            }
        ],
    )
    _write_table(
        source / "software_skills.json",
        [
            {
                "onetsoc_code": "15-0000.00",
                "workplace_example": "Python",
                "hot_technology": "Y",
                "in_demand": "Y",
            }
        ],
    )

    assert import_onet_catalog(source, output) == 2

    with output.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["id"] for row in rows] == ["custom", "onet-15-0000-00"]
    assert rows[1]["required_skills"] == "Python|Critical Thinking"
