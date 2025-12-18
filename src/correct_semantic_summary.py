import re
import json
from collections import defaultdict


def normalize_steel_member(text: str) -> str:
    if not text:
        return text

    t = text.upper()

    # ----------------------------
    # Character-level OCR cleanup
    # ----------------------------
    replacements = {
        "I": "1",
        "O": "0",
        "L": "1",
        "’": "",
        "'": "",
        "%": "",
        "—": "",
        "–": "",
        " ": ""
    }

    for k, v in replacements.items():
        t = t.replace(k, v)

    # ----------------------------
    # W-sections
    # ----------------------------
    t = re.sub(r"W(\d+)[^X]*X[^0-9]*(\d+)", r"W\1X\2", t)

    # ----------------------------
    # Channel sections
    # C8X115 → C8X11.5
    # C6X82  → C6X8.2
    # C10X153 → C10X15.3
    # ----------------------------
    t = re.sub(r"C(\d+)X(\d{2})(\d)", r"C\1X\2.\3", t)
    t = re.sub(r"C(\d+)X(\d)(\d)", r"C\1X\2.\3", t)

    # ----------------------------
    # Generic decimal fix
    # 12X207 → 12X20.7
    # ----------------------------
    t = re.sub(r"(\d+)X(\d{2})(\d)", r"\1X\2.\3", t)

    # ----------------------------
    # HSS cleanup
    # ----------------------------
    t = re.sub(r"(HSS\d+X\d+)X$", r"\1", t)

    # ----------------------------
    # Angle cleanup
    # ----------------------------
    t = re.sub(r"(L\d+X\d+)X?$", r"\1", t)

    return t


def correct_semantic_summary(input_path: str, output_path: str):
    with open(input_path, "r") as f:
        data = json.load(f)

    corrected = {}
    merge_bucket = defaultdict(list)

    # Normalize all keys
    for raw_id, member in data.items():
        fixed_id = normalize_steel_member(raw_id)
        merge_bucket[fixed_id].append(member)

    # Merge duplicates created by OCR noise
    for fixed_id, members in merge_bucket.items():
        base = members[0]

        total_occurrences = sum(m.get("occurrences", 0) for m in members)
        all_variants = set()

        for m in members:
            for v in m.get("variants", []):
                all_variants.add(normalize_steel_member(v))

        corrected[fixed_id] = {
            "id": fixed_id,
            "type": base.get("type", "UNKNOWN"),
            "occurrences": total_occurrences,
            "linked_geometry": base.get("linked_geometry", 0),
            "variants": sorted(all_variants)
        }

    with open(output_path, "w") as f:
        json.dump(corrected, f, indent=2)

    print("✅ Semantic summary OCR correction complete")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Members before: {len(data)}")
    print(f"Members after : {len(corrected)}")


if __name__ == "__main__":
    # 🔴 CHANGE THESE PATHS
    INPUT_JSON = "output_S_103/reports/semantic_summary.json"
    OUTPUT_JSON = "output_S_103/reports/semantic_summary_CORRECTED.json"

    correct_semantic_summary(INPUT_JSON, OUTPUT_JSON)
