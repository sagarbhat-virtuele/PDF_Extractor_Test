"""
Member Validator - Compare Semantic Summary with Excel Reference
Validates OCR-corrected members against AISC standard member database
"""

import pandas as pd
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


class MemberValidator:
    """Validate detected members against Excel reference"""

    def __init__(self, excel_path: str, semantic_json_path: str):
        self.excel_path = excel_path
        self.semantic_json_path = semantic_json_path
        self.excel_members = set()
        self.semantic_data = {}

    def load_excel_members(self):
        """Extract all member designations from Excel"""
        print(f"{'='*80}")
        print("LOADING EXCEL REFERENCE")
        print(f"{'='*80}")
        print(f"Reading: {self.excel_path}")

        # Read Excel
        df = pd.read_excel(self.excel_path, sheet_name=0, header=None)

        # Pattern for member designations
        member_pattern = r'^[A-Z]+\d+[Xx.]\d+\.?\d*$|^[A-Z]+\d+[Xx]\d+$|^L\d+[Xx]\d+[Xx]?\d*\.?\d*$|^HSS\d+[Xx]\d+[Xx]?\d*\.?\d*$'

        # Extract all unique members
        for col_idx in range(df.shape[1]):
            for val in df.iloc[:, col_idx].dropna():
                val_str = str(val).strip().replace(' ', '')

                if re.match(member_pattern, val_str, re.IGNORECASE):
                    # Normalize: uppercase and X instead of x
                    normalized = val_str.upper().replace('x', 'X')
                    self.excel_members.add(normalized)

        print(f"✓ Loaded {len(self.excel_members)} unique members from Excel\n")
        return self.excel_members

    def load_semantic_data(self):
        """Load corrected semantic summary"""
        print(f"{'='*80}")
        print("LOADING SEMANTIC SUMMARY")
        print(f"{'='*80}")
        print(f"Reading: {self.semantic_json_path}")

        with open(self.semantic_json_path, 'r') as f:
            self.semantic_data = json.load(f)

        print(f"✓ Loaded {len(self.semantic_data)} members from semantic summary\n")
        return self.semantic_data

    def validate(self):
        """Validate semantic members against Excel reference"""
        # Load data
        self.load_excel_members()
        self.load_semantic_data()

        print(f"{'='*80}")
        print("VALIDATION")
        print(f"{'='*80}")

        # Validate each member
        matched = []
        partial_matches = []
        unmatched = []

        for member_id, member_info in self.semantic_data.items():
            # Check exact match
            if member_id in self.excel_members:
                matched.append({
                    'id': member_id,
                    'type': member_info.get('type', 'UNKNOWN'),
                    'occurrences': member_info.get('occurrences', 0),
                    'status': 'EXACT_MATCH'
                })
            else:
                # Check if any variant matches
                variant_match = None
                variants = member_info.get('variants', [])

                for variant in variants:
                    if variant in self.excel_members:
                        variant_match = variant
                        break

                if variant_match:
                    partial_matches.append({
                        'id': member_id,
                        'matched_as': variant_match,
                        'type': member_info.get('type', 'UNKNOWN'),
                        'occurrences': member_info.get('occurrences', 0),
                        'status': 'VARIANT_MATCH'
                    })
                else:
                    unmatched.append({
                        'id': member_id,
                        'type': member_info.get('type', 'UNKNOWN'),
                        'occurrences': member_info.get('occurrences', 0),
                        'variants': variants,
                        'status': 'NOT_FOUND'
                    })

        # Calculate statistics
        total_matched = len(matched) + len(partial_matches)
        match_rate = (total_matched / len(self.semantic_data) * 100) if self.semantic_data else 0

        # Print results
        print(f"\n{'='*80}")
        print("RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"Total members in semantic summary: {len(self.semantic_data)}")
        print(f"Total members in Excel reference:  {len(self.excel_members)}")
        print(f"\n✅ Exact matches:    {len(matched):3d} ({len(matched)/len(self.semantic_data)*100:.1f}%)")
        print(f"🔶 Variant matches:  {len(partial_matches):3d} ({len(partial_matches)/len(self.semantic_data)*100:.1f}%)")
        print(f"❌ Not found:        {len(unmatched):3d} ({len(unmatched)/len(self.semantic_data)*100:.1f}%)")
        print(f"\n🎯 Overall match rate: {match_rate:.1f}% ({total_matched}/{len(self.semantic_data)})")

        # Detailed results
        print(f"\n{'='*80}")
        print("✅ EXACT MATCHES")
        print(f"{'='*80}")
        if matched:
            for item in matched:
                print(f"  {item['id']:20s} | {item['type']:15s} | Occurrences: {item['occurrences']:3d}")
        else:
            print("  (none)")

        if partial_matches:
            print(f"\n{'='*80}")
            print("🔶 VARIANT MATCHES")
            print(f"{'='*80}")
            for item in partial_matches:
                print(f"  {item['id']:20s} → {item['matched_as']:20s} | {item['type']:15s} | Occurrences: {item['occurrences']:3d}")

        if unmatched:
            print(f"\n{'='*80}")
            print("❌ NOT FOUND IN EXCEL")
            print(f"{'='*80}")
            for item in unmatched:
                variants_str = ', '.join(item['variants'][:3]) if item['variants'] else 'none'
                print(f"  {item['id']:20s} | {item['type']:15s} | Occurrences: {item['occurrences']:3d} | Variants: {variants_str}")

        # Group results
        results = {
            'summary': {
                'timestamp': datetime.now().isoformat(),
                'total_semantic_members': len(self.semantic_data),
                'total_excel_members': len(self.excel_members),
                'exact_matches': len(matched),
                'variant_matches': len(partial_matches),
                'not_found': len(unmatched),
                'match_rate': match_rate
            },
            'matched': matched,
            'partial_matches': partial_matches,
            'unmatched': unmatched,
            'grouped_results': self._group_by_type(matched, partial_matches, unmatched)
        }

        return results

    def _group_by_type(self, matched, partial_matches, unmatched):
        """Group results by member type"""
        groups = defaultdict(lambda: {'matched': 0, 'partial': 0, 'unmatched': 0})

        for item in matched:
            groups[item['type']]['matched'] += 1
        for item in partial_matches:
            groups[item['type']]['partial'] += 1
        for item in unmatched:
            groups[item['type']]['unmatched'] += 1

        return dict(groups)

    def save_validation_report(self, output_path: str):
        """Save validation results to JSON"""
        results = self.validate()

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Saved validation report to: {output_path}")
        return results


def save_readable_report(results: dict, output_path: str):
    """Save human-readable text report"""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MEMBER VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Generated: {results['summary']['timestamp']}\n\n")

        f.write("-"*80 + "\n")
        f.write("SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Total members detected:     {results['summary']['total_semantic_members']}\n")
        f.write(f"Excel reference members:    {results['summary']['total_excel_members']}\n")
        f.write(f"Exact matches:              {results['summary']['exact_matches']}\n")
        f.write(f"Variant matches:            {results['summary']['variant_matches']}\n")
        f.write(f"Not found:                  {results['summary']['not_found']}\n")
        f.write(f"Match rate:                 {results['summary']['match_rate']:.1f}%\n\n")

        # By type
        f.write("-"*80 + "\n")
        f.write("BY TYPE\n")
        f.write("-"*80 + "\n")
        for member_type, stats in results['grouped_results'].items():
            f.write(f"{member_type}:\n")
            f.write(f"  Matched:   {stats['matched']}\n")
            f.write(f"  Partial:   {stats['partial']}\n")
            f.write(f"  Unmatched: {stats['unmatched']}\n\n")

        # Detailed lists
        f.write("\n" + "="*80 + "\n")
        f.write("EXACT MATCHES\n")
        f.write("="*80 + "\n")
        for item in results['matched']:
            f.write(f"{item['id']:20s} | {item['type']:15s} | Occurrences: {item['occurrences']:3d}\n")

        if results['partial_matches']:
            f.write("\n" + "="*80 + "\n")
            f.write("VARIANT MATCHES\n")
            f.write("="*80 + "\n")
            for item in results['partial_matches']:
                f.write(f"{item['id']:20s} → {item['matched_as']:20s} | Occurrences: {item['occurrences']:3d}\n")

        if results['unmatched']:
            f.write("\n" + "="*80 + "\n")
            f.write("NOT FOUND\n")
            f.write("="*80 + "\n")
            for item in results['unmatched']:
                variants_str = ', '.join(item['variants'][:5]) if item['variants'] else 'none'
                f.write(f"{item['id']:20s} | {item['type']:15s} | Variants: {variants_str}\n")

    print(f"✅ Saved readable report to: {output_path}")


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        excel_path = sys.argv[1]
        semantic_json = sys.argv[2]

        # Run validation
        validator = MemberValidator(excel_path, semantic_json)
        results = validator.validate()

        # Save reports
        output_dir = Path(semantic_json).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_report = output_dir / f"validation_report_{timestamp}.json"
        txt_report = output_dir / f"validation_report_{timestamp}.txt"

        with open(json_report, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Saved JSON report: {json_report}")

        save_readable_report(results, str(txt_report))

    else:
        print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    MEMBER VALIDATOR                                        ║
╚════════════════════════════════════════════════════════════════════════════╝

USAGE:
    python member_validator.py <excel_file> <semantic_json>

EXAMPLE:
    python member_validator.py data.xls semantic_summary_CORRECTED.json

WHAT IT DOES:
    • Loads all standard members from Excel (AISC database)
    • Loads OCR-corrected members from semantic summary
    • Validates each detected member against Excel reference
    • Groups results: Matched, Variant Matched, Not Found
    • Generates detailed validation reports (JSON + TXT)

OUTPUT:
    • validation_report_YYYYMMDD_HHMMSS.json    # Structured data
    • validation_report_YYYYMMDD_HHMMSS.txt     # Human-readable

MATCH TYPES:
    ✅ EXACT_MATCH     - Member ID exactly matches Excel
    🔶 VARIANT_MATCH   - One of the variants matches Excel
    ❌ NOT_FOUND       - No match in Excel (may need manual review)
        """)