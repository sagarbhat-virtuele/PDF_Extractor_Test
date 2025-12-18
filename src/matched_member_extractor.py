"""
Matched Members Location Extractor
Extracts location data for matched members from member_summary JSON
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List


class MatchedMemberExtractor:
    """Extract location data for validated/matched members"""

    def __init__(self, member_summary_path: str, corrected_semantic_path: str, validation_report_path: str = None):
        self.member_summary_path = member_summary_path
        self.corrected_semantic_path = corrected_semantic_path
        self.validation_report_path = validation_report_path

    def load_data(self):
        """Load all required JSON files"""
        print(f"{'='*80}")
        print("LOADING DATA FILES")
        print(f"{'='*80}")

        # Load member summary (with locations)
        print(f"Loading member summary: {self.member_summary_path}")
        with open(self.member_summary_path, 'r') as f:
            self.member_summary = json.load(f)

        # Get members list (could be in 'members' key or root)
        if isinstance(self.member_summary, dict) and 'members' in self.member_summary:
            self.members_with_locations = self.member_summary['members']
        elif isinstance(self.member_summary, list):
            self.members_with_locations = self.member_summary
        else:
            self.members_with_locations = list(self.member_summary.values())

        print(f"  ✓ Loaded {len(self.members_with_locations)} members with locations")

        # Load corrected semantic summary
        print(f"Loading corrected semantic: {self.corrected_semantic_path}")
        with open(self.corrected_semantic_path, 'r') as f:
            self.corrected_semantic = json.load(f)
        print(f"  ✓ Loaded {len(self.corrected_semantic)} corrected members")

        # Load validation report if provided
        if self.validation_report_path:
            print(f"Loading validation report: {self.validation_report_path}")
            with open(self.validation_report_path, 'r') as f:
                self.validation_report = json.load(f)
            print(f"  ✓ Loaded validation report")
        else:
            self.validation_report = None

        print()

    def extract_matched_members(self, include_partial: bool = True):
        """Extract location data for matched members only"""
        print(f"{'='*80}")
        print("EXTRACTING MATCHED MEMBERS WITH LOCATIONS")
        print(f"{'='*80}")

        # Get list of matched member IDs
        matched_ids = set()

        if self.validation_report:
            # Use validation report to get matched members
            for item in self.validation_report.get('matched', []):
                matched_ids.add(item['id'])

            if include_partial:
                for item in self.validation_report.get('partial_matches', []):
                    matched_ids.add(item['id'])

            print(f"Using validation report:")
            print(f"  - Exact matches: {len(self.validation_report.get('matched', []))}")
            if include_partial:
                print(f"  - Variant matches: {len(self.validation_report.get('partial_matches', []))}")
            print(f"  - Total to extract: {len(matched_ids)}")
        else:
            # Use all members from corrected semantic
            matched_ids = set(self.corrected_semantic.keys())
            print(f"No validation report - using all corrected members: {len(matched_ids)}")

        print()

        # Extract data for matched members
        matched_members_data = []
        found_count = 0
        not_found_count = 0

        # Create a lookup dict from member_summary by ID
        member_lookup = {}
        for member in self.members_with_locations:
            if isinstance(member, dict) and 'id' in member:
                member_lookup[member['id']] = member

        print(f"Extracting location data...")
        print(f"{'─'*80}")

        for member_id in sorted(matched_ids):
            if member_id in member_lookup:
                member_data = member_lookup[member_id]

                # Create cleaned version without 'source' field in locations
                cleaned_member = {
                    'id': member_data.get('id'),
                    'raw_text': member_data.get('raw_text'),
                    'occurrences': member_data.get('occurrences', 0),
                    'locations': [],
                    'variants': member_data.get('variants', []),
                    'geometry_ids': member_data.get('geometry_ids', []),
                    'member_type': member_data.get('member_type', 'UNKNOWN')
                }

                # Clean locations (remove 'source' field)
                for loc in member_data.get('locations', []):
                    cleaned_loc = {
                        'page': loc.get('page'),
                        'bbox': loc.get('bbox')
                    }
                    cleaned_member['locations'].append(cleaned_loc)

                matched_members_data.append(cleaned_member)
                found_count += 1
                print(f"  ✓ {member_id:20s} | Occurrences: {cleaned_member['occurrences']:3d} | Locations: {len(cleaned_member['locations']):3d}")
            else:
                not_found_count += 1
                print(f"  ✗ {member_id:20s} | NOT FOUND in member_summary")

        print(f"{'─'*80}")
        print(f"Found:     {found_count}")
        print(f"Not found: {not_found_count}")
        print()

        return matched_members_data

    def save_matched_members(self, output_path: str, include_partial: bool = True):
        """Extract and save matched members with locations"""
        # Load data
        self.load_data()

        # Extract matched members
        matched_members = self.extract_matched_members(include_partial=include_partial)

        # Create output structure
        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'source_member_summary': self.member_summary_path,
                'source_corrected_semantic': self.corrected_semantic_path,
                'source_validation_report': self.validation_report_path,
                'total_matched_members': len(matched_members),
                'include_partial_matches': include_partial
            },
            'members': matched_members
        }

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"{'='*80}")
        print(f"✅ Saved {len(matched_members)} matched members to: {output_path}")
        print(f"{'='*80}")

        return output_data

    def generate_summary_report(self, matched_members_data: List[Dict]) -> str:
        """Generate a summary report"""
        report = []
        report.append("="*80)
        report.append("MATCHED MEMBERS EXTRACTION SUMMARY")
        report.append("="*80)
        report.append("")

        # Statistics
        total_occurrences = sum(m['occurrences'] for m in matched_members_data)
        total_locations = sum(len(m['locations']) for m in matched_members_data)

        # By type
        by_type = {}
        for member in matched_members_data:
            mtype = member['member_type']
            if mtype not in by_type:
                by_type[mtype] = {'count': 0, 'occurrences': 0}
            by_type[mtype]['count'] += 1
            by_type[mtype]['occurrences'] += member['occurrences']

        report.append(f"Total matched members:   {len(matched_members_data)}")
        report.append(f"Total occurrences:       {total_occurrences}")
        report.append(f"Total locations:         {total_locations}")
        report.append("")

        report.append("BY TYPE:")
        report.append("-"*80)
        for mtype, stats in sorted(by_type.items()):
            report.append(f"  {mtype:20s}: {stats['count']:3d} members, {stats['occurrences']:4d} occurrences")

        report.append("")
        report.append("="*80)

        return "\n".join(report)


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        member_summary = sys.argv[1]
        corrected_semantic = sys.argv[2]
        validation_report = sys.argv[3] if len(sys.argv) >= 4 else None

        # Create output path
        output_path = Path(member_summary).parent / f"matched_members_with_locations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Run extraction
        extractor = MatchedMemberExtractor(member_summary, corrected_semantic, validation_report)
        matched_data = extractor.save_matched_members(str(output_path))

        # Print summary
        print("\n" + extractor.generate_summary_report(matched_data['members']))

    else:
        print("""
╔════════════════════════════════════════════════════════════════════════════╗
║              MATCHED MEMBERS LOCATION EXTRACTOR                            ║
╚════════════════════════════════════════════════════════════════════════════╝

USAGE:
    python matched_member_extractor.py <member_summary.json> <corrected_semantic.json> [validation_report.json]

EXAMPLE:
    python matched_member_extractor.py \
        member_summary_20251217_133832.json \
        semantic_summary_CORRECTED.json \
        validation_report.json

WHAT IT DOES:
    1. Loads member_summary JSON (contains bbox locations)
    2. Loads corrected semantic summary (contains matched members)
    3. Optionally loads validation report (to filter only validated matches)
    4. Extracts location data ONLY for matched/validated members
    5. Removes 'source' field from locations (cleanup)
    6. Saves to new JSON file with only matched members

INPUT (member_summary.json):
    {
      "members": [
        {
          "id": "W10X17",
          "occurrences": 9,
          "locations": [
            {"page": 1, "bbox": [x1, y1, x2, y2], "source": "ocr"},
            ...
          ],
          "variants": ["W10x17"],
          "member_type": "WIDE_FLANGE"
        }
      ]
    }

OUTPUT (matched_members_with_locations_*.json):
    {
      "metadata": { ... },
      "members": [
        {
          "id": "W10X17",
          "occurrences": 9,
          "locations": [
            {"page": 1, "bbox": [x1, y1, x2, y2]},   # 'source' removed
            ...
          ],
          "variants": ["W10x17"],
          "member_type": "WIDE_FLANGE"
        }
      ]
    }

BENEFITS:
    ✓ Filters out unmatched/invalid members
    ✓ Keeps only validated members with their exact locations
    ✓ Cleaner output (removes unnecessary fields)
    ✓ Ready for PDF marking or further processing
        """)