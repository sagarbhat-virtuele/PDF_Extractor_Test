"""
OCR Pattern Corrector for Steel Members - Adapted for Dictionary Input
Combines the sophistication of the first code with the simplicity of the second
"""

import re
import json
from typing import Dict, List, Tuple
from collections import defaultdict
from pathlib import Path
from datetime import datetime


class OCRCorrector:
    """Correct common OCR errors in steel member designations"""

    def __init__(self):
        # Character substitution rules (OCR mistakes)
        self.char_substitutions = {
            'I': '1',   # I → 1 (most common)
            'O': '0',   # O → 0
            'l': '1',   # lowercase L → 1
            'S': '5',   # S → 5 (in numbers)
            'B': '8',   # B → 8 (in numbers)
            'Z': '2',   # Z → 2
            'G': '6',   # G → 6
            '—': 'X',   # em dash → X
            '–': 'X',   # en dash → X
            'x': 'X',   # lowercase → uppercase
            '%': '',    # noise character
            "'": '',    # apostrophe noise
            '"': '',    # quote noise
            ',': '.',   # comma → decimal
        }

        # Pattern-based corrections
        self.pattern_rules = [
            # Rule 1: Fix W-sections with I instead of 1
            (r'W([I|l])(\d+)', r'W1\2'),  # WI0X17 → W10X17

            # Rule 2: Fix W-sections with O instead of 0
            (r'W([O])(\d+)', r'W0\2'),    # WO8X10 → W08X10

            # Rule 3: Fix channel designation with missing decimal
            (r'C(\d+)X(\d)(\d)(\d)', r'C\1X\2.\3'),  # C8X115 → C8X11.5

            # Rule 4: Fix HSS with dashes or noise
            (r'HSS(\d+)[xX—–](\d+)[xX—–](.+)', r'HSS\1X\2X\3'),

            # Rule 5: Remove duplicate text (W2.0xW2.0 → W20X...)
            (r'W(\d)\.(\d)XW(\d)\.(\d)', r'W\1\2X\3\4'),

            # Rule 6: Fix missing C prefix for channels
            (r'^(\d+)X(\d+\.?\d*)', r'C\1X\2'),  # 12X20.7 → C12X20.7

            # Rule 7: Fix angle with noise
            (r'L(\d+)[xX](\d+)[xX](.+)', r'L\1X\2X\3'),

            # Rule 8: Fix sections with comma instead of decimal
            (r'X(\d+),(\d+)', r'X\1.\2'),  # X11,5 → X11.5

            # Rule 9: Remove noise characters
            (r'[%\'"]', r''),

            # Rule 10: Fix sections ending with noise
            (r'([A-Z]\d+X\d+)—+', r'\1'),

            # Rule 11: Fix S-sections
            (r'S(\d)(\d)(\d)', r'S\1X\2\3'),  # S227 → S2X27 (if applicable)
        ]

        # Statistics
        self.correction_stats = defaultdict(int)
        self.corrections_made = []

    def correct_member_id(self, member_id: str, context: Dict = None) -> Tuple[str, bool, str]:
        """
        Correct a single member ID

        Args:
            member_id: Original member ID
            context: Optional context (nearby text, page info, etc.)

        Returns:
            (corrected_id, was_corrected, correction_type)
        """
        original = member_id
        corrected = member_id.upper().strip()
        correction_type = 'NONE'

        # Step 1: Character substitutions in numeric parts
        corrected = self._apply_char_substitutions(corrected)

        # Step 2: Pattern-based corrections
        corrected, pattern_applied = self._apply_pattern_rules(corrected)
        if pattern_applied:
            correction_type = 'PATTERN'

        # Step 3: Cleanup
        corrected = self._cleanup(corrected)

        was_corrected = (original != corrected)

        if was_corrected:
            self.correction_stats[correction_type] += 1
            self.corrections_made.append({
                'original': original,
                'corrected': corrected,
                'type': correction_type
            })

        return corrected, was_corrected, correction_type

    def _apply_char_substitutions(self, text: str) -> str:
        """Apply character-level substitutions intelligently"""
        result = []

        for i, char in enumerate(text):
            # Look at surrounding context
            prev_char = text[i-1] if i > 0 else ''
            next_char = text[i+1] if i < len(text)-1 else ''

            # If we're in a numeric context, apply substitutions
            if char in self.char_substitutions:
                # Check if we should substitute
                if self._is_numeric_context(prev_char, next_char, char):
                    result.append(self.char_substitutions[char])
                else:
                    result.append(char)
            else:
                result.append(char)

        return ''.join(result)

    def _is_numeric_context(self, prev_char: str, next_char: str, current_char: str) -> bool:
        """Check if we're in a numeric context"""
        # Special case: Don't replace L at start of angle sections
        if current_char in ['L', 'l'] and (prev_char == '' or prev_char == ' '):
            return False

        # After a letter followed by possible number (W|10 or C|8)
        if prev_char.isalpha() and prev_char not in ['L']:
            return True
        # Between numbers (1|0 or 1|5)
        if prev_char.isdigit() or next_char.isdigit():
            return True
        # After X (W12X|10)
        if prev_char == 'X':
            return True
        return False

    def _apply_pattern_rules(self, text: str) -> Tuple[str, bool]:
        """Apply regex pattern rules"""
        applied = False
        result = text

        for pattern, replacement in self.pattern_rules:
            new_result = re.sub(pattern, replacement, result)
            if new_result != result:
                result = new_result
                applied = True

        return result, applied

    def _cleanup(self, text: str) -> str:
        """Final cleanup"""
        # Remove duplicate characters that don't make sense
        text = re.sub(r'([A-Z])\1{2,}', r'\1', text)  # XXX → X

        # Standardize spacing
        text = text.replace(' ', '')

        # Remove trailing noise
        text = re.sub(r'[^A-Z0-9.X]+$', '', text)

        return text

    def correct_semantic_summary(self, input_path: str, output_path: str = None) -> Dict:
        """
        Correct all members in semantic summary (dictionary format)

        Args:
            input_path: Path to semantic_summary.json (dictionary format)
            output_path: Where to save corrected results (optional)

        Returns:
            Corrected semantic data with merged duplicates
        """
        print(f"{'='*80}")
        print("OCR CORRECTION - STEEL MEMBERS")
        print(f"{'='*80}\n")

        print(f"Loading: {input_path}")
        with open(input_path, 'r') as f:
            data = json.load(f)

        print(f"Found {len(data)} members to check\n")

        print(f"{'─'*80}")
        print("CORRECTIONS")
        print(f"{'─'*80}")

        # Use merge bucket to handle duplicates created by OCR correction
        merge_bucket = defaultdict(list)

        for raw_id, member in data.items():
            # Correct the main ID
            corrected_id, was_corrected, correction_type = self.correct_member_id(raw_id)

            # Correct variants
            corrected_variants = set()
            for variant in member.get('variants', []):
                corr_var, _, _ = self.correct_member_id(variant)
                corrected_variants.add(corr_var)

            # Store in merge bucket
            merge_bucket[corrected_id].append({
                'original_id': raw_id,
                'type': member.get('type', 'UNKNOWN'),
                'occurrences': member.get('occurrences', 0),
                'linked_geometry': member.get('linked_geometry', 0),
                'variants': corrected_variants,
                'was_corrected': was_corrected,
                'correction_type': correction_type
            })

            # Log correction
            if was_corrected:
                print(f"  ✓ {raw_id:20s} → {corrected_id}")

        # Merge duplicates
        corrected_data = {}
        for corrected_id, members in merge_bucket.items():
            # Combine all data from merged members
            total_occurrences = sum(m['occurrences'] for m in members)
            total_linked_geometry = sum(m['linked_geometry'] for m in members)

            all_variants = set()
            for m in members:
                all_variants.update(m['variants'])

            # Use type from first member (they should all be same after merge)
            member_type = members[0]['type']

            corrected_data[corrected_id] = {
                'id': corrected_id,
                'type': member_type,
                'occurrences': total_occurrences,
                'linked_geometry': total_linked_geometry,
                'variants': sorted(all_variants),
                'original_ids': [m['original_id'] for m in members],
                'was_merged': len(members) > 1
            }

        # Print merge statistics
        merged_count = sum(1 for m in corrected_data.values() if m['was_merged'])
        if merged_count > 0:
            print(f"\n{'─'*80}")
            print("MERGES (Duplicates created by OCR errors)")
            print(f"{'─'*80}")
            for corr_id, member in corrected_data.items():
                if member['was_merged']:
                    print(f"  🔗 {corr_id:20s} ← {', '.join(member['original_ids'])}")

        # Print statistics
        print(f"\n{'─'*80}")
        print("CORRECTION STATISTICS")
        print(f"{'─'*80}")
        print(f"Total members (before): {len(data)}")
        print(f"Total members (after):  {len(corrected_data)}")
        print(f"Corrections made:       {len(self.corrections_made)}")
        print(f"Duplicates merged:      {merged_count}")

        if self.correction_stats:
            print(f"\nBy type:")
            for corr_type, count in self.correction_stats.items():
                print(f"  {corr_type}: {count}")

        # Save corrected results
        if output_path is None:
            # Auto-generate output path
            original_path = Path(input_path)
            output_path = original_path.parent / f"{original_path.stem}_CORRECTED{original_path.suffix}"

        with open(output_path, 'w') as f:
            json.dump(corrected_data, f, indent=2)

        print(f"\n✓ Corrected results saved: {output_path}")

        return corrected_data

    def generate_correction_report(self, output_path: str):
        """Generate detailed correction report"""
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("OCR CORRECTION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("-"*80 + "\n")
            f.write("CORRECTIONS MADE\n")
            f.write("-"*80 + "\n\n")

            if self.corrections_made:
                for corr in self.corrections_made:
                    f.write(f"{corr['original']:20s} → {corr['corrected']:20s} [{corr['type']}]\n")
            else:
                f.write("No corrections needed.\n")

            f.write("\n" + "-"*80 + "\n")
            f.write("STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total corrections: {len(self.corrections_made)}\n")
            for corr_type, count in self.correction_stats.items():
                f.write(f"  {corr_type}: {count}\n")

            f.write("\n" + "="*80 + "\n")

        print(f"✓ Correction report saved: {output_path}")


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        input_json = sys.argv[1]
        output_json = sys.argv[2] if len(sys.argv) >= 3 else None

        # Run correction
        corrector = OCRCorrector()
        corrected_data = corrector.correct_semantic_summary(input_json, output_json)

        # Generate report
        if output_json:
            report_path = Path(output_json).parent / f"correction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        else:
            report_path = Path(input_json).parent / f"correction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        corrector.generate_correction_report(str(report_path))

    else:
        print("""
╔════════════════════════════════════════════════════════════════════════════╗
║               OCR CORRECTION FOR STEEL MEMBERS                             ║
╚════════════════════════════════════════════════════════════════════════════╝

USAGE:
    python ocr_corrector_adapted.py <input_json> [output_json]

EXAMPLE:
    python ocr_corrector_adapted.py semantic_summary.json semantic_summary_CORRECTED.json

INPUT FORMAT (Dictionary):
    {
      "WI0X17": {
        "type": "WIDE_FLANGE",
        "occurrences": 5,
        "linked_geometry": 0,
        "variants": ["WI0x17", "W10X17"]
      },
      "C8X115": {
        "type": "CHANNEL",
        "occurrences": 8,
        "variants": ["C8x11.5"]
      }
    }

WHAT IT DOES:
    ✓ Context-aware character substitution (I→1, O→0, etc.)
    ✓ Pattern-based corrections (10+ regex rules)
    ✓ Merges duplicates created by OCR errors
    ✓ Preserves original IDs for audit trail
    ✓ Generates detailed correction reports

COMMON FIXES:
    WI0X17       → W10X17       (I → 1)
    C8X115       → C8X11.5      (missing decimal)
    W20XW20      → W20X20       (duplicate removal)
    12X207       → C12X20.7     (missing C prefix)
    HSS6x6—%     → HSS6X6       (noise removal)

OUTPUT:
    • Corrected JSON with merged duplicates
    • Detailed correction report (.txt)
    • Statistics and audit trail
        """)