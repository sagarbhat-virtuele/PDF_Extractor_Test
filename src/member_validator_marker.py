"""
Member Validator & PDF Marker
Cross-reference detected members with Excel and mark them on PDF
"""

import pandas as pd
import json
import fitz  # PyMuPDF
from pathlib import Path
from collections import defaultdict
import re
from datetime import datetime


class MemberValidator:
    """Validate detected members against Excel reference"""
    
    def __init__(self, excel_path: str, semantic_results_path: str):
        """
        Initialize validator
        
        Args:
            excel_path: Path to Excel file with member list
            semantic_results_path: Path to member_summary_*.json from semantic analysis
        """
        self.excel_path = excel_path
        self.semantic_results_path = semantic_results_path
        
        # Load data
        self.excel_members = self._load_excel()
        self.detected_members = self._load_semantic_results()
        
        # Validation results
        self.validation_results = {
            'matched': [],
            'unmatched_detected': [],
            'missing_from_pdf': [],
            'statistics': {}
        }
    
    def _load_excel(self) -> pd.DataFrame:
        """Load and normalize Excel member data"""
        print(f"Loading Excel file: {self.excel_path}")
        
        # Try to read Excel (handle different extensions)
        if self.excel_path.endswith('.csv'):
            df = pd.read_csv(self.excel_path)
        else:
            df = pd.read_excel(self.excel_path)
        
        print(f"  ✓ Loaded {len(df)} rows")
        print(f"  Columns: {list(df.columns)}")
        
        # Normalize member IDs (assuming there's a column with member names)
        # Auto-detect the member column
        member_col = self._detect_member_column(df)
        
        if member_col:
            df['normalized_member'] = df[member_col].apply(self._normalize_member_id)
            print(f"  ✓ Using column '{member_col}' for member IDs")
        else:
            raise ValueError("Could not find member ID column in Excel. Please specify.")
        
        return df
    
    def _detect_member_column(self, df: pd.DataFrame) -> str:
        """Auto-detect which column contains member IDs"""
        # Common column names for members
        possible_names = [
            'member', 'member_id', 'mark', 'piece_mark', 'steel_member',
            'designation', 'section', 'profile', 'size', 'description'
        ]
        
        # Check for exact matches (case-insensitive)
        for col in df.columns:
            if col.lower() in possible_names:
                return col
        
        # Check for partial matches
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in ['member', 'mark', 'size', 'section']):
                return col
        
        # Check first few rows for member-like patterns
        for col in df.columns:
            sample = df[col].dropna().head(10).astype(str)
            if any(self._looks_like_member(val) for val in sample):
                return col
        
        return None
    
    def _looks_like_member(self, text: str) -> bool:
        """Check if text looks like a member ID"""
        patterns = [
            r'W\d+[xX]\d+',
            r'C\d+[xX][\d.]+',
            r'HSS\d+[xX]\d+',
            r'L\d+[xX]\d+',
            r'PL\d+[xX]\d+'
        ]
        return any(re.search(pattern, text) for pattern in patterns)
    
    def _normalize_member_id(self, member_id) -> str:
        """Normalize member ID for comparison"""
        if pd.isna(member_id):
            return ''
        
        text = str(member_id).upper().strip()
        # Remove spaces
        text = re.sub(r'\s+', '', text)
        # Standardize x vs X
        text = text.replace('x', 'X')
        # Remove common prefixes
        text = text.replace('TYP.', '').replace('TYP', '')
        
        return text
    
    def _load_semantic_results(self) -> dict:
        """Load semantic analysis results"""
        print(f"\nLoading semantic results: {self.semantic_results_path}")
        
        with open(self.semantic_results_path, 'r') as f:
            data = json.load(f)
        
        # Get the members list
        members = data.get('members', [])
        print(f"  ✓ Loaded {len(members)} detected members")
        
        # Normalize detected member IDs
        for member in members:
            member['normalized_id'] = self._normalize_member_id(member['id'])
        
        return members
    
    def validate(self) -> dict:
        """Cross-reference detected members with Excel"""
        print("\n" + "="*80)
        print("VALIDATION ANALYSIS")
        print("="*80)
        
        # Create lookup sets
        excel_members_set = set(self.excel_members['normalized_member'].dropna())
        detected_members_set = set(m['normalized_id'] for m in self.detected_members if m['normalized_id'])
        
        print(f"\nExcel members: {len(excel_members_set)}")
        print(f"Detected members: {len(detected_members_set)}")
        
        # Find matches
        matched = excel_members_set & detected_members_set
        unmatched_detected = detected_members_set - excel_members_set
        missing_from_pdf = excel_members_set - detected_members_set
        
        # Build detailed results
        for member in self.detected_members:
            norm_id = member['normalized_id']
            if not norm_id:
                continue
            
            result = {
                'detected_id': member['id'],
                'normalized_id': norm_id,
                'type': member.get('member_type', 'UNKNOWN'),
                'occurrences': member['occurrences'],
                'locations': member['locations'],
                'variants': member.get('variants', [])
            }
            
            if norm_id in matched:
                # Find matching Excel row
                excel_row = self.excel_members[
                    self.excel_members['normalized_member'] == norm_id
                ].iloc[0].to_dict()
                
                result['status'] = 'MATCHED'
                result['excel_data'] = excel_row
                self.validation_results['matched'].append(result)
            else:
                result['status'] = 'NOT_IN_EXCEL'
                result['possible_ocr_error'] = self._check_ocr_error(norm_id, excel_members_set)
                self.validation_results['unmatched_detected'].append(result)
        
        # Add missing members
        for excel_member in missing_from_pdf:
            excel_row = self.excel_members[
                self.excel_members['normalized_member'] == excel_member
            ].iloc[0].to_dict()
            
            self.validation_results['missing_from_pdf'].append({
                'excel_member': excel_member,
                'status': 'NOT_DETECTED',
                'excel_data': excel_row,
                'possible_matches': self._find_similar_detected(excel_member, detected_members_set)
            })
        
        # Statistics
        self.validation_results['statistics'] = {
            'total_in_excel': len(excel_members_set),
            'total_detected': len(detected_members_set),
            'matched': len(matched),
            'unmatched_detected': len(unmatched_detected),
            'missing_from_pdf': len(missing_from_pdf),
            'match_rate': len(matched) / len(excel_members_set) * 100 if excel_members_set else 0
        }
        
        # Print summary
        self._print_validation_summary()
        
        return self.validation_results
    
    def _check_ocr_error(self, detected: str, excel_set: set) -> list:
        """Check if detection might be an OCR error of an Excel member"""
        from difflib import get_close_matches
        
        # Find similar members (possible OCR errors)
        similar = get_close_matches(detected, excel_set, n=3, cutoff=0.6)
        return similar
    
    def _find_similar_detected(self, excel_member: str, detected_set: set) -> list:
        """Find similar detected members"""
        from difflib import get_close_matches
        
        similar = get_close_matches(excel_member, detected_set, n=3, cutoff=0.6)
        return similar
    
    def _print_validation_summary(self):
        """Print validation summary"""
        stats = self.validation_results['statistics']
        
        print(f"\n{'─'*80}")
        print("VALIDATION SUMMARY")
        print(f"{'─'*80}")
        print(f"✓ Matched:              {stats['matched']:3d} ({stats['match_rate']:.1f}%)")
        print(f"⚠ Unmatched (detected): {stats['unmatched_detected']:3d}")
        print(f"✗ Missing from PDF:     {stats['missing_from_pdf']:3d}")
        print(f"{'─'*80}")
        
        # Show some unmatched details
        if self.validation_results['unmatched_detected']:
            print("\n⚠ UNMATCHED DETECTED MEMBERS (possible OCR errors):")
            for item in self.validation_results['unmatched_detected'][:10]:
                similar = item.get('possible_ocr_error', [])
                if similar:
                    print(f"  • {item['detected_id']} → might be: {', '.join(similar)}")
                else:
                    print(f"  • {item['detected_id']} (no similar match)")
        
        if self.validation_results['missing_from_pdf']:
            print("\n✗ MISSING FROM PDF (should be there):")
            for item in self.validation_results['missing_from_pdf'][:10]:
                similar = item.get('possible_matches', [])
                if similar:
                    print(f"  • {item['excel_member']} → detected as: {', '.join(similar)}?")
                else:
                    print(f"  • {item['excel_member']} (not found)")
    
    def save_validation_report(self, output_path: str):
        """Save validation report to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        print(f"\n✓ Validation report saved: {output_path}")


class PDFMarker:
    """Mark validated members on PDF"""
    
    def __init__(self, pdf_path: str, validation_results: dict):
        self.pdf_path = pdf_path
        self.validation_results = validation_results
        self.doc = fitz.open(pdf_path)
    
    def mark_pdf(self, output_path: str, mark_mode: str = 'matched'):
        """
        Mark members on PDF
        
        Args:
            output_path: Where to save marked PDF
            mark_mode: 'matched' | 'all' | 'unmatched'
        """
        print(f"\n{'='*80}")
        print(f"MARKING PDF: {mark_mode.upper()}")
        print(f"{'='*80}")
        
        # Choose what to mark
        if mark_mode == 'matched':
            members_to_mark = self.validation_results['matched']
            color = (0, 1, 0)  # Green for matched
        elif mark_mode == 'unmatched':
            members_to_mark = self.validation_results['unmatched_detected']
            color = (1, 0, 0)  # Red for unmatched
        else:  # all
            members_to_mark = (self.validation_results['matched'] + 
                              self.validation_results['unmatched_detected'])
            color = None  # Will use different colors
        
        marked_count = 0
        
        for member in members_to_mark:
            # Determine color
            if color is None:
                mark_color = (0, 1, 0) if member['status'] == 'MATCHED' else (1, 0, 0)
            else:
                mark_color = color
            
            # Mark all locations
            for location in member['locations']:
                page_num = location['page'] - 1  # 0-indexed
                bbox = location['bbox']
                
                page = self.doc[page_num]
                
                # Draw rectangle around text
                rect = fitz.Rect(bbox)
                
                # Add highlight
                highlight = page.add_highlight_annot(rect)
                highlight.set_colors(stroke=mark_color)
                highlight.update()
                
                # Add text annotation
                annot_text = f"{member['detected_id']}"
                if member['status'] == 'MATCHED':
                    annot_text += " ✓"
                else:
                    annot_text += " ⚠"
                
                # Add small comment
                point = fitz.Point(bbox[0], bbox[1])
                annot = page.add_text_annot(point, annot_text)
                annot.set_colors(stroke=mark_color)
                annot.update()
                
                marked_count += 1
        
        # Save marked PDF
        self.doc.save(output_path)
        print(f"✓ Marked {marked_count} locations")
        print(f"✓ Saved: {output_path}")
        
        self.doc.close()
    
    def create_marked_versions(self, output_dir: str):
        """Create multiple marked versions"""
        Path(output_dir).mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Version 1: Only matched (green)
        matched_path = Path(output_dir) / f"marked_MATCHED_{timestamp}.pdf"
        self.doc = fitz.open(self.pdf_path)
        self.mark_pdf(str(matched_path), 'matched')
        
        # Version 2: Only unmatched (red)
        unmatched_path = Path(output_dir) / f"marked_UNMATCHED_{timestamp}.pdf"
        self.doc = fitz.open(self.pdf_path)
        self.mark_pdf(str(unmatched_path), 'unmatched')
        
        # Version 3: All (color-coded)
        all_path = Path(output_dir) / f"marked_ALL_{timestamp}.pdf"
        self.doc = fitz.open(self.pdf_path)
        self.mark_pdf(str(all_path), 'all')
        
        print(f"\n✓ Created 3 marked PDF versions in: {output_dir}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def validate_and_mark(
    excel_path: str,
    semantic_results_path: str,
    pdf_path: str,
    output_dir: str = 'validation_output'
):
    """
    Complete validation and marking workflow
    
    Args:
        excel_path: Path to Excel file with reference member list
        semantic_results_path: Path to member_summary_*.json
        pdf_path: Original PDF to mark
        output_dir: Where to save outputs
    """
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"{'='*80}")
    print("MEMBER VALIDATION & MARKING WORKFLOW")
    print(f"{'='*80}\n")
    
    # Step 1: Validate
    validator = MemberValidator(excel_path, semantic_results_path)
    validation_results = validator.validate()
    
    # Save validation report
    report_path = Path(output_dir) / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    validator.save_validation_report(str(report_path))
    
    # Save human-readable report
    txt_report_path = Path(output_dir) / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    save_readable_report(validation_results, str(txt_report_path))
    
    # Step 2: Mark PDF
    marker = PDFMarker(pdf_path, validation_results)
    marker.create_marked_versions(output_dir)
    
    print(f"\n{'='*80}")
    print("✅ VALIDATION & MARKING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nOutputs in: {Path(output_dir).absolute()}")
    
    return validation_results


def save_readable_report(validation_results: dict, output_path: str):
    """Save human-readable validation report"""
    stats = validation_results['statistics']
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MEMBER VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total in Excel:         {stats['total_in_excel']}\n")
        f.write(f"Total Detected in PDF:  {stats['total_detected']}\n")
        f.write(f"Matched:                {stats['matched']} ({stats['match_rate']:.1f}%)\n")
        f.write(f"Unmatched (detected):   {stats['unmatched_detected']}\n")
        f.write(f"Missing from PDF:       {stats['missing_from_pdf']}\n\n")
        
        # Matched members
        f.write("-"*80 + "\n")
        f.write(f"✓ MATCHED MEMBERS ({len(validation_results['matched'])})\n")
        f.write("-"*80 + "\n")
        for item in sorted(validation_results['matched'], key=lambda x: x['occurrences'], reverse=True):
            f.write(f"\n{item['detected_id']}:\n")
            f.write(f"  Type: {item['type']}\n")
            f.write(f"  Occurrences in PDF: {item['occurrences']}\n")
            f.write(f"  Status: ✓ MATCHED IN EXCEL\n")
        
        # Unmatched
        if validation_results['unmatched_detected']:
            f.write("\n" + "-"*80 + "\n")
            f.write(f"⚠ UNMATCHED DETECTED MEMBERS ({len(validation_results['unmatched_detected'])})\n")
            f.write("-"*80 + "\n")
            f.write("These were detected in PDF but NOT found in Excel (possible OCR errors)\n\n")
            for item in validation_results['unmatched_detected']:
                f.write(f"\n{item['detected_id']}:\n")
                f.write(f"  Type: {item['type']}\n")
                f.write(f"  Occurrences: {item['occurrences']}\n")
                f.write(f"  Status: ⚠ NOT IN EXCEL\n")
                if item.get('possible_ocr_error'):
                    f.write(f"  Might be: {', '.join(item['possible_ocr_error'])}\n")
                f.write(f"  Variants detected: {', '.join(item['variants'])}\n")
        
        # Missing
        if validation_results['missing_from_pdf']:
            f.write("\n" + "-"*80 + "\n")
            f.write(f"✗ MISSING FROM PDF ({len(validation_results['missing_from_pdf'])})\n")
            f.write("-"*80 + "\n")
            f.write("These are in Excel but were NOT detected in PDF\n\n")
            for item in validation_results['missing_from_pdf']:
                f.write(f"\n{item['excel_member']}:\n")
                f.write(f"  Status: ✗ NOT DETECTED IN PDF\n")
                if item.get('possible_matches'):
                    f.write(f"  Similar detected members: {', '.join(item['possible_matches'])}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Readable report saved: {output_path}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Example usage
    if len(sys.argv) >= 4:
        excel_path = sys.argv[1]
        semantic_json = sys.argv[2]
        pdf_path = sys.argv[3]
        
        validate_and_mark(
            excel_path=excel_path,
            semantic_results_path=semantic_json,
            pdf_path=pdf_path,
            output_dir='validation_output'
        )
    else:
        print("""
USAGE:
    python member_validator_marker.py <excel_file> <semantic_json> <pdf_file>

EXAMPLE:
    python member_validator_marker.py \\
        member_list.xlsx \\
        output/reports/member_summary_20240115_143022.json \\
        A1.pdf

INPUTS:
    excel_file       - Excel/CSV with reference member list
    semantic_json    - member_summary_*.json from semantic analysis
    pdf_file         - Original PDF to mark

OUTPUTS:
    validation_output/
    ├── validation_report_*.json     # Detailed validation data
    ├── validation_report_*.txt      # Human-readable report
    ├── marked_MATCHED_*.pdf         # PDF with matched members (green)
    ├── marked_UNMATCHED_*.pdf       # PDF with unmatched members (red)
    └── marked_ALL_*.pdf             # PDF with all members (color-coded)

WHAT IT DOES:
    1. Loads Excel reference member list
    2. Loads detected members from semantic analysis
    3. Cross-references and validates
    4. Identifies: matched, unmatched, missing
    5. Suggests possible OCR errors
    6. Marks members on PDF with color coding:
       - Green: Matched with Excel ✓
       - Red: Not in Excel (possible error) ⚠
    7. Creates detailed validation reports
        """)