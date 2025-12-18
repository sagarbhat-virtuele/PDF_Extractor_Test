"""
Semantic Steel Detailing Analyzer
Transforms raw PDF geometry into meaningful steel objects

ARCHITECTURE:
Raw PDF Data → Geometry Features → Classification → Text Fusion → Members → Reports

Layer 1: Feature Extraction (from raw geometry)
Layer 2: Semantic Classification (geometry types)
Layer 3: Text-Geometry Fusion (link labels to shapes)
Layer 4: Member Abstraction (group into steel members)
Layer 5: Filtering (remove non-structural elements)
"""

import json
import math
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy.spatial import KDTree
import numpy as np


# ============================================================================
# LAYER 1: FEATURE EXTRACTION
# ============================================================================

@dataclass
class GeometryFeatures:
    """Enhanced geometry with computed features"""
    id: str
    page: int
    type: str  # from raw extraction
    bbox: List[float]
    source: str
    
    # Computed features
    center_x: float
    center_y: float
    width: float
    height: float
    area: float
    aspect_ratio: float
    
    # Line-specific
    length: Optional[float] = None
    orientation: Optional[str] = None  # HORIZONTAL, VERTICAL, DIAGONAL
    angle_deg: Optional[float] = None
    
    # Stroke properties
    stroke_thickness: Optional[float] = None
    is_dashed: bool = False
    
    # Circle-specific
    radius: Optional[float] = None
    is_small_circle: bool = False  # likely a bolt


class FeatureExtractor:
    """Extract semantic features from raw geometry"""
    
    # Thresholds for classification
    HORIZONTAL_THRESHOLD = 5  # degrees from horizontal
    VERTICAL_THRESHOLD = 5    # degrees from vertical
    SMALL_CIRCLE_RADIUS = 15  # pixels, likely bolt holes
    THIN_LINE_THRESHOLD = 3   # pixels
    THICK_LINE_THRESHOLD = 8  # pixels
    
    def __init__(self, raw_objects: List[Dict]):
        self.raw_objects = raw_objects
        self.features = []
    
    def extract_all(self) -> List[GeometryFeatures]:
        """Extract features from all geometry"""
        for idx, obj in enumerate(self.raw_objects):
            feature = self._extract_features(obj, idx)
            if feature:
                self.features.append(feature)
        return self.features
    
    def _extract_features(self, obj: Dict, idx: int) -> Optional[GeometryFeatures]:
        """Extract features from single geometry object"""
        bbox = obj['bbox']
        
        # Basic bbox features
        x0, y0, x1, y1 = bbox
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        
        feature = GeometryFeatures(
            id=f"geom_{obj['page']}_{idx}",
            page=obj['page'],
            type=obj['type'],
            bbox=bbox,
            source=obj.get('source', 'unknown'),
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height,
            area=area,
            aspect_ratio=aspect_ratio
        )
        
        # Line-specific features
        if obj['type'] in ['LINE', 'line', 'l']:
            self._extract_line_features(feature, obj)
        
        # Circle-specific features
        elif obj['type'] in ['CIRCLE', 'circle', 'c']:
            self._extract_circle_features(feature, obj)
        
        return feature
    
    def _extract_line_features(self, feature: GeometryFeatures, obj: Dict):
        """Extract line-specific features"""
        # Calculate length
        feature.length = math.sqrt(feature.width**2 + feature.height**2)
        
        # Calculate angle and orientation
        if feature.width > 0:
            angle_rad = math.atan2(feature.height, feature.width)
            feature.angle_deg = math.degrees(angle_rad)
        else:
            feature.angle_deg = 90.0
        
        # Classify orientation
        abs_angle = abs(feature.angle_deg)
        if abs_angle < self.HORIZONTAL_THRESHOLD or abs_angle > (180 - self.HORIZONTAL_THRESHOLD):
            feature.orientation = "HORIZONTAL"
        elif abs(abs_angle - 90) < self.VERTICAL_THRESHOLD:
            feature.orientation = "VERTICAL"
        else:
            feature.orientation = "DIAGONAL"
        
        # Estimate stroke thickness from bbox height/width
        feature.stroke_thickness = min(feature.width, feature.height)
    
    def _extract_circle_features(self, feature: GeometryFeatures, obj: Dict):
        """Extract circle-specific features"""
        # Estimate radius from bbox
        feature.radius = min(feature.width, feature.height) / 2
        
        # Small circles are likely bolt holes
        feature.is_small_circle = feature.radius < self.SMALL_CIRCLE_RADIUS


# ============================================================================
# LAYER 2: SEMANTIC CLASSIFICATION
# ============================================================================

class SemanticClassifier:
    """Classify geometry into steel detailing object types"""
    
    # Classification rules
    BEAM_LINE_MIN_LENGTH = 100
    COLUMN_LINE_MIN_LENGTH = 150
    TABLE_BORDER_ASPECT_THRESHOLD = 0.95  # nearly square cells
    LEADER_LINE_MAX_LENGTH = 200
    TITLE_BLOCK_Y_THRESHOLD = 0.9  # bottom 10% of page
    
    def __init__(self, features: List[GeometryFeatures], page_height: float):
        self.features = features
        self.page_height = page_height
    
    def classify_all(self) -> List[Dict]:
        """Classify all geometry into semantic types"""
        semantic_objects = []
        
        for feature in self.features:
            semantic_type = self._classify_feature(feature)
            
            semantic_obj = {
                'id': feature.id,
                'page': feature.page,
                'semantic_type': semantic_type,
                'raw_type': feature.type,
                'bbox': feature.bbox,
                'center': [feature.center_x, feature.center_y],
                'dimensions': {
                    'width': feature.width,
                    'height': feature.height,
                    'area': feature.area
                },
                'features': {}
            }
            
            # Add type-specific features
            if feature.length:
                semantic_obj['features']['length'] = feature.length
            if feature.orientation:
                semantic_obj['features']['orientation'] = feature.orientation
            if feature.angle_deg is not None:
                semantic_obj['features']['angle'] = feature.angle_deg
            if feature.radius:
                semantic_obj['features']['radius'] = feature.radius
            if feature.stroke_thickness:
                semantic_obj['features']['stroke_thickness'] = feature.stroke_thickness
            
            semantic_objects.append(semantic_obj)
        
        return semantic_objects
    
    def _classify_feature(self, f: GeometryFeatures) -> str:
        """Classify a single feature into semantic type"""
        
        # BOLT_CIRCLE: small circles
        if f.type in ['CIRCLE', 'circle'] and f.is_small_circle:
            return 'BOLT_CIRCLE'
        
        # TITLE_BLOCK: rectangles in bottom portion of page
        if f.type in ['RECTANGLE', 'rectangle']:
            if f.center_y > (self.page_height * self.TITLE_BLOCK_Y_THRESHOLD):
                return 'TITLE_BLOCK'
        
        # Lines require more analysis
        if f.type in ['LINE', 'line', 'l'] and f.length:
            
            # GRID_LINE: long horizontal or vertical lines
            if f.orientation in ['HORIZONTAL', 'VERTICAL']:
                if f.length > 500 and f.stroke_thickness and f.stroke_thickness < 3:
                    return 'GRID_LINE'
            
            # BEAM_LINE: medium-long horizontal lines
            if f.orientation == 'HORIZONTAL' and f.length > self.BEAM_LINE_MIN_LENGTH:
                return 'BEAM_LINE'
            
            # COLUMN_LINE: vertical lines above threshold
            if f.orientation == 'VERTICAL' and f.length > self.COLUMN_LINE_MIN_LENGTH:
                return 'COLUMN_LINE'
            
            # LEADER_LINE: short diagonal lines
            if f.orientation == 'DIAGONAL' and f.length < self.LEADER_LINE_MAX_LENGTH:
                return 'LEADER_LINE'
            
            # TABLE_BORDER: short horizontal/vertical lines
            if f.orientation in ['HORIZONTAL', 'VERTICAL'] and f.length < 100:
                return 'TABLE_BORDER'
        
        # RECTANGLE classification
        if f.type in ['RECTANGLE', 'rectangle', 'SQUARE', 'square']:
            # TABLE_CELL: small, nearly square rectangles
            if 0.8 < f.aspect_ratio < 1.2 and f.area < 10000:
                return 'TABLE_CELL'
            
            # DETAIL_BOX: larger rectangles
            if f.area > 10000:
                return 'DETAIL_BOX'
        
        # Default: keep original type with prefix
        return f'UNKNOWN_{f.type}'


# ============================================================================
# LAYER 3: TEXT-GEOMETRY FUSION
# ============================================================================

class TextGeometryFusion:
    """Link text labels to nearby geometry"""
    
    PROXIMITY_THRESHOLD = 50  # pixels
    LEADER_LINE_SEARCH_RADIUS = 100
    
    def __init__(self, semantic_objects: List[Dict], text_items: List[Dict]):
        self.semantic_objects = semantic_objects
        self.text_items = text_items
        self.links = []
    
    def fuse_all(self) -> Dict:
        """Create links between text and geometry"""
        
        # Build spatial index for geometry
        geom_centers = np.array([[obj['center'][0], obj['center'][1]] 
                                  for obj in self.semantic_objects])
        
        if len(geom_centers) == 0:
            return {'geometry': self.semantic_objects, 'links': []}
        
        geom_tree = KDTree(geom_centers)
        
        # For each text item, find nearest geometry
        for text in self.text_items:
            # Prefer vector text over OCR
            if text['source'] == 'ocr':
                continue
            
            text_center = self._get_text_center(text['bbox'])
            
            # Find nearest geometry
            dist, idx = geom_tree.query(text_center, k=1)
            
            if dist < self.PROXIMITY_THRESHOLD:
                nearest_geom = self.semantic_objects[idx]
                
                link = {
                    'text_content': text['text'],
                    'text_bbox': text['bbox'],
                    'geometry_id': nearest_geom['id'],
                    'geometry_type': nearest_geom['semantic_type'],
                    'distance': float(dist),
                    'page': text['page']
                }
                self.links.append(link)
                
                # Add text reference to geometry object
                if 'linked_text' not in nearest_geom:
                    nearest_geom['linked_text'] = []
                nearest_geom['linked_text'].append(text['text'])
        
        return {
            'geometry': self.semantic_objects,
            'links': self.links
        }
    
    def _get_text_center(self, bbox: List[float]) -> np.ndarray:
        """Get center point of text bounding box"""
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])


# ============================================================================
# LAYER 4: MEMBER ABSTRACTION
# ============================================================================

class MemberExtractor:
    """Extract and group steel members from geometry and text"""
    
    # Member ID patterns
    MEMBER_PATTERNS = [
        r'[A-Z]\d+[A-Z]?',  # W12, C8, etc.
        r'ROD\s*\d+',        # ROD5032
        r'[A-Z]{1,3}\d+[xX]\d+',  # W12X45, HSS8X8
        r'PL\d+[xX]\d+',     # PL1X4
        r'\d+\s*[/-]\s*\d+\s*"',  # 1-1/2"
        r'[A-Z]\d+-\d+',     # A563-A
        r'[A-Z]{1,2}\d{4}',  # BO5032
    ]
    
    def __init__(self, fused_data: Dict, text_groups: Dict):
        self.geometry = fused_data['geometry']
        self.links = fused_data['links']
        self.text_groups = text_groups
        self.members = {}
    
    def extract_members(self) -> Dict:
        """Extract member-level abstractions"""
        
        # First pass: identify member IDs from text
        member_texts = self._identify_member_texts()
        
        # Second pass: link geometry to members
        for text_id, member_info in member_texts.items():
            self._link_geometry_to_member(member_info)
        
        # Third pass: aggregate member occurrences
        member_summary = self._create_member_summary()
        
        return {
            'members': list(self.members.values()),
            'summary': member_summary
        }
    
    def _identify_member_texts(self) -> Dict:
        """Identify text that represents steel members"""
        member_texts = {}
        
        for group_key, group in self.text_groups.items():
            text = group['label']
            
            # Check if text matches member patterns
            for pattern in self.MEMBER_PATTERNS:
                if re.search(pattern, text):
                    member_id = self._normalize_member_id(text)
                    
                    if member_id not in member_texts:
                        member_texts[member_id] = {
                            'id': member_id,
                            'raw_text': text,
                            'occurrences': group['count'],
                            'locations': group['locations'],
                            'variants': list(group.get('variants', [text])),
                            'geometry_ids': []
                        }
                    break
        
        self.members = member_texts
        return member_texts
    
    def _normalize_member_id(self, text: str) -> str:
        """Normalize member ID for grouping"""
        # Remove extra spaces
        text = re.sub(r'\s+', '', text)
        # Standardize x vs X
        text = text.replace('x', 'X')
        return text.upper()
    
    def _link_geometry_to_member(self, member_info: Dict):
        """Link geometry objects to a member"""
        raw_text = member_info['raw_text']
        
        # Find links where this text appears
        for link in self.links:
            if link['text_content'] == raw_text:
                member_info['geometry_ids'].append(link['geometry_id'])
    
    def _create_member_summary(self) -> Dict:
        """Create summary statistics for members"""
        summary = {
            'total_unique_members': len(self.members),
            'members_by_type': defaultdict(int),
            'total_occurrences': 0
        }
        
        for member in self.members.values():
            # Classify member type
            member_type = self._classify_member_type(member['id'])
            member['member_type'] = member_type
            summary['members_by_type'][member_type] += 1
            summary['total_occurrences'] += member['occurrences']
        
        summary['members_by_type'] = dict(summary['members_by_type'])
        return summary
    
    def _classify_member_type(self, member_id: str) -> str:
        """Classify member into structural categories"""
        if re.match(r'W\d+', member_id):
            return 'WIDE_FLANGE'
        elif re.match(r'C\d+', member_id):
            return 'CHANNEL'
        elif re.match(r'HSS\d+', member_id):
            return 'HSS_TUBE'
        elif re.match(r'L\d+', member_id):
            return 'ANGLE'
        elif 'ROD' in member_id:
            return 'ANCHOR_ROD'
        elif 'PL' in member_id or 'PLATE' in member_id:
            return 'PLATE'
        elif 'BOLT' in member_id or re.match(r'A\d{3}', member_id):
            return 'FASTENER'
        else:
            return 'OTHER'


# ============================================================================
# LAYER 5: FILTERING
# ============================================================================

class StructuralFilter:
    """Filter out non-structural elements"""
    
    NON_STRUCTURAL_TYPES = {
        'TITLE_BLOCK',
        'GRID_LINE',
        'TABLE_BORDER',
        'TABLE_CELL'
    }
    
    COMMON_NON_STRUCTURAL_TEXT = {
        'SHEET', 'DATE', 'BY', 'CHK', 'DRAWN', 'CHECKED', 
        'SCALE', 'REVISION', 'PAGE', 'OF', 'APPROVED',
        'NORTH', 'PLAN', 'ELEVATION', 'SECTION'
    }
    
    def __init__(self, semantic_objects: List[Dict], members: Dict):
        self.semantic_objects = semantic_objects
        self.members = members
    
    def filter_geometry(self) -> List[Dict]:
        """Filter out non-structural geometry"""
        structural_geom = []
        
        for obj in self.semantic_objects:
            # Skip non-structural types
            if obj['semantic_type'] in self.NON_STRUCTURAL_TYPES:
                continue
            
            # Skip if in title block region (already classified)
            if obj['semantic_type'] == 'TITLE_BLOCK':
                continue
            
            structural_geom.append(obj)
        
        return structural_geom
    
    def filter_members(self) -> Dict:
        """Filter out non-structural member text"""
        structural_members = {}
        
        for member_id, member in self.members.items():
            # Skip common non-structural text
            if any(word in member['raw_text'].upper() 
                   for word in self.COMMON_NON_STRUCTURAL_TEXT):
                continue
            
            # Skip single-letter labels
            if len(member['raw_text']) <= 1:
                continue
            
            structural_members[member_id] = member
        
        return structural_members


# ============================================================================
# INTEGRATION PIPELINE
# ============================================================================

class SemanticPipeline:
    """End-to-end semantic extraction pipeline"""
    
    def __init__(self, raw_extraction_data: Dict):
        """
        Initialize with output from existing extraction script
        
        Args:
            raw_extraction_data: Dict with keys:
                - 'objects': List of raw geometry
                - 'texts': List of text items
                - 'text_groups': Grouped text with counts
                - 'pages': Per-page data
        """
        self.raw_data = raw_extraction_data
        self.results = {}
    
    def process(self) -> Dict:
        """Run full semantic extraction pipeline"""
        
        print("\n" + "="*80)
        print("SEMANTIC EXTRACTION PIPELINE")
        print("="*80)
        
        # Layer 1: Feature Extraction
        print("\n[1/5] Extracting geometric features...")
        extractor = FeatureExtractor(self.raw_data['objects'])
        features = extractor.extract_all()
        print(f"  ✓ Extracted features from {len(features)} objects")
        
        # Layer 2: Semantic Classification
        print("\n[2/5] Classifying geometry into semantic types...")
        # Get page height from first page
        page_height = self.raw_data['pages'][0]['dimensions']['height'] if self.raw_data['pages'] else 800
        classifier = SemanticClassifier(features, page_height)
        semantic_objects = classifier.classify_all()
        print(f"  ✓ Classified {len(semantic_objects)} semantic objects")
        
        # Count by type
        type_counts = defaultdict(int)
        for obj in semantic_objects:
            type_counts[obj['semantic_type']] += 1
        print("  Types found:", dict(type_counts))
        
        # Layer 3: Text-Geometry Fusion
        print("\n[3/5] Fusing text with geometry...")
        fusion = TextGeometryFusion(semantic_objects, self.raw_data['texts'])
        fused_data = fusion.fuse_all()
        print(f"  ✓ Created {len(fused_data['links'])} text-geometry links")
        
        # Layer 4: Member Extraction
        print("\n[4/5] Extracting steel members...")
        member_extractor = MemberExtractor(fused_data, self.raw_data['text_groups'])
        member_data = member_extractor.extract_members()
        print(f"  ✓ Identified {member_data['summary']['total_unique_members']} unique members")
        print(f"  ✓ Total occurrences: {member_data['summary']['total_occurrences']}")
        
        # Layer 5: Filtering
        print("\n[5/5] Filtering non-structural elements...")
        filter_engine = StructuralFilter(semantic_objects, member_extractor.members)
        structural_geom = filter_engine.filter_geometry()
        structural_members = filter_engine.filter_members()
        print(f"  ✓ Filtered to {len(structural_geom)} structural geometry objects")
        print(f"  ✓ Filtered to {len(structural_members)} structural members")
        
        # Compile results
        self.results = {
            'semantic_objects': semantic_objects,
            'structural_geometry': structural_geom,
            'text_geometry_links': fused_data['links'],
            'members': member_data['members'],
            'structural_members': list(structural_members.values()),
            'member_summary': member_data['summary'],
            'statistics': {
                'total_semantic_objects': len(semantic_objects),
                'structural_objects': len(structural_geom),
                'non_structural_filtered': len(semantic_objects) - len(structural_geom),
                'text_geometry_links': len(fused_data['links']),
                'total_members': len(member_data['members']),
                'structural_members': len(structural_members),
                'member_types': member_data['summary']['members_by_type']
            }
        }
        
        print("\n" + "="*80)
        print("✅ SEMANTIC EXTRACTION COMPLETE")
        print("="*80)
        
        return self.results
    
    def save_semantic_reports(self, output_dir: str):
        """Save semantic analysis reports"""
        import os
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Semantic Objects Report
        semantic_file = os.path.join(output_dir, f"semantic_objects_{timestamp}.json")
        with open(semantic_file, 'w') as f:
            json.dump({
                'objects': self.results['semantic_objects'],
                'statistics': self.results['statistics']
            }, f, indent=2)
        print(f"  ✓ Saved: {os.path.basename(semantic_file)}")
        
        # 2. Member Summary Report
        member_file = os.path.join(output_dir, f"member_summary_{timestamp}.json")
        with open(member_file, 'w') as f:
            json.dump({
                'summary': self.results['member_summary'],
                'members': self.results['structural_members']
            }, f, indent=2)
        print(f"  ✓ Saved: {os.path.basename(member_file)}")
        
        # 3. Geometry-Text Links
        links_file = os.path.join(output_dir, f"geometry_text_links_{timestamp}.json")
        with open(links_file, 'w') as f:
            json.dump(self.results['text_geometry_links'], f, indent=2)
        print(f"  ✓ Saved: {os.path.basename(links_file)}")
        
        # 4. Structural Geometry Only
        struct_file = os.path.join(output_dir, f"structural_geometry_{timestamp}.json")
        with open(struct_file, 'w') as f:
            json.dump(self.results['structural_geometry'], f, indent=2)
        print(f"  ✓ Saved: {os.path.basename(struct_file)}")
        
        # 5. Human-Readable Summary
        summary_file = os.path.join(output_dir, f"semantic_summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            self._write_summary_report(f)
        print(f"  ✓ Saved: {os.path.basename(summary_file)}")
    
    def _write_summary_report(self, f):
        """Write human-readable summary"""
        stats = self.results['statistics']
        
        f.write("="*80 + "\n")
        f.write("SEMANTIC STEEL DETAILING ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write("GEOMETRY CLASSIFICATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Objects: {stats['total_semantic_objects']}\n")
        f.write(f"Structural Objects: {stats['structural_objects']}\n")
        f.write(f"Non-Structural Filtered: {stats['non_structural_filtered']}\n\n")
        
        f.write("MEMBER SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Unique Members: {stats['total_members']}\n")
        f.write(f"Structural Members: {stats['structural_members']}\n\n")
        
        f.write("Members by Type:\n")
        for mtype, count in stats['member_types'].items():
            f.write(f"  {mtype}: {count}\n")
        
        f.write("\n" + "DETAILED MEMBER LIST\n")
        f.write("-"*80 + "\n")
        for member in sorted(self.results['structural_members'], 
                            key=lambda x: x['occurrences'], reverse=True)[:30]:
            f.write(f"\n{member['id']}:\n")
            f.write(f"  Type: {member.get('member_type', 'UNKNOWN')}\n")
            f.write(f"  Occurrences: {member['occurrences']}\n")
            f.write(f"  Linked Geometry: {len(member['geometry_ids'])} objects\n")
            if member['variants']:
                f.write(f"  Variants: {', '.join(member['variants'][:3])}\n")


# ============================================================================
# USAGE EXAMPLE / INTEGRATION
# ============================================================================

def integrate_with_existing_script(raw_dump_file: str, output_dir: str):
    """
    Integration function to add to existing extraction script
    
    Usage:
        # At the end of your existing script, after creating all_data.json:
        integrate_with_existing_script(
            raw_dump_file='output/raw_dump/all_data_TIMESTAMP.json',
            output_dir='output/reports'
        )
    """
    
    # Load raw extraction data
    with open(raw_dump_file, 'r') as f:
        raw_data = json.load(f)
    
    # Run semantic pipeline
    pipeline = SemanticPipeline(raw_data)
    results = pipeline.process()
    
    # Save semantic reports
    pipeline.save_semantic_reports(output_dir)
    
    return results


# ============================================================================
# EXAMPLE JSON SCHEMAS
# ============================================================================

EXAMPLE_SCHEMAS = {
    "semantic_object": {
        "id": "geom_1_42",
        "page": 1,
        "semantic_type": "BEAM_LINE",
        "raw_type": "LINE",
        "bbox": [100.5, 200.3, 450.8, 202.1],
        "center": [275.65, 201.2],
        "dimensions": {
            "width": 350.3,
            "height": 1.8,
            "area": 630.54
        },
        "features": {
            "length": 350.31,
            "orientation": "HORIZONTAL",
            "angle": 0.29,
            "stroke_thickness": 1.8
        },
        "linked_text": ["W12X45", "TYP"]
    },
    
    "member": {
        "id": "W12X45",
        "member_type": "WIDE_FLANGE",
        "raw_text": "W12X45",
        "occurrences": 24,
        "variants": ["W12X45", "W12x45"],
        "locations": [
            {"page": 1, "bbox": [100, 200, 150, 215], "source": "vector"},
            {"page": 1, "bbox": [300, 400, 350, 415], "source": "vector"}
        ],
        "geometry_ids": ["geom_1_42", "geom_1_78", "geom_1_103"]
    },
    
    "text_geometry_link": {
        "text_content": "ROD5032",
        "text_bbox": [245.5, 312.8, 298.2, 327.1],
        "geometry_id": "geom_1_156",
        "geometry_type": "BOLT_CIRCLE",
        "distance": 12.3,
        "page": 1
    }
}


if __name__ == "__main__":
    print(__doc__)
    print("\nExample Schemas:")
    print(json.dumps(EXAMPLE_SCHEMAS, indent=2))