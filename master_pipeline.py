"""
Complete Integrated Steel PDF Analyzer
Combines raw extraction + semantic analysis
"""

# Add these imports at the top of your existing script
from utils.semantic_steel_analyzer import SemanticPipeline
import numpy as np
from scipy.spatial import KDTree
import os
# ... [Keep all your existing code unchanged until the end] ...

"""
Enhanced Steel Detailing PDF Analyzer
Combines PyMuPDF, OpenCV, and OCR for comprehensive extraction
"""

import fitz  # PyMuPDF
import os
import json
import cv2
import pytesseract
import numpy as np
import csv
import re
from pdf2image import convert_from_path
from rapidfuzz import process, fuzz
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

# ================== CONFIG ==================
PDF_PATH = "/home/virtuele/Documents/Framing/S.103.pdf"
OUT_DIR = "output_S_103"
DPI = 300
FUZZY_THRESHOLD = 85
MIN_OBJECT_AREA = 150
OCR_CONFIDENCE = 60

# ================== SETUP ==================
os.makedirs(OUT_DIR, exist_ok=True)
RAW_DIR = os.path.join(OUT_DIR, "raw_dump")
IMG_DIR = os.path.join(OUT_DIR, "images")
REP_DIR = os.path.join(OUT_DIR, "reports")
VIS_DIR = os.path.join(OUT_DIR, "visualizations")

for d in [RAW_DIR, IMG_DIR, REP_DIR, VIS_DIR]:
    os.makedirs(d, exist_ok=True)

# ================== HELPERS ==================
def normalize_label(text: str) -> str:
    """Normalize text for comparison"""
    text = text.upper()
    text = re.sub(r"[^\w\s/\-]", " ", text)  # Keep some special chars
    text = re.sub(r"\s+", " ", text).strip()
    return text

def bbox_center(b):
    """Get center point of bounding box"""
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

def bbox_area(b):
    """Calculate area of bounding box"""
    return (b[2] - b[0]) * (b[3] - b[1])

def is_circle(contour):
    """Check if contour is approximately circular"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    return circularity > 0.7

def detect_shape(contour):
    """Classify shape from contour"""
    area = cv2.contourArea(contour)
    if area < MIN_OBJECT_AREA:
        return None
    
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    vertices = len(approx)
    
    if is_circle(contour):
        return "CIRCLE"
    elif vertices == 2:
        return "LINE"
    elif vertices == 3:
        return "TRIANGLE"
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h) if h != 0 else 0
        if 0.95 <= aspect_ratio <= 1.05:
            return "SQUARE"
        else:
            return "RECTANGLE"
    elif vertices > 4:
        return "POLYGON"
    else:
        return "POLYLINE"

def visualize_objects(img_path, objects, output_path):
    """Draw detected objects on image"""
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    
    colors = {
        'CIRCLE': (255, 0, 0),
        'RECTANGLE': (0, 255, 0),
        'SQUARE': (0, 0, 255),
        'LINE': (255, 255, 0),
        'TRIANGLE': (255, 0, 255),
        'POLYGON': (0, 255, 255)
    }
    
    for obj in objects:
        bbox = obj['bbox']
        color = colors.get(obj['type'], (128, 128, 128))
        draw.rectangle(bbox, outline=color, width=2)
        
        # Add label
        label = f"{obj['type']}"
        draw.text((bbox[0], bbox[1] - 15), label, fill=color)
    
    img.save(output_path)

def visualize_text(img_path, texts, output_path):
    """Draw detected text bounding boxes"""
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    
    for t in texts:
        bbox = t['bbox']
        color = (0, 255, 0) if t['source'] == 'vector' else (255, 165, 0)
        draw.rectangle(bbox, outline=color, width=1)
    
    img.save(output_path)

# ================== MAIN EXTRACTION ==================
print(f"{'='*80}")
print(f"PDF Analysis Started: {os.path.basename(PDF_PATH)}")
print(f"{'='*80}\n")

doc = fitz.open(PDF_PATH)
print(f"Converting PDF to images at {DPI} DPI...")
images = convert_from_path(PDF_PATH, dpi=DPI)

all_objects = []
all_texts = []
raw_pages = []
statistics = {
    'total_pages': len(doc),
    'total_objects': 0,
    'total_texts': 0,
    'objects_by_type': defaultdict(int),
    'texts_by_source': defaultdict(int)
}

# ================== PAGE LOOP ==================
for page_index, page in enumerate(doc):
    page_num = page_index + 1
    print(f"\n{'─'*80}")
    print(f"Processing Page {page_num}/{len(doc)}")
    print(f"{'─'*80}")

    # ---------- Render image ----------
    img_path = os.path.join(IMG_DIR, f"page_{page_num}.png")
    images[page_index].save(img_path)
    print(f"  ✓ Saved page image: {os.path.basename(img_path)}")

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------- VECTOR DRAWINGS ----------
    print(f"  → Extracting vector drawings...")
    drawings = page.get_drawings()
    vector_objects = []

    for d in drawings:
        rect = d.get("rect")
        if not rect:
            continue

        obj = {
            "page": page_num,
            "type": d.get("type", "UNKNOWN"),
            "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
            "width": float(rect.width),
            "height": float(rect.height),
            "area": float(rect.width * rect.height),
            "color": d.get("color"),
            "fill": d.get("fill"),
            "source": "vector"
        }
        vector_objects.append(obj)
        all_objects.append(obj)
        statistics['objects_by_type'][obj['type']] += 1

    print(f"    Found {len(vector_objects)} vector objects")

    # ---------- OPENCV SHAPE DETECTION ----------
    print(f"  → Detecting shapes with OpenCV...")
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 3
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    opencv_objects = []
    for cnt in contours:
        shape = detect_shape(cnt)
        if not shape:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        obj = {
            "page": page_num,
            "type": shape,
            "bbox": [x, y, x + w, y + h],
            "width": w,
            "height": h,
            "area": w * h,
            "source": "opencv"
        }
        opencv_objects.append(obj)
        all_objects.append(obj)
        statistics['objects_by_type'][shape] += 1

    print(f"    Found {len(opencv_objects)} OpenCV shapes")

    # ---------- TEXT EXTRACTION (VECTOR) ----------
    print(f"  → Extracting vector text...")
    text_dict = page.get_text("dict")
    vector_texts = []

    for block in text_dict["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                txt = span["text"].strip()
                if not txt:
                    continue

                b = span["bbox"]
                entry = {
                    "page": page_num,
                    "text": txt,
                    "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    "font": span.get("font", "Unknown"),
                    "size": float(span.get("size", 0)),
                    "color": span.get("color", 0),
                    "source": "vector"
                }
                vector_texts.append(entry)
                all_texts.append(entry)
                statistics['texts_by_source']['vector'] += 1

    print(f"    Found {len(vector_texts)} vector text items")

    # ---------- OCR TEXT ----------
    print(f"  → Running OCR...")
    try:
        ocr = pytesseract.image_to_data(
            Image.open(img_path),
            output_type=pytesseract.Output.DICT
        )

        ocr_texts = []
        for i, t in enumerate(ocr["text"]):
            if t.strip() and int(ocr["conf"][i]) > OCR_CONFIDENCE:
                x = ocr["left"][i]
                y = ocr["top"][i]
                w = ocr["width"][i]
                h = ocr["height"][i]

                entry = {
                    "page": page_num,
                    "text": t,
                    "bbox": [x, y, x + w, y + h],
                    "confidence": int(ocr["conf"][i]),
                    "source": "ocr"
                }
                ocr_texts.append(entry)
                all_texts.append(entry)
                statistics['texts_by_source']['ocr'] += 1

        print(f"    Found {len(ocr_texts)} OCR text items")
    except Exception as e:
        print(f"    ⚠ OCR failed: {e}")
        ocr_texts = []

    # ---------- CREATE VISUALIZATIONS ----------
    print(f"  → Creating visualizations...")
    vis_obj_path = os.path.join(VIS_DIR, f"page_{page_num}_objects.png")
    vis_text_path = os.path.join(VIS_DIR, f"page_{page_num}_text.png")
    
    visualize_objects(img_path, vector_objects + opencv_objects, vis_obj_path)
    visualize_text(img_path, vector_texts + ocr_texts, vis_text_path)

    # ---------- RAW DUMP ----------
    raw_page = {
        "page": page_num,
        "dimensions": {
            "width": float(page.rect.width),
            "height": float(page.rect.height)
        },
        "vector_drawings": vector_objects,
        "opencv_shapes": opencv_objects,
        "vector_text": vector_texts,
        "ocr_text": ocr_texts,
        "summary": {
            "total_objects": len(vector_objects) + len(opencv_objects),
            "total_texts": len(vector_texts) + len(ocr_texts)
        }
    }
    raw_pages.append(raw_page)

    with open(os.path.join(RAW_DIR, f"page_{page_num}.json"), "w") as f:
        json.dump(raw_page, f, indent=2)

statistics['total_objects'] = len(all_objects)
statistics['total_texts'] = len(all_texts)

# ================== GROUP TEXT MEMBERS ==================
print(f"\n{'─'*80}")
print("Grouping similar text members with fuzzy matching...")
print(f"{'─'*80}")

groups = {}
unmatched_count = 0

for t in all_texts:
    norm = normalize_label(t["text"])
    if not norm:
        continue

    if norm in groups:
        groups[norm]["count"] += 1
        groups[norm]["locations"].append({
            "page": t["page"],
            "bbox": t["bbox"],
            "source": t["source"]
        })
    else:
        # Fuzzy match existing
        if groups:
            result = process.extractOne(
                norm, groups.keys(), scorer=fuzz.partial_ratio
            )
            match, score = result[0], result[1] if result else (None, 0)
        else:
            match, score = None, 0

        if score > FUZZY_THRESHOLD:
            groups[match]["count"] += 1
            groups[match]["locations"].append({
                "page": t["page"],
                "bbox": t["bbox"],
                "source": t["source"]
            })
            groups[match]["variants"].add(t["text"])
        else:
            groups[norm] = {
                "label": norm,
                "original": t["text"],
                "count": 1,
                "variants": {t["text"]},
                "locations": [{
                    "page": t["page"],
                    "bbox": t["bbox"],
                    "source": t["source"]
                }]
            }
            unmatched_count += 1

print(f"  ✓ Created {len(groups)} unique text groups")
print(f"  ✓ Matched {statistics['total_texts'] - unmatched_count} duplicates")

# ================== EXPORT REPORTS ==================
print(f"\n{'─'*80}")
print("Generating reports...")
print(f"{'─'*80}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ----- Object Report (CSV) -----
obj_csv = os.path.join(REP_DIR, f"object_report_{timestamp}.csv")
with open(obj_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["page", "type", "x0", "y0", "x1", "y1", "width", "height", "area", "source"])
    for o in all_objects:
        writer.writerow([
            o["page"], o["type"], 
            o["bbox"][0], o["bbox"][1], o["bbox"][2], o["bbox"][3],
            o.get("width", 0), o.get("height", 0), o.get("area", 0),
            o.get("source", "unknown")
        ])
print(f"  ✓ Object report: {os.path.basename(obj_csv)}")

# ----- Object Report (JSON) -----
obj_json = os.path.join(REP_DIR, f"object_report_{timestamp}.json")
with open(obj_json, "w") as f:
    json.dump(all_objects, f, indent=2)

# ----- Text Report (CSV) -----
text_csv = os.path.join(REP_DIR, f"text_report_{timestamp}.csv")
with open(text_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["page", "text", "x0", "y0", "x1", "y1", "source"])
    for t in all_texts:
        writer.writerow([
            t["page"], t["text"],
            t["bbox"][0], t["bbox"][1], t["bbox"][2], t["bbox"][3],
            t["source"]
        ])
print(f"  ✓ Text report: {os.path.basename(text_csv)}")

# ----- Member Summary (CSV) -----
member_csv = os.path.join(REP_DIR, f"member_summary_{timestamp}.csv")
with open(member_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["member", "count", "variants", "locations"])
    
    # Sort by count
    sorted_groups = sorted(groups.values(), key=lambda x: x["count"], reverse=True)
    
    for g in sorted_groups:
        variants = ", ".join(list(g["variants"])[:5])  # Limit variants shown
        locations_str = "; ".join([f"P{loc['page']}({loc['bbox'][0]:.0f},{loc['bbox'][1]:.0f})" 
                                    for loc in g["locations"][:10]])  # Limit locations shown
        writer.writerow([g["label"], g["count"], variants, locations_str])
print(f"  ✓ Member summary: {os.path.basename(member_csv)}")

# ----- Member Summary (JSON) -----
member_json = os.path.join(REP_DIR, f"member_summary_{timestamp}.json")
groups_export = {}
for key, val in groups.items():
    groups_export[key] = {
        **val,
        "variants": list(val["variants"])
    }
with open(member_json, "w", encoding="utf-8") as f:
    json.dump(groups_export, f, indent=2)

# ----- Global Raw Dump -----
raw_dump = os.path.join(RAW_DIR, f"all_data_{timestamp}.json")
with open(raw_dump, "w") as f:
    json.dump({
        "metadata": {
            "filename": os.path.basename(PDF_PATH),
            "timestamp": timestamp,
            "total_pages": statistics['total_pages'],
            "dpi": DPI,
            "fuzzy_threshold": FUZZY_THRESHOLD
        },
        "statistics": dict(statistics),
        "objects": all_objects,
        "texts": all_texts,
        "text_groups": groups_export,
        "pages": raw_pages
    }, f, indent=2)
print(f"  ✓ Raw data dump: {os.path.basename(raw_dump)}")

# ----- Summary Report (TXT) -----
summary_txt = os.path.join(REP_DIR, f"summary_{timestamp}.txt")
with open(summary_txt, "w") as f:
    f.write("="*80 + "\n")
    f.write("STEEL DETAILING PDF ANALYSIS SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"File: {os.path.basename(PDF_PATH)}\n")
    f.write(f"Analysis Date: {timestamp}\n")
    f.write(f"Total Pages: {statistics['total_pages']}\n\n")
    
    f.write("-"*80 + "\n")
    f.write("OBJECTS\n")
    f.write("-"*80 + "\n")
    f.write(f"Total Objects: {statistics['total_objects']}\n\n")
    f.write("By Type:\n")
    for obj_type, count in sorted(statistics['objects_by_type'].items(), 
                                   key=lambda x: x[1], reverse=True):
        f.write(f"  {obj_type}: {count}\n")
    
    f.write("\n" + "-"*80 + "\n")
    f.write("TEXT\n")
    f.write("-"*80 + "\n")
    f.write(f"Total Text Items: {statistics['total_texts']}\n")
    f.write(f"Unique Groups: {len(groups)}\n\n")
    f.write("By Source:\n")
    for source, count in statistics['texts_by_source'].items():
        f.write(f"  {source}: {count}\n")
    
    f.write("\n" + "-"*80 + "\n")
    f.write("TOP 20 MOST FREQUENT TEXT\n")
    f.write("-"*80 + "\n")
    sorted_groups = sorted(groups.values(), key=lambda x: x["count"], reverse=True)
    for idx, g in enumerate(sorted_groups[:20], 1):
        f.write(f"{idx:2d}. \"{g['original']}\" - Count: {g['count']}\n")
    
    f.write("\n" + "="*80 + "\n")

print(f"  ✓ Summary report: {os.path.basename(summary_txt)}")

# ================== COMPLETION ==================
print(f"\n{'='*80}")
print("✅ EXTRACTION COMPLETE!")
print(f"{'='*80}")
print(f"\nResults saved to: {os.path.abspath(OUT_DIR)}")
print(f"\nQuick Stats:")
print(f"  • Pages processed: {statistics['total_pages']}")
print(f"  • Objects found: {statistics['total_objects']}")
print(f"  • Text items: {statistics['total_texts']}")
print(f"  • Unique text groups: {len(groups)}")
print(f"\nOutput folders:")
print(f"  • Images: {IMG_DIR}")
print(f"  • Visualizations: {VIS_DIR}")
print(f"  • Reports: {REP_DIR}")
print(f"  • Raw data: {RAW_DIR}")
print()
# ============================================================================
# ADD THIS SECTION AT THE VERY END (after "EXTRACTION COMPLETE")
# ============================================================================

# ================== SEMANTIC ANALYSIS ==================
print(f"\n{'='*80}")
print("STARTING SEMANTIC ANALYSIS")
print(f"{'='*80}")

# Prepare data for semantic pipeline
semantic_input = {
    'objects': all_objects,
    'texts': all_texts,
    'text_groups': groups_export,
    'pages': raw_pages
}

# Run semantic pipeline
pipeline = SemanticPipeline(semantic_input)
semantic_results = pipeline.process()

# Save semantic reports
print(f"\n{'─'*80}")
print("Saving semantic reports...")
print(f"{'─'*80}")
pipeline.save_semantic_reports(REP_DIR)

# ================== ENHANCED VISUALIZATIONS ==================
print(f"\n{'─'*80}")
print("Creating semantic visualizations...")
print(f"{'─'*80}")

def visualize_semantic_objects(img_path, semantic_objects, output_path):
    """Draw semantic objects with type-specific colors"""
    from PIL import Image, ImageDraw, ImageFont
    
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    
    # Semantic type colors
    colors = {
        'BEAM_LINE': (255, 0, 0),           # Red
        'COLUMN_LINE': (0, 0, 255),         # Blue
        'BOLT_CIRCLE': (255, 165, 0),       # Orange
        'LEADER_LINE': (128, 128, 128),     # Gray
        'DETAIL_BOX': (0, 255, 0),          # Green
        'GRID_LINE': (200, 200, 200),       # Light gray
        'TABLE_BORDER': (150, 150, 150),    # Gray
        'TITLE_BLOCK': (100, 100, 100),     # Dark gray
    }
    
    for obj in semantic_objects:
        bbox = obj['bbox']
        semantic_type = obj['semantic_type']
        
        # Skip non-structural in visualization
        if semantic_type in ['TITLE_BLOCK', 'GRID_LINE', 'TABLE_BORDER']:
            continue
        
        color = colors.get(semantic_type, (128, 0, 128))
        
        # Draw bounding box
        draw.rectangle(bbox, outline=color, width=3)
        
        # Add label with linked text if available
        label_parts = [semantic_type]
        if 'linked_text' in obj and obj['linked_text']:
            label_parts.append(f": {obj['linked_text'][0]}")
        
        label = "".join(label_parts)
        
        # Position label above bbox
        text_y = max(0, bbox[1] - 20)
        draw.text((bbox[0], text_y), label, fill=color)
    
    img.save(output_path)

def visualize_members(img_path, semantic_objects, members, output_path):
    """Visualize member locations and groupings"""
    from PIL import Image, ImageDraw
    
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    
    # Member type colors
    member_colors = {
        'WIDE_FLANGE': (255, 0, 0),
        'CHANNEL': (0, 255, 0),
        'ANCHOR_ROD': (255, 165, 0),
        'PLATE': (0, 0, 255),
        'FASTENER': (255, 0, 255),
        'HSS_TUBE': (0, 255, 255),
        'ANGLE': (255, 255, 0),
        'OTHER': (128, 128, 128)
    }
    
    # Draw each member's geometry
    for member in members:
        if 'geometry_ids' not in member or not member['geometry_ids']:
            continue
        
        member_type = member.get('member_type', 'OTHER')
        color = member_colors.get(member_type, (128, 128, 128))
        
        # Find and draw all geometry for this member
        for geom_id in member['geometry_ids']:
            geom = next((obj for obj in semantic_objects if obj['id'] == geom_id), None)
            if geom:
                bbox = geom['bbox']
                draw.rectangle(bbox, outline=color, width=4)
                
                # Label with member ID
                draw.text((bbox[0], bbox[1] - 25), 
                         f"{member['id']} ({member_type})", 
                         fill=color)
    
    img.save(output_path)

# Create semantic visualizations for each page
for page_num in range(1, len(doc) + 1):
    img_path = os.path.join(IMG_DIR, f"page_{page_num}.png")
    
    # Filter objects and members for this page
    page_objects = [obj for obj in semantic_results['semantic_objects'] 
                    if obj['page'] == page_num]
    page_members = [m for m in semantic_results['structural_members'] 
                    if any(loc['page'] == page_num for loc in m['locations'])]
    
    # Semantic objects visualization
    sem_vis_path = os.path.join(VIS_DIR, f"page_{page_num}_semantic.png")
    visualize_semantic_objects(img_path, page_objects, sem_vis_path)
    
    # Members visualization
    mem_vis_path = os.path.join(VIS_DIR, f"page_{page_num}_members.png")
    visualize_members(img_path, page_objects, page_members, mem_vis_path)
    
    print(f"  ✓ Created semantic visualizations for page {page_num}")

# ================== FINAL STATISTICS ==================
print(f"\n{'='*80}")
print("✅ COMPLETE ANALYSIS FINISHED!")
print(f"{'='*80}")

print(f"\n📊 RAW EXTRACTION STATISTICS:")
print(f"  • Pages processed: {statistics['total_pages']}")
print(f"  • Raw objects: {statistics['total_objects']}")
print(f"  • Text items: {statistics['total_texts']}")
print(f"  • Text groups: {len(groups)}")

print(f"\n🎯 SEMANTIC ANALYSIS STATISTICS:")
stats = semantic_results['statistics']
print(f"  • Semantic objects: {stats['total_semantic_objects']}")
print(f"  • Structural objects: {stats['structural_objects']}")
print(f"  • Non-structural filtered: {stats['non_structural_filtered']}")
print(f"  • Text-geometry links: {stats['text_geometry_links']}")
print(f"  • Unique members: {stats['total_members']}")
print(f"  • Structural members: {stats['structural_members']}")

print(f"\n📁 OUTPUT STRUCTURE:")
print(f"  {OUT_DIR}/")
print(f"  ├── images/              # Original page renders")
print(f"  ├── visualizations/")
print(f"  │   ├── *_objects.png    # Raw object detection")
print(f"  │   ├── *_text.png       # Text bounding boxes")
print(f"  │   ├── *_semantic.png   # ✨ Semantic classification")
print(f"  │   └── *_members.png    # ✨ Member groupings")
print(f"  ├── reports/")
print(f"  │   ├── object_report_*.csv/json")
print(f"  │   ├── text_report_*.csv/json")
print(f"  │   ├── member_summary_*.csv/json")
print(f"  │   ├── semantic_objects_*.json      # ✨ NEW")
print(f"  │   ├── structural_geometry_*.json   # ✨ NEW")
print(f"  │   ├── geometry_text_links_*.json   # ✨ NEW")
print(f"  │   ├── semantic_summary_*.txt       # ✨ NEW")
print(f"  │   └── summary_*.txt")
print(f"  └── raw_dump/")
print(f"      ├── page_*.json")
print(f"      └── all_data_*.json")

print(f"\n💡 KEY IMPROVEMENTS:")
print(f"  ✅ Moved from s/f operators → semantic object types")
print(f"  ✅ Extracted meaningful features (length, orientation, etc.)")
print(f"  ✅ Classified geometry into steel-specific categories")
print(f"  ✅ Linked text labels to nearby geometry")
print(f"  ✅ Grouped elements into member abstractions")
print(f"  ✅ Filtered non-structural elements")
print(f"  ✅ Generated semantic reports and visualizations")

print(f"\n🔍 MEMBER TYPES DETECTED:")
for mtype, count in sorted(stats['member_types'].items(), key=lambda x: x[1], reverse=True):
    print(f"  • {mtype}: {count}")

print(f"\n📈 TOP 10 STRUCTURAL MEMBERS:")
top_members = sorted(semantic_results['structural_members'], 
                    key=lambda x: x['occurrences'], reverse=True)[:10]
for idx, member in enumerate(top_members, 1):
    print(f"  {idx:2d}. {member['id']:15s} - {member['occurrences']:3d}x "
          f"({member.get('member_type', 'UNKNOWN')})")

print(f"\n{'='*80}")
print(f"Analysis results saved to: {os.path.abspath(OUT_DIR)}")
print(f"{'='*80}\n")