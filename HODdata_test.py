#!/usr/bin/env python3
"""SAM and Qwen3-VL inference on the HOD dataset."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "hod"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import sys
import numpy as np
import cv2
import torch
import re
import json
import random
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# Configuration
HOD_ROOT = str(DATA_DIR / "hod" / "class")

OUTPUT_NPY_DIR = str(OUTPUT_DIR / "arrays")
OUTPUT_INTERMEDIATE_DIR = str(OUTPUT_DIR / "intermediate")
os.makedirs(OUTPUT_NPY_DIR, exist_ok=True)
os.makedirs(OUTPUT_INTERMEDIATE_DIR, exist_ok=True)

OUTPUT_CSV = str(OUTPUT_DIR / "metrics.csv")
SUMMARY_OUT = str(OUTPUT_DIR / "summary.txt")
ERROR_LOG = str(OUTPUT_DIR / "errors.txt")

SAM_CHECKPOINT = os.environ.get("SAM_CHECKPOINT", "/home/hangsheng/hdd/hangsheng/workspace/toxic_proj/sam_vit_h.pth")
SAM_MODEL_TYPE = "vit_h"
QWEN_MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", '/home/hangsheng/models/Qwen/Qwen3-VL-8B-Instruct')

CATEGORY_PROMPTS = {
    "alcohol": ["alcohol bottle", "beer can", "wine glass", "liquor bottle", "drink"],
    "blood": ["blood", "wound", "injury", "bleeding", "gore"],
    "cigarette": ["cigarette", "smoking", "tobacco", "cigar"],
    "gun": ["gun", "firearm", "pistol", "rifle", "weapon", "handgun"],
    "insulting_gesture": ["middle finger", "insulting gesture", "offensive hand gesture", "rude gesture"],
    "knife": ["knife", "blade", "dagger", "sword", "machete"],
}

ALL_DETECTION_PROMPTS = list({word for words in CATEGORY_PROMPTS.values() for word in words})

SCORING_PROMPT = '''Detect whether there are harmful objects in the image. The target categories are: alcohol, insulting gesture, blood (gore), cigarette, gun, knife.
If the image involves any of the above objects, output 1; otherwise, output 0.
Please ensure you output only a single digit, either 1 or 0, without any additional content.
'''
SYSTEM_MESSAGE = "You are a professional content moderation expert."

THRESHOLD = 0.5  # Prediction threshold
MASK_SCORE_THRESHOLD = 0.5
SAM_DEVICE = "cuda:0"
MAX_IMAGE_SIZE = 512
SAMPLE_RATIO = 0.3

SAM_AUTO = SamAutomaticMaskGenerator(
    sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(device=SAM_DEVICE),
    points_per_side=32, pred_iou_thresh=0.86, stability_score_thresh=0.92, min_mask_region_area=100
)

error_count = 0
def log_error(msg):
    global error_count
    error_count += 1
    with open(ERROR_LOG, 'a') as f:
        f.write(f"[{error_count}] {msg}\n")

# Load models
print("Loading models...")
qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
    QWEN_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
)
qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_PATH)
qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH)
sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=SAM_DEVICE)
sam_predictor = SamPredictor(sam)
print("Models loaded!")

def resize_image(image, max_size=MAX_IMAGE_SIZE):
    try:
        if max(image.size) > max_size:
            scale = max_size / max(image.size)
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.LANCZOS)
        return image
    except Exception as e:
        log_error(f"resize_image error: {e}")
        return image

def get_logit_score(image_pil):
    try:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": [{"type": "image", "image": image_pil}, {"type": "text", "text": SCORING_PROMPT}]}
        ]
        text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = qwen_processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(qwen_model.device)
        
        with torch.no_grad():
            outputs = qwen_model(**inputs)
        logits = outputs.logits[0, -1, :]
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            log_error("logits contains NaN/Inf")
            return 0.0
        
        probs = torch.nn.functional.softmax(logits, dim=0)
        token_1 = qwen_tokenizer.encode("1", add_special_tokens=False)[0]
        score = probs[token_1].item()
        
        if np.isnan(score) or np.isinf(score) or score < 0 or score > 1:
            log_error(f"Invalid score: {score}")
            return 0.0
        return score
    except Exception as e:
        log_error(f"get_logit_score error: {e}")
        return 0.0

def get_logit_score_cropped(image_bgr, x1, y1, x2, y2):
    try:
        cropped = image_bgr[y1:y2, x1:x2]
        if cropped.size == 0:
            return 0.0
        pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        return get_logit_score(resize_image(pil))
    except Exception as e:
        log_error(f"get_logit_score_cropped error: {e}")
        return 0.0

def parse_bbox_text(output_text, img_w, img_h):
    """Extract bounding boxes from the model response."""
    bboxes = []
    try:
        bbox_pattern = r'<bbox>(\d+)<\/bbox>'
        matches = re.findall(bbox_pattern, output_text)
        for match in matches:
            coords = [int(x) for x in match.split()]
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                x1 = x1 * img_w / 1000.0
                y1 = y1 * img_h / 1000.0
                x2 = x2 * img_w / 1000.0
                y2 = y2 * img_h / 1000.0
                bboxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        
        if not bboxes:
            coord_pattern = r'\[(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\]'
            matches = re.findall(coord_pattern, output_text)
            for match in matches:
                x1, y1, x2, y2 = [int(c) for c in match]
                x1 = x1 * img_w / 1000.0
                y1 = y1 * img_h / 1000.0
                x2 = x2 * img_w / 1000.0
                y2 = y2 * img_h / 1000.0
                bboxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        
        if not bboxes:
            bbox_2d_pattern = r'bbox_2d[:\s]*\[(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\]'
            matches = re.findall(bbox_2d_pattern, output_text)
            for match in matches:
                x1, y1, x2, y2 = [int(c) for c in match]
                x1 = x1 * img_w / 1000.0
                y1 = y1 * img_h / 1000.0
                x2 = x2 * img_w / 1000.0
                y2 = y2 * img_h / 1000.0
                bboxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        
        if not bboxes:
            numbers = re.findall(r'\b(\d{2,3})\b', output_text)
            if len(numbers) >= 4:
                for i in range(0, len(numbers) - 3, 4):
                    x1, y1, x2, y2 = [int(n) for n in numbers[i:i+4]]
                    x1 = x1 * img_w / 1000.0
                    y1 = y1 * img_h / 1000.0
                    x2 = x2 * img_w / 1000.0
                    y2 = y2 * img_h / 1000.0
                    bboxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
    except Exception as e:
        log_error(f"Parse bbox text error: {e}, text: {output_text[:100]}")
    return bboxes

def qwen_detect_targets(image_path, target_words):
    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
    except Exception as e:
        log_error(f"Cannot open image {image_path}: {e}")
        return []
    
    targets = ", ".join(target_words)
    prompt_text = f"Detect: {targets}. Output bounding boxes as [x1, y1, x2, y2]."
    messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_text}]}]
    
    try:
        text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = qwen_processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(qwen_model.device)
        
        with torch.no_grad():
            gen_ids = qwen_model.generate(**inputs, max_new_tokens=512)
        output_text = qwen_processor.batch_decode([gen_ids[0][len(inputs.input_ids[0]):]], skip_special_tokens=False)[0]
    except Exception as e:
        log_error(f"Qwen generate error for {image_path}: {e}")
        torch.cuda.empty_cache()
        return []
    
    bboxes = parse_bbox_text(output_text, img_w, img_h)
    return bboxes

def parse_xml_bbox(xml_path):
    try:
        tree = ET.parse(xml_path)
        return [[int(b.find(t).text) for t in ['xmin', 'ymin', 'xmax', 'ymax']] for b in tree.getroot().findall('.//bndbox')]
    except:
        return []

def semantic_segment_branch(image_bgr, bboxes):
    h, w = image_bgr.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    if not bboxes:
        return heatmap, 0
    
    try:
        sam_predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    except Exception as e:
        log_error(f"SAM set_image error: {e}")
        return heatmap, 0
    
    count = 0
    for x1, y1, x2, y2 in bboxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        try:
            masks, scores, _ = sam_predictor.predict(box=np.array([x1, y1, x2, y2])[None, :], multimask_output=True)
            if scores.max() >= MASK_SCORE_THRESHOLD:
                mask = masks[np.argmax(scores)]
                score = get_logit_score_cropped(image_bgr, x1, y1, x2, y2)
                heatmap[mask] += score * scores.max()
                count += 1
        except Exception as e:
            log_error(f"SAM predict error: {e}")
    return heatmap, count

def full_segment_branch(image_bgr, original_score):
    h, w = image_bgr.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    try:
        masks = SAM_AUTO.generate(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        for m in masks[:5]:
            mask = m['segmentation']
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            chopped = image_bgr.copy()
            chopped[~mask] = 0
            cropped = chopped[y1:y2, x1:x2]
            if cropped.size > 0:
                score = get_logit_score(Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)))
                heatmap[mask] += score * m.get('predicted_iou', 1.0)
    except Exception as e:
        log_error(f"Full segment error: {e}")
    return heatmap

def normalize_heatmap(heatmap):
    """Normalize the heatmap values."""
    min_val = heatmap.min()
    max_val = heatmap.max()
    if max_val > min_val:
        return (heatmap - min_val) / (max_val - min_val)
    else:
        return heatmap

def main():
    print("=" * 70)
    print("HOD Dataset - Inference")
    print("=" * 70)
    
    # Use the exact sampled IDs from the original March 28 run to ensure comparability
    SAMPLED_IDS_FILE = str(DATA_DIR / "hod" / "sampled_ids.txt")
    id_set = set()
    if os.path.exists(SAMPLED_IDS_FILE):
        with open(SAMPLED_IDS_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    id_set.add(line)
        print(f"Loaded {len(id_set)} sampled IDs from {SAMPLED_IDS_FILE}")

    all_images = []
    for category in sorted(os.listdir(HOD_ROOT)):
        cat_path = os.path.join(HOD_ROOT, category)
        if not os.path.isdir(cat_path):
            continue
        for subdir in ["hard_cases", "normal_cases"]:
            sub_path = os.path.join(cat_path, subdir)
            if not os.path.isdir(sub_path):
                continue
            img_dir = os.path.join(sub_path, "jpg")
            xml_dir = os.path.join(sub_path, "xml")
            if not os.path.isdir(img_dir):
                continue
            for fname in sorted(os.listdir(img_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    entry_id = f"{category}_{subdir}_{fname}"
                    if id_set and entry_id not in id_set:
                        continue
                    all_images.append((category, subdir, fname, img_dir, xml_dir))

    if id_set:
        sampled_images = all_images
        sample_size = len(sampled_images)
        print(f"Using fixed sample list: {sample_size} images")
    else:
        sample_size = int(len(all_images) * SAMPLE_RATIO)
        random.seed(42)
        sampled_images = random.sample(all_images, sample_size)
    print(f"\nAvailable images: {len(all_images)}, selected images: {sample_size}")
    
    per_image_ious = []
    total_detected = 0
    total_gt_objs = 0
    total_has_pred = 0
    total_precise = 0
    category_stats = {}
    processed_count = 0
    error_image_count = 0
    
    existing_npy = set()
    for root, dirs, files in os.walk(OUTPUT_NPY_DIR):
        for f in files:
            if f.endswith('_heatmap.npy'):
                existing_npy.add(f)
    
    print(f"Previously processed images: {len(existing_npy)}")
    
    with open(OUTPUT_CSV, 'a' if os.path.exists(OUTPUT_CSV) else 'w') as f:
        if f.tell() == 0:
            f.write("category,subdir,image_name,sem_masks,full_masks,original_score\n")
    
    for idx, (category, subdir, fname, img_dir, xml_dir) in enumerate(tqdm(sampled_images, desc="Processing")):
        img_path = os.path.join(img_dir, fname)
        xml_path = os.path.join(xml_dir, os.path.splitext(fname)[0] + ".xml")
        base_name = os.path.splitext(fname)[0]
        
        out_subdir = os.path.join(OUTPUT_NPY_DIR, category, subdir)
        inter_subdir = os.path.join(OUTPUT_INTERMEDIATE_DIR, category, subdir)
        os.makedirs(out_subdir, exist_ok=True)
        os.makedirs(inter_subdir, exist_ok=True)
        
        npy_path = os.path.join(out_subdir, f"{base_name}_heatmap.npy")
        
        if os.path.exists(npy_path):
            continue
        
        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                log_error(f"Cannot read image: {img_path}")
                error_image_count += 1
                continue
        except Exception as e:
            log_error(f"Image read error {img_path}: {e}")
            error_image_count += 1
            continue
        
        h_img, w_img = img_bgr.shape[:2]
        
        try:
            original_score = get_logit_score(resize_image(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))))
        except Exception as e:
            log_error(f"Original score error: {e}")
            original_score = 0.0
        
        try:
            detected_boxes = qwen_detect_targets(img_path, ALL_DETECTION_PROMPTS)
            num_semantic_masks = len(detected_boxes)
        except Exception as e:
            log_error(f"Detect targets error: {e}")
            detected_boxes = []
            num_semantic_masks = 0
        
        try:
            semantic_heatmap, num_valid_semantic = semantic_segment_branch(img_bgr, detected_boxes)
        except Exception as e:
            log_error(f"Semantic branch error: {e}")
            semantic_heatmap = np.zeros((h_img, w_img), dtype=np.float32)
            num_valid_semantic = 0
        
        try:
            full_heatmap = full_segment_branch(img_bgr, original_score)
        except Exception as e:
            log_error(f"Full segment error: {e}")
            full_heatmap = np.zeros((h_img, w_img), dtype=np.float32)

        semantic_normalized = normalize_heatmap(semantic_heatmap)
        full_normalized = normalize_heatmap(full_heatmap)

        semantic_binary = (semantic_normalized > THRESHOLD).astype(np.uint8)
        full_binary = (full_normalized > THRESHOLD).astype(np.uint8)

        final_binary = np.logical_or(semantic_binary, full_binary).astype(np.uint8)

        toxicity_map = final_binary.astype(np.float32) * original_score
        
        # Save intermediate outputs
        try:
            np.save(npy_path, toxicity_map.astype(np.float32))
            np.save(os.path.join(inter_subdir, f"{base_name}_semantic_heatmap.npy"), semantic_heatmap.astype(np.float32))
            np.save(os.path.join(inter_subdir, f"{base_name}_full_heatmap.npy"), full_heatmap.astype(np.float32))
            np.save(os.path.join(inter_subdir, f"{base_name}_semantic_normalized.npy"), semantic_normalized.astype(np.float32))
            np.save(os.path.join(inter_subdir, f"{base_name}_full_normalized.npy"), full_normalized.astype(np.float32))
            np.save(os.path.join(inter_subdir, f"{base_name}_semantic_binary.npy"), semantic_binary)
            np.save(os.path.join(inter_subdir, f"{base_name}_full_binary.npy"), full_binary)
            np.save(os.path.join(inter_subdir, f"{base_name}_union_binary.npy"), final_binary)
        except Exception as e:
            log_error(f"Save error: {e}")
        
        try:
            with open(OUTPUT_CSV, 'a') as f:
                f.write(f"{category},{subdir},{fname},{num_valid_semantic},{np.sum(full_binary > 0)},{original_score:.6f}\n")
        except Exception as e:
            log_error(f"CSV write error: {e}")
        
        # Evaluate predictions
        try:
            gt_bboxes = parse_xml_bbox(xml_path)
            gt_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            for box in gt_bboxes:
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                x1, y1 = max(0, x1), max(0, int(y1))
                x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))
                if x2 > x1 and y2 > y1:
                    gt_mask[y1:y2, x1:x2] = 1
            
            inter = np.logical_and(final_binary, gt_mask).sum()
            union = np.logical_or(final_binary, gt_mask).sum()
            iou = inter / union if union > 0 else 0.0
            per_image_ious.append(iou)
            
            if category not in category_stats:
                category_stats[category] = {'ious': [], 'detected': 0, 'total_gt': 0, 'precise': 0, 'has_pred': 0}
            category_stats[category]['ious'].append(iou)
            
            for box in gt_bboxes:
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                x1, y1 = max(0, x1), max(0, int(y1))
                x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                total_gt_objs += 1
                category_stats[category]['total_gt'] += 1
                if final_binary[y1:y2, x1:x2].sum() / ((x2-x1)*(y2-y1)) >= 0.5:
                    total_detected += 1
                    category_stats[category]['detected'] += 1
            
            has_pred = final_binary.sum() > 0
            if has_pred:
                total_has_pred += 1
                category_stats[category]['has_pred'] += 1
                pred_in_gt = np.logical_and(final_binary.astype(bool), gt_mask.astype(bool)).sum()
                pred_total = final_binary.sum()
                if pred_total > 0 and pred_in_gt / pred_total >= 0.5:
                    total_precise += 1
                    category_stats[category]['precise'] += 1
        except Exception as e:
            log_error(f"Evaluation error: {e}")
        
        processed_count += 1
        
        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    gIoU = np.mean(per_image_ious) if per_image_ious else 0
    recall = total_detected / total_gt_objs if total_gt_objs > 0 else 0
    precision = total_precise / total_has_pred if total_has_pred > 0 else 0
    
    summary_lines = [
        "=" * 70,
        "HOD Dataset - Results",
        f"Threshold={THRESHOLD}, Sample={SAMPLE_RATIO}",
        f"Processed: {processed_count}, Errors: {error_image_count}",
        "=" * 70,
        f"Total Images: {len(per_image_ious)}",
        f"gIoU: {gIoU:.6f}",
        f"Recall@{THRESHOLD}: {recall:.6f} ({total_detected}/{total_gt_objs})",
        f"Precision@{THRESHOLD}: {precision:.6f} ({total_precise}/{total_has_pred})",
        "",
        "=" * 50,
        "Per-Category:",
        "=" * 50,
    ]
    
    for cat in sorted(category_stats.keys()):
        s = category_stats[cat]
        cat_giou = np.mean(s['ious']) if s['ious'] else 0
        cat_recall = s['detected'] / s['total_gt'] if s['total_gt'] > 0 else 0
        cat_precision = s['precise'] / s['has_pred'] if s['has_pred'] > 0 else 0
        summary_lines.append(f"\n[{cat}]")
        summary_lines.append(f"  Images: {len(s['ious'])}")
        summary_lines.append(f"  gIoU: {cat_giou:.4f}")
        summary_lines.append(f"  Recall: {cat_recall:.4f}")
        summary_lines.append(f"  Precision: {cat_precision:.4f}")
    
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    
    with open(SUMMARY_OUT, 'w') as f:
        f.write(summary_text)
    
    print(f"\nDone! Summary: {SUMMARY_OUT}")
    print(f"Errors logged to: {ERROR_LOG}")
    print(f"Total errors: {error_count}")

if __name__ == "__main__":
    main()
