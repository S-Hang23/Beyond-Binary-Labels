#!/usr/bin/env python3
"""SAM and Qwen3-VL inference on the Own dataset."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "own"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

import numpy as np
import torch
import cv2
import re
import json
import random
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# Configuration
IMAGE_DIRS = [
    str(DATA_DIR / "own" / "images" / "illegal"),
    str(DATA_DIR / "own" / "images" / "porn"),
    str(DATA_DIR / "own" / "images" / "violence"),
    str(DATA_DIR / "own" / "images" / "others"),
]
GT_JSON = str(DATA_DIR / "own" / "annotations.json")

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

CATEGORY_DETECTION_PROMPTS = {
    "porn": ["person", "human body", "skin", "standing person", "lying person", "sitting person", "human figure", "body"],
    "illegal": ["gun", "firearm", "pistol", "rifle", "drugs", "drug paraphernalia", "syringe", "pills", "gambling chips", "gambling table", "weapon", "knife", "blade"],
    "violence": ["blood", "wound", "injury", "corpse", "dead body", "bleeding person", "weapon", "gun", "knife", "fire", "explosion", "fight", "attack"],
    "others": ["middle finger", "insulting gesture", "extremism symbol", "hate symbol", "offensive sign", "gun", "blood", "drugs"],
}

ALL_DETECTION_PROMPTS = list({word for words in CATEGORY_DETECTION_PROMPTS.values() for word in words})

SCORING_PROMPT = '''
Does the image contain inappropriate, pornographic, violent, horrific, bloody, gambling, drug-related, or superstitious content that violates laws and regulations? If so, output 1; otherwise, output 0. Detailed rules are as follows:

Pornography and Nudity
    Image contains direct exposure of sexual organs or sexual acts
    Image contains sexual suggestions involving minors (even without direct nudity)
    Image contains excessive nudity or provocative behavior not for artistic/medical purposes

Violence and Gore
    Image contains corpses, dismembered body parts, large amounts of blood, or other visually disturbing content
    Image contains murder, abuse, war, or fighting scenes
    Image contains explicit depictions or suggestions of self-harm or suicide

Illegal and Dangerous Activities
    Image contains scenes of drug production, trafficking, or consumption
    Image contains the use or display of illegal weapons (such as firearms, explosives, and regulated knives)
    Image contains depictions of theft, robbery, arson, gambling, or other illegal activities

Hate and Discrimination
    Image contains stereotypical derogation, discrimination, or hateful content (such as racial discrimination, desecration of religious symbols, offensive gestures, etc.)
    Image contains Nazi symbols, extremism, or cult-related content

If the image involves any of the above categories, it is considered illegal and inappropriate content, output 1; otherwise, output 0.
Please ensure you output only a single digit, either 1 or 0, without any additional content.
'''
SYSTEM_MESSAGE = "You are a content moderation expert."

THRESHOLD = 0.5
MASK_SCORE_THRESHOLD = 0.5
SAM_DEVICE = "cuda:0"
MAX_IMAGE_SIZE = 512
SAMPLE_RATIO = 1.0  # Use all annotated images

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

# Load ground-truth annotations
print("Loading GT...")
with open(GT_JSON) as f:
    coco = json.load(f)
id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
gt_map = {}
for ann in coco['annotations']:
    fname = id_to_filename.get(ann['image_id'])
    if fname:
        x, y, w, h = ann['bbox']
        gt_map.setdefault(fname, []).append([x, y, x+w, y+h])

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

# Helper functions
def resize_image(image, max_size=MAX_IMAGE_SIZE):
    if max(image.size) > max_size:
        scale = max_size / max(image.size)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.LANCZOS)
    return image

def get_logit_score(image_pil):
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
    probs = torch.nn.functional.softmax(logits, dim=0)
    token_1 = qwen_tokenizer.encode("1", add_special_tokens=False)[0]
    return probs[token_1].item()

def get_logit_score_cropped(image_bgr, x1, y1, x2, y2):
    cropped = image_bgr[y1:y2, x1:x2]
    if cropped.size == 0:
        return 0.0
    pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    return get_logit_score(resize_image(pil))

def parse_bbox_text(output_text, img_w, img_h):
    bboxes = []
    try:
        bbox_pattern = r'<bbox>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\u003c\/bbox>'
        matches = re.findall(bbox_pattern, output_text)
        for match in matches:
            x1, y1, x2, y2 = [int(c) for c in match]
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
    except:
        pass
    return bboxes

def qwen_detect_targets(image_path, target_words):
    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
    except:
        return []
    targets = ", ".join(target_words)
    prompt_text = f"Detect: {targets}. Output bounding boxes."
    messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_text}]}]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = qwen_processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(qwen_model.device)
    with torch.no_grad():
        gen_ids = qwen_model.generate(**inputs, max_new_tokens=512)
    output_text = qwen_processor.batch_decode([gen_ids[0][len(inputs.input_ids[0]):]], skip_special_tokens=False)[0]
    return parse_bbox_text(output_text, img_w, img_h)

def semantic_segment_branch(image_bgr, bboxes):
    h, w = image_bgr.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    if not bboxes:
        return heatmap, 0
    sam_predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
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
        except:
            pass
    return heatmap, count

def full_segment_branch(image_bgr, original_score):
    h, w = image_bgr.shape[:2]
    heatmap = np.zeros((h, w), np.float32)
    try:
        masks = SAM_AUTO.generate(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        for m in masks[:5]:
            mask = m['segmentation']
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            # Prepare the masked image and its counterpart
            masked_image = image_bgr.copy()
            masked_image[mask] = 0  # Hide the selected region
            chopped_image = image_bgr.copy()
            chopped_image[~mask] = 0  # Keep the selected region
            # Score both images
            masked_score = get_logit_score(resize_image(Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))))
            chopped_score = get_logit_score(resize_image(Image.fromarray(cv2.cvtColor(chopped_image, cv2.COLOR_BGR2RGB))))
            # Accumulate weighted scores
            mask_01 = mask.astype(np.uint8)
            inverted_mask = 1 - mask_01
            iou_weight = m.get('predicted_iou', 1.0)
            heatmap += masked_score * inverted_mask * iou_weight
            heatmap += chopped_score * mask_01 * iou_weight
    except:
        pass
    return heatmap

def save_heatmap_vis(arr, save_path, cmap='jet', vmin=None, vmax=None):
    try:
        plt.figure(figsize=(8, 6))
        plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        log_error(f"Visualization save error: {e}")

def normalize_heatmap(heatmap):
    min_val = heatmap.min()
    max_val = heatmap.max()
    if max_val > min_val:
        return (heatmap - min_val) / (max_val - min_val)
    return heatmap

def main():
    print("=" * 70)
    print("Own Dataset - Inference")
    print("=" * 70)
    
    file_index = {}
    for d in IMAGE_DIRS:
        if not os.path.isdir(d):
            continue
        cat = os.path.basename(d)
        for f in os.listdir(d):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_index[f] = (os.path.join(d, f), cat)
    
    all_images = [(fname, *file_index[fname]) for fname in gt_map if fname in file_index]
    
    # Use all annotated images
    sampled_images = all_images
    print(f"\nImages with ground-truth annotations: {len(all_images)}")
    
    per_image_ious = []
    total_detected = 0
    total_gt_objs = 0
    total_has_pred = 0
    total_precise = 0
    category_stats = {}
    
    with open(OUTPUT_CSV, 'w') as f:
        f.write("category,image_name,sem_masks,full_masks,original_score\n")
    
    for idx, (fname, img_path, category) in enumerate(tqdm(sampled_images, desc="Processing")):
        gt_bboxes = gt_map[fname]
        base_name = os.path.splitext(fname)[0]
        
        out_subdir = os.path.join(OUTPUT_NPY_DIR, category)
        inter_subdir = os.path.join(OUTPUT_INTERMEDIATE_DIR, category)
        os.makedirs(out_subdir, exist_ok=True)
        os.makedirs(inter_subdir, exist_ok=True)
        
        npy_path = os.path.join(out_subdir, f"{base_name}_heatmap.npy")
        sem_heat_path = os.path.join(inter_subdir, f"{base_name}_semantic_heatmap.npy")
        full_heat_path = os.path.join(inter_subdir, f"{base_name}_full_heatmap.npy")
        sem_norm_path = os.path.join(inter_subdir, f"{base_name}_semantic_normalized.npy")
        full_norm_path = os.path.join(inter_subdir, f"{base_name}_full_normalized.npy")
        union_path = os.path.join(inter_subdir, f"{base_name}_union_binary.npy")

        skip_model = False
        if (os.path.exists(npy_path) and os.path.exists(sem_heat_path)
                and os.path.exists(full_heat_path) and os.path.exists(sem_norm_path)
                and os.path.exists(full_norm_path) and os.path.exists(union_path)):
            toxicity_map = np.load(npy_path).astype(np.float32)
            semantic_heatmap = np.load(sem_heat_path).astype(np.float32)
            full_heatmap = np.load(full_heat_path).astype(np.float32)
            semantic_normalized = np.load(sem_norm_path).astype(np.float32)
            full_normalized = np.load(full_norm_path).astype(np.float32)
            final_binary = np.load(union_path).astype(np.uint8)
            semantic_binary = (semantic_normalized > THRESHOLD).astype(np.uint8)
            full_binary = (full_normalized > THRESHOLD).astype(np.uint8)
            h_img, w_img = toxicity_map.shape
            original_score = float(toxicity_map.max()) if final_binary.sum() > 0 else 0.0
            num_valid_semantic = 0
            skip_model = True

        if not skip_model:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            h_img, w_img = img_bgr.shape[:2]

            original_score = get_logit_score(resize_image(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))))

            detected_boxes = qwen_detect_targets(img_path, ALL_DETECTION_PROMPTS)

            semantic_heatmap, num_valid_semantic = semantic_segment_branch(img_bgr, detected_boxes)
            full_heatmap = full_segment_branch(img_bgr, original_score)

            semantic_normalized = normalize_heatmap(semantic_heatmap)
            full_normalized = normalize_heatmap(full_heatmap)

            semantic_binary = (semantic_normalized > THRESHOLD).astype(np.uint8)
            full_binary = (full_normalized > THRESHOLD).astype(np.uint8)
            final_binary = np.logical_or(semantic_binary, full_binary).astype(np.uint8)
            toxicity_map = final_binary.astype(np.float32) * original_score

            np.save(npy_path, toxicity_map.astype(np.float32))
            np.save(sem_heat_path, semantic_heatmap.astype(np.float32))
            np.save(full_heat_path, full_heatmap.astype(np.float32))
            np.save(sem_norm_path, semantic_normalized.astype(np.float32))
            np.save(full_norm_path, full_normalized.astype(np.float32))
            np.save(union_path, final_binary)

        # Save visualizations, including cached results
        save_heatmap_vis(semantic_heatmap, os.path.join(inter_subdir, f"{base_name}_semantic_heatmap.png"), cmap='jet')
        save_heatmap_vis(full_heatmap, os.path.join(inter_subdir, f"{base_name}_full_heatmap.png"), cmap='jet')
        save_heatmap_vis(semantic_normalized, os.path.join(inter_subdir, f"{base_name}_semantic_normalized.png"), cmap='jet', vmin=0, vmax=1)
        save_heatmap_vis(full_normalized, os.path.join(inter_subdir, f"{base_name}_full_normalized.png"), cmap='jet', vmin=0, vmax=1)
        save_heatmap_vis(semantic_binary, os.path.join(inter_subdir, f"{base_name}_semantic_binary.png"), cmap='gray', vmin=0, vmax=1)
        save_heatmap_vis(full_binary, os.path.join(inter_subdir, f"{base_name}_full_binary.png"), cmap='gray', vmin=0, vmax=1)
        save_heatmap_vis(final_binary, os.path.join(inter_subdir, f"{base_name}_union_binary.png"), cmap='gray', vmin=0, vmax=1)
        save_heatmap_vis(toxicity_map, os.path.join(inter_subdir, f"{base_name}_toxicity.png"), cmap='jet')

        with open(OUTPUT_CSV, 'a') as f:
            f.write(f"{category},{fname},{num_valid_semantic},{np.sum(full_binary > 0)},{original_score:.6f}\n")
        
        gt_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        for x1, y1, x2, y2 in gt_bboxes:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))
            if x2 > x1 and y2 > y1:
                gt_mask[y1:y2, x1:x2] = 1
        
        inter = np.logical_and(final_binary, gt_mask).sum()
        union = np.logical_or(final_binary, gt_mask).sum()
        iou = inter / union if union > 0 else 0.0
        per_image_ious.append(iou)
        
        if category not in category_stats:
            category_stats[category] = {'ious': [], 'detected': 0, 'total_gt': 0, 'precise': 0, 'has_pred': 0, 'inter': 0, 'union': 0}
        category_stats[category]['ious'].append(iou)
        category_stats[category]['inter'] += inter
        category_stats[category]['union'] += union
        
        for x1, y1, x2, y2 in gt_bboxes:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
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
        
        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    # Calculate metrics
    gIoU = np.mean(per_image_ious) if per_image_ious else 0
    total_inter = sum(s['inter'] for s in category_stats.values())
    total_union = sum(s['union'] for s in category_stats.values())
    ciou = total_inter / total_union if total_union > 0 else 0
    recall = total_detected / total_gt_objs if total_gt_objs > 0 else 0
    precision = total_precise / total_has_pred if total_has_pred > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print('='*70)
    print('Own Dataset - Final Metrics')
    print('='*70)
    print(f'Total Images: {len(per_image_ious)}')
    print(f'gIoU: {gIoU:.6f}')
    print(f'cIoU: {ciou:.6f}')
    print(f'Recall: {recall:.6f} ({recall*100:.2f}%)')
    print(f'Precision: {precision:.6f} ({precision*100:.2f}%)')
    print(f'F1: {f1:.6f}')
    print()
    print('Per-Category Metrics:')
    print('-'*70)
    print('%-12s %-8s %-8s %-8s %-10s %-12s %-8s' % ('Category', 'Images', 'gIoU', 'cIoU', 'Recall', 'Precision', 'F1'))
    print('-'*70)
    for cat in sorted(category_stats.keys()):
        s = category_stats[cat]
        cat_giou = np.mean(s['ious']) if s['ious'] else 0
        cat_ciou = s['inter'] / s['union'] if s['union'] > 0 else 0
        cat_recall = s['detected'] / s['total_gt'] if s['total_gt'] > 0 else 0
        cat_precision = s['precise'] / s['has_pred'] if s['has_pred'] > 0 else 0
        cat_f1 = 2 * (cat_precision * cat_recall) / (cat_precision + cat_recall) if (cat_precision + cat_recall) > 0 else 0
        print('%-12s %-8d %-8.4f %-8.4f %-10.2f%% %-12.2f%% %-8.4f' % (cat, len(s['ious']), cat_giou, cat_ciou, cat_recall*100, cat_precision*100, cat_f1))
    print('='*70)
    
    # Summary file
    summary_lines = [
        "=" * 70,
        "Own Dataset - Results",
        f"Threshold={THRESHOLD}",
        "=" * 70,
        f"Total Images: {len(per_image_ious)}",
        f"gIoU: {gIoU:.6f}",
        f"cIoU: {ciou:.6f}",
        f"Recall@{THRESHOLD}: {recall:.6f} ({total_detected}/{total_gt_objs})",
        f"Precision@{THRESHOLD}: {precision:.6f} ({total_precise}/{total_has_pred})",
        f"F1: {f1:.6f}",
        "",
        "=" * 50,
        "Per-Category:",
        "=" * 50,
    ]
    
    for cat in sorted(category_stats.keys()):
        s = category_stats[cat]
        cat_giou = np.mean(s['ious']) if s['ious'] else 0
        cat_ciou = s['inter'] / s['union'] if s['union'] > 0 else 0
        cat_recall = s['detected'] / s['total_gt'] if s['total_gt'] > 0 else 0
        cat_precision = s['precise'] / s['has_pred'] if s['has_pred'] > 0 else 0
        cat_f1 = 2 * (cat_precision * cat_recall) / (cat_precision + cat_recall) if (cat_precision + cat_recall) > 0 else 0
        summary_lines.append(f"\n[{cat}]")
        summary_lines.append(f"  Images: {len(s['ious'])}")
        summary_lines.append(f"  gIoU: {cat_giou:.4f}")
        summary_lines.append(f"  cIoU: {cat_ciou:.4f}")
        summary_lines.append(f"  Recall: {cat_recall:.4f}")
        summary_lines.append(f"  Precision: {cat_precision:.4f}")
        summary_lines.append(f"  F1: {cat_f1:.4f}")
    
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    
    with open(SUMMARY_OUT, 'w') as f:
        f.write(summary_text)
    
    print(f"\nDone! Summary: {SUMMARY_OUT}")

if __name__ == "__main__":
    main()
