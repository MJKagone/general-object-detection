import matplotlib
matplotlib.use('Agg')  # Prevents PySide6 debugpy event loop crash

import argparse
import json
import logging
import os
import random
from types import SimpleNamespace
import cv2 as cv
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import YOLOWorld
from EgoObjects.egoobjects_api.egoobjects import EgoObjects
from EgoObjects.egoobjects_api.eval import EgoObjectsEval
from EgoObjects.egoobjects_api.results import EgoObjectsResults


GT_PATH = "EgoObjects/data/EgoObjectsV1_unified_eval.json"
METADATA_PATH = "EgoObjects/data/EgoObjectsV1_unified_metadata.json"
IMAGE_DIR = "EgoObjects/data/images"
DEFAULT_PRED_DIR = "results/predictions"

METRIC_FILTER = {
    "cat_det": [
        {"iou": "coco", "gr": "all", "ar": "all", "bg": "all", "lt": "all", "df": "all"},
        {"iou": "50", "gr": "all", "ar": "all", "bg": "all", "lt": "all", "df": "all"},
        {"iou": "75", "gr": "all", "ar": "all", "bg": "all", "lt": "all", "df": "all"},
    ]
}


def get_egoobjects_meta(metadata_path):
    with open(metadata_path, "r") as fp:
        metadata = json.load(fp)

    cat_det_cat_id_2_name = {cat["id"]: cat["name"] for cat in metadata["cat_det_cats"]}
    cat_det_cat_ids = sorted(cat_det_cat_id_2_name.keys())
    metadata["cat_det_cat_id_2_cont_id"] = {
        cat_id: i for i, cat_id in enumerate(cat_det_cat_ids)
    }
    metadata["cat_det_cat_names"] = [cat_det_cat_id_2_name[cat_id] for cat_id in cat_det_cat_ids]
    return SimpleNamespace(**metadata)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EgoObjects predictions with YoloWorld or Grounding DINO")
    parser.add_argument(
        "--model",
        choices=["yoloworld", "gdino"],
        default="gdino",
        help="Model backend used for inference when no cached predictions are found.",
    )
    parser.add_argument(
        "--pred-path",
        default=None,
        help="Optional path to prediction JSON. If omitted, uses results/predictions/EgoObjects_predictions_<model>.json",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Number of CPU worker processes for EgoObjects evaluation. Use 1 for single-process mode.",
    )
    return parser.parse_args()

def save_sanity_check_images(ego_gt, predictions, image_dir, out_dir="results/sanity_checks", num_images=20):
    """Draws GT and Predictions on separate image copies and saves them to disk."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Group predictions by image_id for faster lookup
    preds_by_img = {}
    for p in predictions:
        preds_by_img.setdefault(p["image_id"], []).append(p)
        
    # Sample random images that contain at least one prediction
    valid_img_ids = list(preds_by_img.keys())
    if not valid_img_ids:
        logging.warning("No predictions available to visualize.")
        return
        
    sample_img_ids = random.sample(valid_img_ids, min(num_images, len(valid_img_ids)))
    
    logging.info(f"Generating {len(sample_img_ids) * 2} sanity check images in {out_dir}...")
    
    for img_id in sample_img_ids:
        img_info = ego_gt.imgs[img_id]
        file_name = (
            img_info.get("file_name") or img_info.get("image_path") or 
            img_info.get("path") or img_info.get("name")
        )
        if not file_name and "url" in img_info:
            file_name = img_info["url"].split("/")[-1]
            
        base_name, ext = os.path.splitext(os.path.basename(file_name))
        file_path = os.path.join(image_dir, os.path.basename(file_name))
        if not os.path.exists(file_path):
            file_path = os.path.join(image_dir, file_name)
            
        img = cv.imread(file_path)
        if img is None:
            logging.warning(f"Could not read image for visualization: {file_path}")
            continue
            
        # Create separate image copies
        img_gt = img.copy()
        img_pred = img.copy()
            
        # Draw Ground Truth (Green)
        gt_anns = ego_gt.load_anns("cat_det", ego_gt.get_ann_ids("cat_det", img_ids=[img_id]))
        for ann in gt_anns:
            x, y, w, h = map(int, ann["bbox"])
            cat_name = ego_gt.cats["cat_det"][ann["category_id"]]["name"]
            cv.rectangle(img_gt, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(img_gt, f"GT: {cat_name}", (x, max(15, y-5)), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        # Draw Predictions (Red)
        for pred in preds_by_img[img_id]:
            if pred["score"] < 0.15: 
                continue
                
            x, y, w, h = map(int, pred["bbox"])
            
            cat_dict = ego_gt.cats["cat_det"].get(pred["category_id"])
            cat_name = cat_dict["name"] if cat_dict else f"ID:{pred['category_id']}"
            
            cv.rectangle(img_pred, (x, y), (x+w, y+h), (0, 0, 255), 2)
            text = f"Pred: {cat_name} ({pred['score']:.2f})"
            cv.putText(img_pred, text, (x, max(15, y-5)), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        # Save separate images
        gt_out_path = os.path.join(out_dir, f"{base_name}_gt{ext}")
        pred_out_path = os.path.join(out_dir, f"{base_name}_pred{ext}")
        
        cv.imwrite(gt_out_path, img_gt)
        cv.imwrite(pred_out_path, img_pred)

def build_prompt_for_image(img_id):
    names = sorted(image_category_names.get(img_id, set()), key=lambda s: s.lower())
    if not names:
        return "object."
    return ". ".join(names) + "."

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

args = parse_args()
MODEL_BACKEND = args.model
NUM_PROCESSES = max(1, args.num_processes)
PRED_PATH = args.pred_path or os.path.join(
    DEFAULT_PRED_DIR,
    f"EgoObjects_predictions_{MODEL_BACKEND}.json",
)

# 1. Load raw JSON
with open(GT_PATH, 'r') as f:
    gt_data = json.load(f)

ego_metadata = get_egoobjects_meta(METADATA_PATH)

ego_gt = EgoObjects(GT_PATH, ego_metadata)

# 2. Build category mapping for open-vocabulary prompts
categories = sorted(ego_metadata.cat_det_cats, key=lambda cat: cat["id"])

text_prompts = [cat["name"] for cat in categories]
idx_to_cat_id = {i: cat["id"] for i, cat in enumerate(categories)}
category_name_to_id = {cat["name"].lower(): cat["id"] for cat in categories}
image_category_names = {}
for ann in gt_data.get("annotations", []):
    img_id = ann.get("image_id")
    if img_id is None:
        continue
    name = ann.get("category_freeform")
    if name:
        image_category_names.setdefault(img_id, set()).add(name)

# 3. Load existing prediction checkpoint or run inference loop
if os.path.exists(PRED_PATH):
    print(f"Loading saved predictions from {PRED_PATH}...")
    with open(PRED_PATH, "r") as f:
        results = json.load(f)
elif MODEL_BACKEND == "yoloworld":
    print("No cached predictions found. Starting model inference...")
    model = YOLOWorld("models/yolov8x-worldv2.pt")
    model.set_classes(text_prompts)

    results = []

    for img_id in ego_gt.get_img_ids():
        img_info = ego_gt.imgs[img_id]

        file_name = (
            img_info.get("file_name")
            or img_info.get("image_path")
            or img_info.get("path")
            or img_info.get("name")
        )

        if not file_name and "url" in img_info:
            file_name = img_info["url"].split("/")[-1]

        if not file_name:
            raise KeyError(f"Could not find valid file path in img_info keys: {list(img_info.keys())}")

        base_name = os.path.basename(file_name)
        file_path = os.path.join(IMAGE_DIR, base_name)

        if not os.path.exists(file_path):
            file_path = os.path.join(IMAGE_DIR, file_name)

        preds = model.predict(file_path, conf=0.1, verbose=False)[0]

        boxes = preds.boxes.xyxy.cpu().numpy()
        scores = preds.boxes.conf.cpu().numpy()
        cls_indices = preds.boxes.cls.cpu().numpy().astype(int)

        for box, score, cls_idx in zip(boxes, scores, cls_indices):
            x1, y1, x2, y2 = box
            coco_box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

            results.append({
                "image_id": img_id,
                "category_id": idx_to_cat_id[cls_idx],
                "bbox": [round(c, 2) for c in coco_box],
                "score": float(score)
            })

    # Save point after inference completes
    print(f"Saving predictions checkpoint to {PRED_PATH}...")
    with open(PRED_PATH, "w") as f:
        json.dump(results, f, indent=2)

elif MODEL_BACKEND == "gdino":
    print("No cached predictions found. Starting Grounding DINO inference...")
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    results = []

    for img_id in ego_gt.get_img_ids():
        img_info = ego_gt.imgs[img_id]

        file_name = (
            img_info.get("file_name")
            or img_info.get("image_path")
            or img_info.get("path")
            or img_info.get("name")
        )

        if not file_name and "url" in img_info:
            file_name = img_info["url"].split("/")[-1]

        if not file_name:
            raise KeyError(f"Could not find valid file path in img_info keys: {list(img_info.keys())}")

        base_name = os.path.basename(file_name)
        file_path = os.path.join(IMAGE_DIR, base_name)

        if not os.path.exists(file_path):
            file_path = os.path.join(IMAGE_DIR, file_name)

        image = cv.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Image could not be read: {file_path}")

        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        prompt = build_prompt_for_image(img_id)

        inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        detections = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.20,
            text_threshold=0.20,
            target_sizes=[pil_image.size[::-1]],
        )[0]

        boxes = detections["boxes"].cpu().numpy()
        scores = detections["scores"].cpu().numpy()
        labels = detections["labels"]

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(float, box)
            coco_box = [float(x1), float(y1), float(max(x2 - x1, 0.0)), float(max(y2 - y1, 0.0))]
            label_name = str(label).strip().lower()
            category_id = category_name_to_id.get(label_name)

            if category_id is None:
                for cat in categories:
                    cat_name = cat["name"].lower()
                    if label_name == cat_name or label_name in cat_name or cat_name in label_name:
                        category_id = cat["id"]
                        break

            if category_id is None:
                continue

            results.append({
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [round(c, 2) for c in coco_box],
                "score": float(score)
            })

    print(f"Saving predictions checkpoint to {PRED_PATH}...")
    with open(PRED_PATH, "w") as f:
        json.dump(results, f, indent=2)
else:
    raise ValueError(f"Unsupported model backend: {MODEL_BACKEND}")

# 4. Save sanity check images
if results:
    save_sanity_check_images(ego_gt, results, IMAGE_DIR, out_dir="results/sanity_checks", num_images=20)

# 5. Evaluate with EgoObjects API
if results:
    print(f"Evaluating {len(results)} detections...")
    print(f"Using {NUM_PROCESSES} CPU processes for evaluation.")
    ego_dt = EgoObjectsResults(
        ego_gt,
        cat_det_dt_anns=results,
        inst_det_dt_anns=[],
        max_dets=100,
    )
    evaluator = EgoObjectsEval(
        ego_gt,
        ego_dt,
        num_processes=NUM_PROCESSES,
        max_dets=100,
        eval_type=("cat_det",),
    )
    evaluator.run(METRIC_FILTER)
    evaluator.print_results()
else:
    print("No detections generated.")