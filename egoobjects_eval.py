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
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
import torchvision
import re


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

class Detection(BaseModel):
    box_2d: List[int] = Field(description="[ymin, xmin, ymax, xmax] normalized to 0-1000")
    label: str = Field(description="The detected object category label")
    score: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)

class DetectionList(BaseModel):
    detections: List[Detection]

def match_category(predicted_label: str, category_map: dict, categories_list: list):
    clean = predicted_label.strip().lower().rstrip(".")
    if clean in category_map:
        return category_map[clean]
    
    # Check simple plural
    if clean.endswith("s") and clean[:-1] in category_map:
        return category_map[clean[:-1]]
        
    # Match whole word boundaries only
    for cat in categories_list:
        target = cat["name"].lower()
        if re.search(rf"\b{re.escape(clean)}\b", target) or re.search(rf"\b{re.escape(target)}\b", clean):
            return cat["id"]
            
    return None

def prompt(client, config, img, categories):
    prompt = f"Detect all instances of objects in the image. The list of labels is: {categories}. Only use the precise labels from the list. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000. Include the label and confidence score in the JSON response."
    response = client.models.generate_content(model="gemini-3.7-flash",
                                            contents=[img, prompt],
                                            config=config
                                            )
    return response

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
        choices=["yoloworld", "gdino", "gemini"],
        default="yoloworld",
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
    parser.add_argument(
        "--percentage",
        type=float,
        default=100.0,
        help="Percentage of dataset images to evaluate, from 0 to 100.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling a dataset subset.",
    )
    return parser.parse_args()

def save_sanity_check_images(ego_gt, predictions, image_dir, out_dir="results/sanity_checks", num_images=20, seed=42):
    """Draws GT and Predictions on separate image copies and saves them to disk."""
    os.makedirs(out_dir, exist_ok=True)
    
    preds_by_img = {}
    for p in predictions:
        preds_by_img.setdefault(p["image_id"], []).append(p)
        
    # 1. Use the full, sorted dataset image list so all models sample the exact same image set
    all_img_ids = sorted(ego_gt.get_img_ids())
    if not all_img_ids:
        logging.warning("No images available to visualize.")
        return
        
    # 2. Use a seeded instance of Random
    rng = random.Random(seed)
    sample_img_ids = rng.sample(all_img_ids, min(num_images, len(all_img_ids)))
    
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
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
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
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        # Save separate images
        gt_out_path = os.path.join(out_dir, f"{base_name}_gt{ext}")
        pred_out_path = os.path.join(out_dir, f"{base_name}_pred{ext}")
        
        cv.imwrite(gt_out_path, img_gt)
        cv.imwrite(pred_out_path, img_pred)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

args = parse_args()
MODEL_BACKEND = args.model
NUM_PROCESSES = max(1, args.num_processes)
if not 0 < args.percentage <= 100:
    raise ValueError("--percentage must be greater than 0 and at most 100")
PRED_PATH = args.pred_path or os.path.join(
    DEFAULT_PRED_DIR,
    (
        f"EgoObjects_predictions_{MODEL_BACKEND}.json"
        if args.percentage == 100.0
        else f"EgoObjects_predictions_{MODEL_BACKEND}_{args.percentage:g}pct_seed{args.seed}.json"
    ),
)

# 1. Load EgoObjects metadata and ground truth
ego_metadata = get_egoobjects_meta(METADATA_PATH)
ego_gt = EgoObjects(GT_PATH, ego_metadata)

if args.percentage < 100.0:
    rng = random.Random(args.seed)
    all_img_ids = ego_gt.get_img_ids()
    sample_size = max(1, int(len(all_img_ids) * args.percentage / 100.0))
    subset_image_ids = set(rng.sample(all_img_ids, sample_size))
    # Restrict active image IDs in ego_gt to the sampled subset
    ego_gt.imgs = {img_id: img for img_id, img in ego_gt.imgs.items() if img_id in subset_image_ids}
else:
    subset_image_ids = set(ego_gt.get_img_ids())

# 2. Build category mapping for open-vocabulary prompts
categories = sorted(ego_metadata.cat_det_cats, key=lambda cat: cat["id"])

text_prompts = [cat["name"] for cat in categories]
idx_to_cat_id = {i: cat["id"] for i, cat in enumerate(categories)}
category_name_to_id = {cat["name"].lower(): cat["id"] for cat in categories}

# 3. Load existing prediction checkpoint or run inference loop
if os.path.exists(PRED_PATH):
    print(f"Loading saved predictions from {PRED_PATH}...")
    with open(PRED_PATH, "r") as f:
        results = json.load(f)
    results = [p for p in results if p["image_id"] in subset_image_ids]

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

        preds = model.predict(file_path, conf=0.01, verbose=False)[0]

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
    print("No cached predictions found. Starting model inference with Grounding DINO...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Load model directly in fp16 on CUDA
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    # Split categories into token-safe chunks (<200 tokens)
    chunks = []
    current_chunk = []
    for cat in categories:
        test_chunk = current_chunk + [cat]
        test_prompt = ". ".join(c["name"].lower() for c in test_chunk) + "."
        token_len = len(processor.tokenizer(test_prompt)["input_ids"])
        if token_len > 200 and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [cat]
        else:
            current_chunk = test_chunk
    if current_chunk:
        chunks.append(current_chunk)

    # Precompute prompt strings and category lookup maps
    chunk_prompts = [". ".join(cat["name"].lower() for cat in chunk) + "." for chunk in chunks]
    chunk_maps = [{cat["name"].lower(): cat["id"] for cat in chunk} for chunk in chunks]

    print(f"Split {len(categories)} categories into {len(chunks)} sub-prompts. Executing batched inference...")

    results = []
    CHUNK_BATCH_SIZE = 4  # Adjust batch size based on available VRAM

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

        pil_img = Image.open(file_path).convert("RGB")
        w, h = pil_img.size

        # Accumulators for all chunks on this image
        img_boxes = []
        img_scores = []
        img_cat_ids = []

        # Run inference over text chunks in mini-batches
        for i in range(0, len(chunks), CHUNK_BATCH_SIZE):
            batch_chunks = chunks[i : i + CHUNK_BATCH_SIZE]
            batch_prompts = chunk_prompts[i : i + CHUNK_BATCH_SIZE]
            batch_maps = chunk_maps[i : i + CHUNK_BATCH_SIZE]
            b_size = len(batch_prompts)

            inputs = processor(
                images=[pil_img] * b_size,
                text=batch_prompts,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.inference_mode():
                if device == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)

            try:
                post_res_list = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.1,
                    text_threshold=0.1,
                    target_sizes=[(h, w)] * b_size,
                )
            except TypeError:
                post_res_list = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=0.1,
                    text_threshold=0.1,
                    target_sizes=[(h, w)] * b_size,
                )

            for post_res, chunk_map in zip(post_res_list, batch_maps):
                boxes = post_res["boxes"].cpu().numpy()
                scores = post_res["scores"].cpu().numpy()
                labels = post_res.get("text_labels", post_res.get("labels", []))

                for box, score, label in zip(boxes, scores, labels):
                    clean_label = str(label).strip().rstrip(".").lower()
                    cat_id = match_category(clean_label, chunk_map, categories)

                    if cat_id is not None:
                        img_boxes.append(box.tolist())  # [x1, y1, x2, y2]
                        img_scores.append(float(score))
                        img_cat_ids.append(cat_id)

        # Run Batched NMS across all collected chunk detections for this image
        if img_boxes:
            boxes_t = torch.tensor(img_boxes, dtype=torch.float32)
            scores_t = torch.tensor(img_scores, dtype=torch.float32)
            labels_t = torch.tensor(img_cat_ids, dtype=torch.int64)

            keep = torchvision.ops.batched_nms(boxes_t, scores_t, labels_t, iou_threshold=0.5)

            for idx in keep.tolist()[:100]:
                x1, y1, x2, y2 = boxes_t[idx].tolist()
                coco_box = [x1, y1, x2 - x1, y2 - y1]
                results.append({
                    "image_id": img_id,
                    "category_id": int(labels_t[idx]),
                    "bbox": [round(c, 2) for c in coco_box],
                    "score": float(scores_t[idx]),
                })

    # Save point after inference completes
    print(f"Saving predictions checkpoint to {PRED_PATH}...")
    with open(PRED_PATH, "w") as f:
        json.dump(results, f, indent=2)

elif MODEL_BACKEND == "gemini":
    print("No cached predictions found. Starting model inference with Gemini API...")
    client = genai.Client()
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=DetectionList,
        temperature=0.1,
    )
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

        # Ensure image is RGB before passing to the API
        img = Image.open(file_path).convert("RGB")
        width, height = img.size

        response = prompt(client, config, img, text_prompts)

        try:
            parsed = json.loads(response.text)
            pred_boxes = parsed.get("detections", parsed) if isinstance(parsed, dict) else parsed
        except (json.JSONDecodeError, AttributeError):
            print(f"Failed to parse JSON response for image {file_path}. Response text: {response.text}")
            continue

        for pred in pred_boxes:
            if not isinstance(pred, dict) or "box_2d" not in pred or "label" not in pred:
                continue

            ymin, xmin, ymax, xmax = pred["box_2d"]
            abs_y1 = max(0.0, ymin / 1000.0 * height)
            abs_x1 = max(0.0, xmin / 1000.0 * width)
            abs_y2 = min(float(height), ymax / 1000.0 * height)
            abs_x2 = min(float(width), xmax / 1000.0 * width)

            w_box = abs_x2 - abs_x1
            h_box = abs_y2 - abs_y1

            if w_box <= 0 or h_box <= 0:
                continue

            coco_box = [abs_x1, abs_y1, w_box, h_box]
            cat_id = match_category(pred["label"], category_name_to_id, categories)

            if cat_id is None:
                continue

            score = float(pred.get("score", 0.9))

            results.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [round(c, 2) for c in coco_box],
                "score": score,
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