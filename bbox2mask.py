"""
Usage:

python bbox2mask.py -i path/to/image.jpg -b path/to/bbox.json
"""

import argparse
import os
import json

import matplotlib.pyplot as plt
import numpy as np

import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

import torch

# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")

model = build_sam3_image_model(
    bpe_path=bpe_path,
    device="cuda",
    checkpoint_path="models/sam3.1_multiplex.pt",
    load_from_HF=False
)

def convert_bbox(bbox):
    """
    Convert a bounding box from [xmin, ymin, xmax, ymax] format to [center_x, center_y, w, h] format.
    """
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    return [cx, cy, w, h]

def main(image_path, bbox):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    converted_bbox = convert_bbox(bbox)
    normalized_bbox = normalize_bbox(converted_bbox, width, height)

    processor = Sam3Processor(model)
    
    # 1. Process the image to extract features and initialize the state
    inference_state = processor.set_image(img)
    
    # 2. Add the geometric prompt to the newly generated state
    inference_state = processor.add_geometric_prompt(state=inference_state, box=normalized_bbox, label=True)
    
    plot_results(img, inference_state)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script for SAM3")
    parser.add_argument("-i", "--image", type=str, help="Path to the local image file to be processed")
    parser.add_argument("-b", "--bbox", type=str, help="Path to the bounding box from Gemini API in JSON format")
    args = parser.parse_args()

    with open(args.bbox, "r") as f:
        bbox = json.load(f)[0]

    main(args.image, bbox)