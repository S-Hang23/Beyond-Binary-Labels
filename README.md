# Beyond Binary Labels

Code and data for **Beyond Binary Labels: Explainable Malicious Image Localization via MLLM-Driven Attribution**. The inference script combines Qwen3-VL-8B-Instruct with SAM ViT-H. It computes heatmaps for a semantic, box-prompted segmentation branch and a full-image automatic segmentation branch, combines thresholded branch masks, and evaluates the result against bounding-box annotations.

## Repository layout

| Path | Contents |
| --- | --- |
| `owndata_test.py` | Inference and evaluation on the curated image dataset. |
| `data/own/annotations.json` | COCO-style image records and ground-truth bounding boxes. |
| `data/own/images/{illegal,porn,violence,others}/` | Curated dataset images. |
| `data/own/t2i_prompts.json` | Original category-wise text-to-image prompt pool for the FLUX-generated images. |

The script reads the dataset relative to this repository. Model weights are **not** included.

## Setup

Use a CUDA-capable environment with PyTorch, `transformers` (including `Qwen3VLForConditionalGeneration`), `qwen-vl-utils`, `segment-anything`, OpenCV, Pillow, NumPy, tqdm, and Matplotlib. Download the Qwen3-VL-8B-Instruct model and the SAM ViT-H checkpoint separately. Set their paths before running the script:

```bash
export QWEN_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
export SAM_CHECKPOINT=/path/to/sam_vit_h.pth
```

The script uses `cuda:0` within the visible devices. Set `CUDA_VISIBLE_DEVICES` to select a GPU. If it is unset, `owndata_test.py` selects GPU 4 according to its local default.

## Run inference and evaluation

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python owndata_test.py
```

The script uses images listed in `data/own/annotations.json` and found under `data/own/images/`.

Outputs are written under `outputs/own/`. A run saves heatmap arrays in `arrays/`, branch-level intermediate arrays and heatmap images in `intermediate/`, a `metrics.csv`, a `summary.txt`, and an `errors.txt` when errors occur.

In the summary, `gIoU` is the mean per-image IoU and `cIoU` is computed from pooled intersection and union pixels. The printed `Recall` corresponds to object coverage (`ObjCovR@0.5`): a ground-truth box is counted when the prediction covers at least half of it. The printed `Precision` corresponds to image overlap (`ImgOvP@0.5`): an image with a nonempty prediction is counted when at least half of its predicted area falls inside ground-truth boxes. These have different counting units, so the harmonic-mean `F1` is a coarse summary rather than conventional pixel-level F1.

## Text-to-image prompts

[`data/own/t2i_prompts.json`](data/own/t2i_prompts.json) is an unchanged copy of the original FLUX prompt list. It contains 390 prompts grouped under eight keys: `gun`, `gamble`, `drug`, `robbery`, `violence`, `discrimination`, `extremism`, and `thievery`. These keys correspond to the prefixes of generated image filenames such as `drug_t2i_21.jpg`. The JSON documents the prompt pool. It does not contain a prompt-to-image index or the random seed used for each image, and the inference script does not read it.

The original generation script used FLUX.1-schnell at 512 × 512 pixels, 25 inference steps, guidance scale 3.5, and randomly selected seeds. Because those per-image seeds were not recorded in the prompt JSON, the file supports inspection of the prompt set but does not by itself reproduce individual generated images exactly.
