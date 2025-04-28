import argparse
import datetime
import json
import random
import time
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from collections import defaultdict

import datasets
import util.misc as utils
from transformers import DetrImageProcessor, DetrForObjectDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from models import build_model

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def draw_boxes(ax, boxes, labels, scores=None, color='red', title=''):
    for i, box in enumerate(boxes):
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        label = category_id_to_name.get(labels[i], str(labels[i]))
        score_str = f'{scores[i]:.2f}' if scores is not None else ''
        ax.text(x, y, f'{label} {score_str}', color=color, fontsize=8, verticalalignment='top')
    ax.set_title(title)

def visualize_prediction_and_gt(img_path, image_id, finetuned, baseline, weather, id_name):
    image = Image.open(img_path).convert("RGB")
    ann_ids = coco_gt.getAnnIds(imgIds=image_id)
    anns = coco_gt.loadAnns(ann_ids)
    gt_boxes = [ann["bbox"] for ann in anns]
    gt_labels = [ann["category_id"] for ann in anns]

    f_idx = finetuned["scores"].topk(len(gt_boxes)).indices
    pred_boxes = finetuned["boxes"][f_idx].cpu().numpy()
    pred_scores = finetuned["scores"][f_idx].cpu().numpy()
    pred_labels = finetuned["labels"][f_idx].cpu().numpy()
    pred_boxes_coco = [[x0, y0, x1-x0, y1-y0] for x0, y0, x1, y1 in pred_boxes]

    b_idx = baseline["scores"].topk(len(gt_boxes)).indices
    pred_boxes_b = baseline["boxes"][b_idx].cpu().numpy()
    pred_scores_b = baseline["scores"][b_idx].cpu().numpy()
    pred_labels_b = baseline["labels"][b_idx].cpu().numpy()
    pred_boxes_coco_b = [[x0, y0, x1-x0, y1-y0] for x0, y0, x1, y1 in pred_boxes_b]

    fig, ax = plt.subplots(1, 2, figsize=(16, 16))
    ax[1].imshow(image)
    draw_boxes(ax[1], gt_boxes, gt_labels, color='green', title='Ground Truth')
    draw_boxes(ax[1], pred_boxes_coco, pred_labels, scores=pred_scores, color='red', title='Finetuned')

    ax[0].imshow(image)
    draw_boxes(ax[0], gt_boxes, gt_labels, color='green', title='Ground Truth')
    draw_boxes(ax[0], pred_boxes_coco_b, pred_labels_b, scores=pred_scores_b, color='red', title='Baseline')
    for ax_ in ax.flat:
        ax_.axis('off')
    plt.savefig('results/{}_{}.png'.format(weather, id_name))


if __name__ == '__main__':
    IMAGES_DIR = 'DAWN-Dataset.v2i.coco/test'
    ANNOTATION_PATH = 'DAWN-Dataset.v2i.coco/annotations/instances_test.json'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    model, criterion, postprocessors = build_model(args)
    baseline = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    baseline = baseline.to(device)
    baseline.eval()
    checkpoint = torch.load('ckpts/checkpoint0999.pth')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    dataset_test = datasets.build_dataset(image_set='val', args=args)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    dataloader_val = torch.utils.data.DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    with open(ANNOTATION_PATH, 'r') as f:
        coco_data = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
    image_id_to_weather = {img["id"]: img["file_name"].split('-')[0] for img in coco_data["images"]}
    weather_types = set(image_id_to_weather.values())

    weather_to_image_paths = {weather: [] for weather in weather_types}
    for img_id, fname in id_to_filename.items():
        weather = image_id_to_weather[img_id]
        weather_to_image_paths[weather].append(os.path.join(IMAGES_DIR, fname))

    coco_gt = COCO(ANNOTATION_PATH)
    results = defaultdict(list)
    vis_fines = defaultdict(list)
    vis_bases = defaultdict(list)
    for samples, targets in dataloader_val:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            outputs = model(samples)
            base_outputs = baseline(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results_per_image = postprocessors['bbox'](outputs, orig_target_sizes)
        results_dict = results_per_image[0]     # BATCH_SIZE=1
        bases_per_image = postprocessors['bbox'](base_outputs, orig_target_sizes)
        bases_dict = bases_per_image[0]

        image_id = int(targets[0]['image_id'])
        weather = image_id_to_weather[image_id]

        for score, label, box in zip(results_dict["scores"], results_dict["labels"], results_dict["boxes"]):
            box = box.tolist()
            results[weather].append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [box[0], box[1], box[2]-box[0], box[3]-box[1]],
                "score": score.item()
            })
        results_dict["image_id"] = image_id
        vis_fines[weather].append(results_dict)
        vis_bases[weather].append(bases_dict)
            

    category_id_to_name = {cat["id"]: cat["name"] for cat in coco_gt.dataset["categories"]}

    # METRICS
    # for weather, result in results.items():
        # print(f"\nWeather Samples: {weather}, {len(weather_to_image_paths[weather])}")
    #     result_file = f'detr_results_{weather}.json'
    #     with open(result_file, 'w') as f:
    #         json.dump(result, f, indent=2)

    #     coco_dt = coco_gt.loadRes(result_file)
    #     coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    #     coco_eval.evaluate()
    #     coco_eval.accumulate()
    #     coco_eval.summarize()

    # VISUALIZATION
    for weather, result in vis_fines.items(): 
        print(f"\nWeather Samples: {weather}, {len(result)}")
        for ri, res in enumerate(result):
            sample_image_id = res["image_id"]
            sample_img_path = os.path.join(IMAGES_DIR, id_to_filename[sample_image_id])
            visualize_prediction_and_gt(sample_img_path, sample_image_id, res, vis_bases[weather][ri], weather, ri)
