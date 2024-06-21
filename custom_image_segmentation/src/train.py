# Import modules 

import argparse, os, json
from pathlib import Path
import torch, detectron2


TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, yaml

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper, build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.data import transforms as T

def parse_configuration_file(config_file: str) -> tuple:

    with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            info = config['info']
            hyperparameters = config['hyperparameters']

    return (info, hyperparameters)

def construct_dataset_dict(dataset_dir: str, class_dict: dict) -> dict:

    annotations_file = os.path.join(dataset_dir, "annotations.json")
    assert os.path.exists(annotations_file), f"'annotations.json' is missing in the folder: {dataset_dir}"

    with open(annotations_file) as f:
        all_annotations = json.load(f)

    dataset_dicts = []
    for idx, image in enumerate(all_annotations.values()):
        record = {}

        record["file_name"] = os.path.join(dataset_dir, image["filename"])
        record["image_id"] = idx

        annotations = image["regions"]

        objects = []

        for _, annotation in annotations.items():

            regions = annotation["region_attributes"]
            shape_attr = annotation["shape_attributes"]
            px = shape_attr["all_points_x"]
            py = shape_attr["all_points_y"]

            polygon_coords = [(x, y) for x, y in zip(px, py)]
            polygon = [p for x in polygon_coords for p in x]

            category_id = class_dict[regions["label"]]

            object = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [polygon],
                "category_id": category_id,
            }

            objects.append(object)
        
        record["annotations"] = objects
        dataset_dicts.append(record)

    return dataset_dicts

class AffineAugsTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        augs = T.AugmentationList([
            T.RandomRotation(angle=[0, 180], expand=False),
            T.RandomBrightness(0.5, 2),
            T.RandomContrast(0.5, 2),
            T.RandomSaturation(0.5, 2),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        ])
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augs.augs)
        return build_detection_train_loader(cfg, mapper=mapper)

def main(args):
    print("Configuration File:", args.config)
    print("Output Directory:", args.output_dir)
    print("Device: ", args.device)
    print("Data Directory: ", args.data_dir)

    info, hyperparameters = parse_configuration_file(args.config)

    DatasetCatalog.clear()
    MetadataCatalog.clear()

    for d in ["train", "val"]:
        DatasetCatalog.register(info["name"] + " " + d, lambda d=d: construct_dataset_dict(Path(args.data_dir) / d, info["classes"]))
        MetadataCatalog.get(info["name"] + " " + d).set(thing_classes=list(info["classes"].keys()))
        
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (info["name"] + " " + "train")
    cfg.DATASETS.TEST = ()
    cfg.MODEL.DEVICE = args.device 
    cfg.DATALOADER.NUM_WORKERS = 0 if args.device == "cpu" else args.num_workers
    cfg.SOLVER.IMS_PER_BATCH = hyperparameters['batch_num']
    cfg.SOLVER.BASE_LR = hyperparameters['learning_rate']
    cfg.SOLVER.MAX_ITER = hyperparameters['num_epochs']
    cfg.SOLVER.STEPS = []        
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = hyperparameters['batch_size_per_img']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(list(info["classes"].keys())) 

    os.makedirs(args.output_dir, exist_ok=True)
    trainer = AffineAugsTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference with Detectron2")
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    parser.add_argument("--output_dir", required=False, help="Path to the output directory", default="output/model.pth")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device to use for training (default: cpu)")
    parser.add_argument("--data_dir", default="./data", help="Directory for custom annotated dataset")
    parser.add_argument("--num-workers", type=int, default=4, help="(CUDA only) Number of workers to use for data loading")

    args = parser.parse_args()
    main(args)