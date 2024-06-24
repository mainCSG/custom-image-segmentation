from pathlib import Path
import copy
import argparse
import json
import albumentations as A
from annotate import NumpyEncoder
from PIL import Image
import numpy as np
import cv2
from imantics import Mask

def get_exp_data_augmentation():
    return A.Compose([
        A.VerticalFlip(p=0.7),
        A.RandomBrightnessContrast(brightness_limit=(0.05,0.075), contrast_limit=(0.05, 0.1), p=0.8),
        A.GaussNoise(var_limit=(0.0,70.0), p=0.8),
        A.Affine(scale=(1,1.4), p=0.8),
        A.RandomToneCurve(p=0.8),
        A.AdvancedBlur(blur_limit=(9,11),p=0.8),
        A.RingingOvershoot(p=0.5),
    ], is_check_shapes=False)

def get_sim_data_augmentation():
    return A.Compose([
        A.VerticalFlip(p=0.7),
        A.RandomBrightnessContrast(brightness_limit=(0.05,0.1), contrast_limit=(0.05, 0.1), p=1),
        A.GaussNoise(var_limit=(50.0,120.0), p=0.8),
        A.GridDistortion(distort_limit=0.2, p=0.8),
        A.Affine(scale=(1,1.3), p=0.8),
        A.RandomToneCurve(p=1),
        A.AdvancedBlur(blur_limit=(9,11),p=1),
        A.RingingOvershoot(p=0.5)
    ], is_check_shapes=False)

def polygon_to_mask(polygon_coords: np.ndarray, image_shape: tuple) -> np.ndarray:
    mask = np.zeros(image_shape)
    cv2.fillPoly(mask, [polygon_coords], 255)
    return np.array(mask)

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def mask_to_polygon(mask: np.ndarray) -> np.ndarray:
    polygons = Mask(mask).polygons()
    if len(polygons.points) != 0:
        polygon_area = []
        for i in range(len(polygons.points)):
            x,y = zip(*polygons.points[i])
            polygon_area.append(PolyArea(x,y))
        idx = np.argmax(polygon_area)

        return polygons.points[idx]
    else: 
        return np.array([[0,0]])

def augment_image(filepath: str, annotations: str, augmentations: A.BaseCompose) -> tuple:
    image = Image.open(filepath)
    image_np = np.array(image)

    if isinstance(annotations["regions"], list): # custom JSONs are in list format need to make them the same
        annotations["regions"] = dict(enumerate(annotations["regions"]))

    # First convert all segmentations into masks for Albumentations
    masks = []
    for region_number, (region_index, region_info) in enumerate(annotations["regions"].items()):
        xs = region_info["shape_attributes"]["all_points_x"]
        ys = region_info["shape_attributes"]["all_points_y"]
        polygon = np.vstack((xs,ys)).T
        masks.append(
            polygon_to_mask(polygon, image.size)
        )

    # Augment the image and the masks
    augmentation = augmentations(
        image=image_np,
        masks=masks
    )
    augmented_image = augmentation["image"]
    augmented_masks = augmentation["masks"]

    # Convert masks back to the appropriate polygon format
    augmented_polygons = []
    for augmented_mask in augmented_masks:
        augmented_polygon = mask_to_polygon(augmented_mask)
        augmented_polygons.append(augmented_polygon)

    augmented_annotations = copy.deepcopy(annotations)
    for region_number, (region_index, region_info) in enumerate(augmented_annotations["regions"].items()):

        new_polygon = augmented_polygons[region_number]
        new_xs = new_polygon.T[0,:].tolist()
        new_ys = new_polygon.T[1,:].tolist()
        region_info["shape_attributes"]["all_points_x"] = new_xs
        region_info["shape_attributes"]["all_points_y"] = new_ys

    return augmented_image, augmented_annotations

def main(training_data_dir: Path, num_exp_aug: float, num_sim_aug: int) -> None:
    
    custom_annotations_file = training_data_dir / "annotations.json"
    with open(custom_annotations_file) as f:
        annotations = json.load(f)

    for image_filepath in training_data_dir.glob("*.jpg"):
        print(f"Augmenting {image_filepath}")
        if "exp" in image_filepath.name:
            num_of_augments = num_exp_aug
            augmentation_list = get_exp_data_augmentation()
        else:
            num_of_augments = num_sim_aug
            augmentation_list = get_sim_data_augmentation()

        for augment_number in range(num_of_augments):
            new_image, new_annotations = augment_image(
                filepath=image_filepath, 
                annotations=annotations[image_filepath.name],
                augmentations=augmentation_list
            )
            # Save new image
            augmented_image_name = image_filepath.stem + f"_aug{augment_number}" + ".jpg"
            augmented_image = Image.fromarray(new_image)
            augmented_image.save(training_data_dir / augmented_image_name)

            # Save new annotations
            new_annotations["filename"] = augmented_image_name
            annotations[augmented_image_name] = new_annotations

    print(f"Saving AUGMENTATED annotations for directory {dir}")
    annotations_json_file = dir / "annotations.json"
    with open(annotations_json_file, 'w') as f:
        json.dump(annotations, f, cls=NumpyEncoder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Augmentation of Training Data")

    parser.add_argument(
        "--num_sim_aug", 
        type=int,
        default=5, 
        help="Number of simulated augmentations to perform and save (default: 5)"
    )
    parser.add_argument(
        "--num_exp_aug", 
        type=float,
        default=2, 
        help="Number of simulated augmentations to perform and save (default: 2)"
    )
    args = parser.parse_args()

    # Set data directory
    data_dir = Path("data")
    training_dir = data_dir / "train"

    main(training_dir, args.num_exp_aug, args.num_sim_aug)