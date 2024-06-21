from convert_npy_to_jpg import extract_raw_data
from pathlib import Path
from PIL import Image
import os
import numpy as np
import json
import skimage as sk

# From QFlow
class_dict = {
    0: "(0,0)",
    1: "(0,1)",
    2: "(0,2)",
    3: "(1,0)",
    4: "(1,1)",
    5: "(1,2)",
    6: "(2,0)",
    7: "(2,1)",
    8: "(2,2)"
}

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def organize_array_clockwise(arr: np.ndarray) -> np.ndarray:
            
    # Calculate the centroid of the points
    centroid = np.mean(arr, axis=0)

    # Calculate the angle of each point with respect to the centroid
    angles = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])

    # Sort the points based on the angles in clockwise order
    indices = np.argsort(angles)
    sorted_arr = arr[indices]

    return sorted_arr       

def find_polygon_centroid(coordinates: np.ndarray):
    n = len(coordinates)
    
    # Check if all x-values are the same
    x_values = [x for x, _ in coordinates]
    if len(set(x_values)) == 1:
        centroid_x = x_values[0]
        
        # Calculate the average of the y-values
        y_values = [y for _, y in coordinates]
        centroid_y = sum(y_values) / n
    else:
        # Calculate the signed area of the polygon
        signed_area = 0
        for i in range(n):
            x_i, y_i = coordinates[i]
            x_j, y_j = coordinates[(i + 1) % n]
            signed_area += (x_i * y_j - x_j * y_i)
        signed_area *= 0.5
        
        # Calculate the coordinates of the centroid
        centroid_x = 0
        centroid_y = 0
        for i in range(n):
            x_i, y_i = coordinates[i]
            x_j, y_j = coordinates[(i + 1) % n]
            factor = x_i * y_j - x_j * y_i
            centroid_x += (x_i + x_j) * factor
            centroid_y += (y_i + y_j) * factor
        centroid_x /= (6 * signed_area)
        centroid_y /= (6 * signed_area)
    
    return centroid_x, centroid_y

def get_state_polygons(raw_data: np.ndarray) -> tuple[str, np.ndarray]:

    background_value = -1 # barrier or no dot
    correction = 2 # Need integer values for Mask-RCNN

    states_regions_labelled = sk.measure.label(
        (correction * raw_data).astype(np.uint8), 
        background=background_value, 
        # connectivity=1
    )

    states_regions = sk.measure.regionprops(states_regions_labelled)

    results = []
    for index, region in enumerate(states_regions):
        region_coords = region.coords
        
        # Get boundaries of coordinates
        temp = {}
        for row in region_coords:
            key = row[0]
            value = row[1]
            if key not in temp:
                temp[key] = [value, value]  # Initialize with the current value
            else:
                temp[key][0] = min(temp[key][0], value)  # Update minimum value
                temp[key][1] = max(temp[key][1], value)  # Update maximum value

        boundary_coords = np.array([[key, minmax[0]] for key, minmax in temp.items()] + [[key, minmax[1]] for key, minmax in temp.items()])

        y, x = boundary_coords.T
        polygon = np.vstack((x,y)).T

        # Filter any weird artifacts for detectron2
        if len(x) <= 10 or len(y) <= 10:
            continue
        
        polygon_clockwise = organize_array_clockwise(polygon)
        centroid = find_polygon_centroid(polygon_clockwise)
        centroid_y, centroid_x = centroid
        polygon_state = raw_data[
            centroid_x.astype(int), 
            centroid_y.astype(int)
            ]

        # ignore -1 bc those are not spin qubit states
        if polygon_state == -1:
            continue
        polygon_class = class_dict[polygon_state]
        
        results.append((polygon_class, polygon_clockwise))

    return results

def annotate_polygons(file: str, polygons: dict[str, np.ndarray]) -> dict:
    image_info = {}

    # General image information
    image_info["filename"] = file.name
    img = Image.open(file)
    width, height = img.size
    image_info["height"] = height
    image_info["width"] = width
    image_info["size"] = height * width
    image_info["regions"] = {}

    # Loop through all polygons and store annotations
    for index, class_polygon in enumerate(polygons):

        class_name, polygon = class_polygon

        image_info["regions"][index] = {}

        image_info["regions"][index]["shape_attributes"] = {}

        image_info["regions"][index]["shape_attributes"]["name"] = "polygon"

        image_info["regions"][index]["shape_attributes"]["all_points_x"] = polygon[:,0]
        image_info["regions"][index]["shape_attributes"]["all_points_y"] = polygon[:,1]

        image_info["regions"][index]["region_attributes"] = {}
        image_info["regions"][index]["region_attributes"]["label"] = class_name

    return image_info

def main():
    data_dir = Path("data")

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    all_dirs = [train_dir, val_dir, test_dir]

    for dir in all_dirs:
        annotations = {}
        for file in os.listdir(dir):
            if file.endswith(".jpg"):
                print(f"Processing {dir/file}")
                npy_file = Path(file).stem + ".npy"
                try:
                    raw_data, _, _ = extract_raw_data(dir / npy_file, data_type='charge')
                except KeyError:
                    print(f"Need custom annotations for file {npy_file}!")
                try:
                    polygons = get_state_polygons(raw_data)
                    file_annotation = annotate_polygons(dir / file, polygons)
                    annotations[file] = file_annotation
                except AttributeError:
                    print(f"Need custom annotations for file {npy_file}!")

        print(f"Saving annotations for directory {dir}")
        annotations_json_file = dir / "annotations.json"
        with open(annotations_json_file, 'w') as f:
            json.dump(annotations, f, cls=NumpyEncoder)


if __name__ == "__main__":
    main()
