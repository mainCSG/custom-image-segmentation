# Custom Image Segmentation

This repository is an open source tool to build and train image segmentation models on *custom* data with the ability to inference existing models!

## Installation

Begin by creating a conda environment and activating it,
```console
conda create -n <ENV NAME> python=3.10; conda activate <ENV NAME>
```
and install all of the requirements,
```console
python -m pip install -r /path/to/requirements.txt
```

## Running scripts on a HPC cluster 
The scripts will work fine provided you have all the installed requirements. For the alliance canada
HPC cluster, the following commands executed in this order allows for proper environment set-up,
```console
virtualenv --no-download ENV; source ENV/bin/activate
```
then load up the requirements,
```console
module load python/3.10;
python -m pip install -r requirements.txt;
module load scipy-stack;
module spider gcc cuda opencv/4.9.0
```

## Training a Custom Model

### 1. Construct your dataset

Organize your dataset in the project directory so that it has the following structure,

```
├── data
│   ├── train
│   │   ├── <train0>.jpg
│   │   ├── ...
│   │   ├── <trainX>.jpg
│   │   ├── annotations.json
│   ├── val
│   │   ├── <val0>.jpg
│   │   ├── ...
│   │   ├── <valY>.jpg
│   │   ├── annotations.json
│   ├── test
│   │   ├── <test0>.jpg
│   │   ├── ...
│   │   ├── <testZ>.jpg
│   │   ├── annotations.json
```

> Note: You will have to generate the annotations yourself; this can
be done easily through an online annotation tool such as [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/via.html) and exporting it such that it follows standard formatting,

![Alt text](./custom_image_segmentation/examples/dot_configuration/photos/vgg_screenshot.jpg)

In the end, each of your `annotations.json` file should look like,

```json
{
  "<train0>.jpg": {
    "filename": "<image0>.jpg",
    "size": 1234,
    "regions": [
      {
        "shape_attributes": {
          "name": "polygon",
          "all_points_x": [
            119,
            94,
            200,
            200
          ],
          "all_points_y": [
            114,
            197,
            199,
            90
          ]
        },
        "region_attributes": {
          "label": "<image0: region 1 label>"
        }
      },
      ... more regions ...
    ],
  },
  ... more images ...
}
```

### 2. Design your configuration file

In addition to your dataset, you need a configuration file that tells detectron2 how to build and train your Mask-RCNN model. See the example below from [./examples/dot_configuration/configuration.yaml](./custom_image_segmentation/examples/dot_configuration/configuration.yaml).

```yaml
info:
    name: quantum dot configuration
    # Need to specify class dictionary for custom datasets,
    # 'nd' -> 'no dot'
    # 'ld' -> 'left dot'
    # 'cd' -> 'central dot'
    # 'rd' -> 'right dot'
    # 'dd' -> 'double dot'
    # Dictionary keys MUST match your annotations.json labels
    # Dictionary values can be anything as long as they are unique 
    classes: {"nd": 0, "ld": 1, "cd": 2, "rd": 3, "dd": 4}

hyperparameters:

    learning_rate: 2.0e-4
    num_epochs: 7000
    batch_num: 10
    batch_size_per_img: 128
```

### 3. Begin training your model

To begin training your model on your `data/train` folder, simply execute the following command,
```python
python train.py --config <configuration>.yaml --data_dir <path/to/data>
```

> Note: Use `--help` to see all of the flags available to you.

### 4. Validate your model

In order to validate your model's performance, you can inference the model on the `data/val` folder. It is recommended to first do it interactively using the `src/inference.ipynb` notebook.

### 5. Test your model

With your model validated, you can inference the model on the `data/test` folder. It is recommended to first do it interactively using the `src/inference.ipynb` notebook.

## Inference Trained Model 

### 1. Prerequisites

Before running inference, you need to provide two files along with your image of interest,

1. `<model>.pth`: This file contains the trained model weights.

2. `configuration.yaml`: This file contains the _exact_ configuration settings that were used for the trained model you are trying to inference (see **Training** section).

3. `<image>.jpg`: This file is the image you wish to inference

### 2. Begin inferencing your image (IN PROGRESS)

To begin training simply execute the following command,
```python
python inference.py --model_weights <model>.pth --config <configuration>.yaml  --image <image>.jpg 
```

> Note: Use `--help` to see all of the flags available to you.

