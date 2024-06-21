# Dot Configuration

This model takes in a [charge stability diagram](https://www.qutube.nl/machine-learning-for-semiconductor-quantum-devices/charge-stability-diagrams) specifically measured by a charge sensor (not direct current!) and is able to segment the image into the various dot charge states.

If you wish to recreate this model, follow the steps below to generate the custom dataset by processing the external [QFlow](https://data.nist.gov/od/id/66492819760D3FF6E05324570681BA721894) dataset from NIST. 

### 1. Download 

Begin by downloading the required datasets,
```python
python src/download.py --version lite
```

>Note: Only the "lite" dataset from qflow has pre-defined charge states

### 2. Partition

Partition into `train`, `val` and `test` datasets,
```python
python src/partition.py --train 0.8 --val 0.1 --test 0.1
```

### 3. Image-ify

Convert each `.npy` file found in `data/[train,val,test]` to its respective image,
```python
python src/convert_npy_to_jpg.py --data_type sensor
```

### 4. Annotate 

Annotate `train` and `val` datasets,
```python
python src/annotate.py
```

### Example Training Data

![Alt text](photos/example_training_data.svg)

