# Dataset preparation

This repository supports multiple fine-grained datasets. The paper reports results on **CUB-200-2011**, **Stanford Dogs**, and **Oxford 102 Flowers**.

Raw images are not redistributed due to dataset licenses. Please download them from their official sources.

## CUB-200-2011

Expected structure:

```text
$PAAE_DATA_ROOT/
└── CUB_200_2011/
    └── CUB_200_2011/
        ├── images/
        ├── images.txt
        ├── image_class_labels.txt
        ├── train_test_split.txt
        └── classes.txt
```

The implementation uses the official train/test split from `train_test_split.txt`.

## Stanford Dogs

Expected structure:

```text
$PAAE_DATA_ROOT/
└── dogs/
    ├── Images/
    ├── Annotation/
    ├── train_list.mat
    └── test_list.mat
```

The implementation uses `train_list.mat` and `test_list.mat` as the standard train/test split.

## Oxford 102 Flowers

Download the following from the official source and place them under `$PAAE_DATA_ROOT/flowers/`:

- `102flowers.tgz` (extract into `jpg/`): https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
- `imagelabels.mat`: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
- `setid.mat`: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat

Expected structure:

```text
$PAAE_DATA_ROOT/
└── flowers/
    ├── jpg/                 # image_00001.jpg ... image_08189.jpg
    ├── imagelabels.mat
    └── setid.mat
```

The data loader expects `train.txt` and `test.txt` under `flowers/`. Generate them once with the helper script in the project root:

```bash
python prepare_flowers.py --root $PAAE_DATA_ROOT/flowers
```

This follows the common FGVC setting: train = trnid + valid (2040 images), test = tstid (6149 images). Pass `--no-val` to train on the official 1020-image split only. Labels are written 0-indexed (0..101).

## Setting the dataset root

Linux/macOS:

```bash
export PAAE_DATA_ROOT=/path/to/datasets
```

Windows PowerShell:

```powershell
$env:PAAE_DATA_ROOT="D:\path\to\datasets"
```
