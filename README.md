# MMSports-Seg-AutoID

## Setup Environment

**Step1:** Create a conda environment with Python=3.9. Command:

```bash
conda create --name mmsports-py39 python=3.9
```

**Step2:** Install Pytorch with CUDA 11.8. Command:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Step3:** Install mmdetection 3.1.0 (we will use it as a dependency). Command:

```bash
pip install -U openmim
mim install "mmengine==0.8.4"
mim install "mmcv>=2.0.0"
mim install "mmdet==3.1.0"
```

**Step4:** We also use some backbones from mmpretrain. Command:

```bash
mim install "mmpretrain[multimodal]>=1.0.0rc8"
```

Other packages can be install by pip or conda --forge. Please check requirements.txt for more detail

## Data preparation

**Step1:** extract and prepare dataset folder as follow:

```text
data
├── annotaions
│   ├── challenge.json
│   ├── test.json
│   ├── train.json
│   ├── val.json
│   ├── trainval.json
│   └── trainvaltest.json
└── (image folders)
...
```

**Step2:** we also use a specialized CopyPaste augmentation technique. To train models with our CopyPaste technique, you need to prepare a folder that contains all ground-truth instances from trainning set. Run the bellow command:

```bash
python utils/extract_objects.py data_root mode
```

A **{mode}_cropped_objects** folder will be created inside the data folder.

**Step3 (optional):** In case you want to visualize the augmented images, we provide a tool to help you. Run the below command:

```bash
python utils/browser_dataset.py data_config --output-dir data_sample --not-show
```

## Learn about configs

### Get ready with mmdet config

Everything in mmdet is about config. Before starting with mmdet, you should read their tutorial first. mmdetection (or mmdet) is built on top of a core package called mmengine. I highly recommend you to check their homepage and github for detail documentation and tutorials.

- Github: <https://github.com/open-mmlab/mmengine>

- Hompage: <https://mmengine.readthedocs.io/en/latest/get_started/introduction.html>

Or read their config explanation at least:

- Github: <https://github.com/open-mmlab/mmengine/tree/main/docs/en/tutorials>

### Our configs

Our dataset config can be found at:

```text
./configs/_base_/datasets/mmsports_instance.py
```

Our model configs:

```text
./configs/exp/{model_name}/*.py
```

## Train a model

### Single GPU training

Create a config for your experiment and save in ./configs folder. Then run below command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py config_path model_name
```

Training outputs (checkpoints, logs, merged config, tensorboard log, etc.) will be available in ./output/(model_name) folder

### Distributed training

We also support distributed training. Run the below command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py config_path model_name --launcher pytorch
```

### Run SWA

To apply SWA (Stochastic Weight Averaging) after the main training is finished. Run the bellow command (you can use single GPU or Distributed training):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 swa_train.py config_path last_checkpoint model_name --launcher pytorch
```

After that, use utils/get_swa_model.py to average all checkpoints exported by swa training process:

```bash
python utils/get_swa_model.py checkpoint_dir
```

## Export submission result

After training a model, you can export submission result. Submission can be produced on val set or test set (if val set, you can see the evalution score). Run below commands:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py config_path checkpoint_path [--valid] [--tta]
```

If you produce result on test set, omit --valid.
If you want to apply TTA (test time augmentation), use --tta

All results should be in ./output folder. After running the test command, you can find the inference result as "challenge-output.json". Now it is ready to submit to the test server.

Our last checkpoint and config for the competition can be found [here](https://drive.google.com/drive/folders/1x5jmwaHIoSHcs2QgBJDQ9ZhW3kcig0Fk?usp=sharing)

Our technical report can be found (TBD)
