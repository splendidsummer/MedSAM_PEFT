# MedSAM
This is the official repository for MedSAM: Segment Anything in Medical Images.

Welcome to join our [mailing list](https://forms.gle/hk4Efp6uWnhjUHFP6) to get updates.


## News

- 2025.04.07: Release [MedSAM2](https://github.com/bowang-lab/MedSAM2) for 3D and video segmentation.
- 2025.02: Welcome to join CVPR 2025 Challenges: [Interactive](https://www.codabench.org/competitions/5263/) and [Text-guided](https://www.codabench.org/competitions/5651/) 3D Biomedical Image Segmentation
- 2024.01.15: Welcome to join [CVPR 2024 Challenge: MedSAM on Laptop](https://www.codabench.org/competitions/1847/)
- 2024.01.15: Release [LiteMedSAM](https://github.com/bowang-lab/MedSAM/blob/LiteMedSAM/README.md) and [3D Slicer Plugin](https://github.com/bowang-lab/MedSAMSlicer), 10x faster than MedSAM! 


## Installation
1. Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it `conda activate medsam`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. `git clone https://github.com/bowang-lab/MedSAM`
4. Enter the MedSAM folder `cd MedSAM` and run `pip install -e .`
                                              #python -m pip install -e . | cat


## Get Started
### Download Pretrained Weights
Download the [model checkpoint](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN?usp=drive_link) and place it at the current repo, e.g., `work_dir/MedSAM/medsam_vit_b`. 

### Download Dataset
[**MICCAI FLARE 2022 Chanllenge**](https://flare22.grand-challenge.org/Dataset/). 
**Dataset Description:** 
* **Training set:** 50 labeled cases with pancreas disease and 2000 unlabeled cases with liver, kidney, spleen, or pancreas diseases.
* **Tuning set:** 50 cases with liver, kidney, spleen, or pancreas diseases.
* **Validation set:**  The internal validation set contains 100 cases with liver, kidney, spleen, or pancreas diseases and 100 cases with uterine corpus endometrial, urothelial bladder, stomach, sarcomas, or ovarian diseases. We also conducted a post-challenge analysis on three external validation sets and each one has 200 cases. 

[**Example Dataset with full mask annotation labels**] from the above challenge can be downloaded through [google drive link](https://drive.google.com/drive/folders/1oZGLgM4lKpIeBhtK8i0zRt2MpmoKma6Q)
![Labeled 50 Cases for Training](/images/miccai_flare_training.png)  

Unzip and place the dataset in folder assigned by the code lines in `pre_CT_MR.py`. 

``` python
# Extract the images.zip and place all the data under the below path
nii_path = str(home_dir / "Datasets" / "MICCAI_FLARE_2022" / "Train" / "images")  # nii图像路径
# Extract the labels.zip and place all the data under the below path
gt_path = str(home_dir / "Datasets" / "MICCAI_FLARE_2022" / "Train" / "labels")  # 标签路径
``` 

## Inference  
1. Command line

```bash
python MedSAM_Inference.py # segment the demo image
```

Segment other images with the following flags
```bash
-i input_img
-o output path
--box bounding box of the segmentation target
```

2. Jupyter-notebook

We provide a step-by-step tutorial on [CoLab](https://colab.research.google.com/drive/19WNtRMbpsxeqimBlmJwtd1dzpaIvK2FZ?usp=sharing)

You can also run it locally with `tutorial_quickstart.ipynb`.

3. GUI

Install `PyQt5` with [pip](https://pypi.org/project/PyQt5/): `pip install PyQt5 ` or [conda](https://anaconda.org/anaconda/pyqt): `conda install -c anaconda pyqt`

```bash
python gui.py
```

Load the image to the GUI and specify segmentation targets by drawing bounding boxes.



https://github.com/bowang-lab/MedSAM/assets/19947331/a8d94b4d-0221-4d09-a43a-1251842487ee





## Model Training

### Data preprocessing

Download [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place it at `work_dir/SAM/sam_vit_b_01ec64.pth` .

Download the demo [dataset](https://zenodo.org/record/7860267) and unzip it to `data/FLARE22Train/`.

This dataset contains 50 abdomen CT scans and each scan contains an annotation mask with 13 organs. The names of the organ label are available at [MICCAI FLARE2022](https://flare22.grand-challenge.org/).

Run pre-processing

Install `cc3d`: `pip install connected-components-3d`

```bash
python pre_CT_MR.py
```

- split dataset: 80% for training and 20% for testing
- adjust CT scans to [soft tissue](https://radiopaedia.org/articles/windowing-ct) window level (40) and width (400)
- max-min normalization
- resample image size to `1024x1024`
- save the pre-processed images and labels as `npy` files


### Training on multiple GPUs (Recommend)

The model was trained on five A100 nodes and each node has four GPUs (80G) (20 A100 GPUs in total). Please use the slurm script to start the training process.

```bash
sbatch train_multi_gpus.sh
```

When the training process is done, please convert the checkpoint to SAM's format for convenient inference.

```bash
python utils/ckpt_convert.py # Please set the corresponding checkpoint path first
```

### Training on one GPU

```bash
python train_one_gpu.py
```

## Acknowledgements
- We highly appreciate all the challenge organizers and dataset owners for providing the public dataset to the community.
- We thank Meta AI for making the source code of [segment anything](https://github.com/facebookresearch/segment-anything) publicly available.
- We also thank Alexandre Bonnet for sharing this great [blog](https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/)


## Reference

```
@article{MedSAM,
  title={Segment Anything in Medical Images},
  author={Ma, Jun and He, Yuting and Li, Feifei and Han, Lin and You, Chenyu and Wang, Bo},
  journal={Nature Communications},
  volume={15},
  pages={654},
  year={2024}
}
```
