# MedSAM-PEFT: Medical Image Segmentation with Segment Anything Model

本仓库基于 **MedSAM** 模型，结合参数高效微调（PEFT）方法，对医学图像分割任务进行了实验与结果总结。

---

## 📌 项目简介
- 模型基于 [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)，并针对医学图像进行了适配与微调。  
- 支持 **单 GPU / 多 GPU** 训练与推理。  
- 实验任务涵盖 **CT、MRI、病理切片、显微镜图像** 等多种模态。  

---

## ⚙️ 环境配置
```bash
git clone https://github.com/splendidsummer/MedSAM_PEFT.git
cd MedSAM_PEFT
conda create -n medsam python=3.10 -y
conda activate medsam
pip install -r requirements.txt
