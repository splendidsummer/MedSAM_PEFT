## Data Format

### DICOM文件
👉 医学数字成像与通信标准文件
全称是：Digital Imaging and Communications in Medicine (DICOM)。

🔹 解释

DICOM 是国际医学影像领域的通用标准，专门用于：

存储 医学影像（如 CT、MRI、超声、PET 等）。

传输 医学影像及其相关信息（比如病人信息、扫描参数、成像设备信息）。

它不仅仅是一张图片，而是一个包含 影像数据 + 元数据（Metadata） 的文件。

影像数据：CT、MRI 扫描得到的像素/体素矩阵。

元数据：例如病人姓名、检查日期、扫描设备型号、像素间距 (Pixel Spacing)、层厚 (Slice Thickness) 等。

🔹 举例

一份 CT 检查可能会生成几百个 DICOM 文件：

每个 DICOM 文件对应一张 2D 切片图像；

这些文件堆叠起来，就可以还原为 3D 体数据；

里面还能读取体素大小、成像参数，用于计算体积或做 3D 重建。

🔹 中文总结

DICOM 文件 = 医学影像的行业标准格式，里面既有图像（像素/体素），又有扫描和病人相关的描述信息。
<!-- TODO: What is 体素 in data?--> 


## 🏥 常见医学影像文件格式

| 格式 | 中文名称 | 特点 | 常见应用 |
|------|---------|------|----------|
| **DICOM (.dcm)** | 医学数字成像与通信标准 | - 行业标准，包含 **影像数据 + 元数据（病人信息、扫描参数等）**<br>- 支持 2D 切片序列和 3D 体数据<br>- 文件大、冗余信息多 | CT、MRI、超声、PET、放射科日常诊断 |
| **NIfTI (.nii / .nii.gz)** | 神经影像学信息交换格式 | - 专门为医学图像分析设计<br>- 支持 **3D/4D 数据**（体积 + 时间序列）<br>- 元数据比 DICOM 简洁<br>- 压缩版本 `.nii.gz` 常用 | 脑科学 (fMRI, DTI)，科研数据处理 |
| **Analyze 7.5 (.hdr + .img)** | 医学图像分析格式 | - 由 Mayo Clinic 提出<br>- 分为两部分：`.hdr`（头文件，存储元信息），`.img`（图像体数据）<br>- 已逐渐被 NIfTI 取代 | 早期神经影像研究 |
| **MHD / RAW (.mhd + .raw)** | MetaImage 格式 | - `.mhd` 存元信息，`.raw` 存图像数据<br>- 简单灵活，便于科研使用 | 图像处理、仿真软件 (ITK, VTK) |
| **NRRD (.nrrd)** | Nearly Raw Raster Data | - 结构清晰，支持多维数据<br>- 和 ITK/VTK 软件配合使用 | 医学图像科研、可视化 |
| **MINC (.mnc)** | Medical Imaging NetCDF | - 基于 NetCDF，支持复杂元信息<br>- 灵活但社区较小 | 脑影像、医学研究 |
| **MHA (.mha)** | MetaImage 单文件格式 | - 与 `.mhd + .raw` 类似，但所有数据和头信息合并在一个文件中 | 图像科研处理 |


## 数据预处理建议

在医学影像分割任务中，数据预处理是提升模型性能和泛化能力的重要环节。常见的数据预处理方法包括：

1. 标准化/归一化：对CT/MR影像进行窗宽窗位调整或分位数归一化，将像素值归一化到[0,1]或[0,255]区间。
2. 去噪与小连通域移除：利用连通域分析去除体素数较小的噪声区域，提升标签质量。
3. 切片与裁剪：只保留有标注的切片，自动裁剪ROI，减少无关数据。
4. 尺寸统一：将所有切片resize到统一尺寸，便于批量训练。
5. 标签处理：合并/去除不需要的标签，或将语义分割标签转为实例分割标签。
6. 数据增强：训练时可加入旋转、缩放、弹性变形、强度扰动等增强方式。
7. 数据格式转换：将nii、dcm等医学影像格式转换为npy、npz等高效格式，便于快速读取和训练。
8. 空间信息保留：保留像素间距（spacing）、方向等元数据，便于后续结果还原到原始空间。

根据具体任务和数据特点，可灵活选择和组合上述方法。
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
