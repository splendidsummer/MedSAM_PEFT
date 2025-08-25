# -*- coding: utf-8 -*-
# %% 导入所需的包
# pip install connected-components-3d
import numpy as np  # 导入numpy用于数值计算

# import nibabel as nib
import SimpleITK as sitk  # 导入SimpleITK用于医学图像处理
import os  # 导入os用于文件和路径操作

join = os.path.join  # 简化路径拼接函数
from skimage import transform  # 导入skimage的transform模块用于图像变换
from tqdm import tqdm  # 导入tqdm用于进度条显示
import cc3d  # 导入cc3d用于三维连通域分析
from pathlib import Path  # 导入Path用于跨平台路径操作
home_dir = Path.home()  # 获取用户主目录路径

# 将nii图像转换为npz文件，包括原始图像和对应的掩码
modality = "CT"  # 设置模态为CT
anatomy = "Abd"  # 设置解剖部位
img_name_suffix = "_0000.nii.gz"  # 图像文件名后缀
gt_name_suffix = ".nii.gz"  # 标签文件名后缀
prefix = modality + "_" + anatomy + "_"  # 文件名前缀

nii_path = str(home_dir / "Datasets" / "MICCAI_FLARE_2022" / "Train" / "images")  # nii图像路径
gt_path = str(home_dir / "Datasets" / "MICCAI_FLARE_2022" / "Train" / "labels")  # 标签路径
npy_path = str(home_dir / "Datasets" / "MICCAI_FLARE_2022" / "Train" / "npy" / prefix[:-1])  # npy保存路径
os.makedirs(join(npy_path, "gts"), exist_ok=True)  # 创建标签保存文件夹
os.makedirs(join(npy_path, "imgs"), exist_ok=True)  # 创建图像保存文件夹

image_size = 1024  # 图像缩放大小
voxel_num_thre2d = 100  # 2D切片最小体素数阈值
voxel_num_thre3d = 1000  # 3D体最小体素数阈值

names = sorted(os.listdir(gt_path))  # 获取标签文件名列表并排序
print(f"ori \# files {len(names)=}")  # 打印原始文件数量
names = [
    name
    for name in names
    if os.path.exists(join(nii_path, name.split(gt_name_suffix)[0] + img_name_suffix))
]  # 只保留有对应图像的标签
print(f"after sanity check \# files {len(names)=}")  # 打印筛选后的文件数量

# 设置需要排除的标签id
remove_label_ids = [
    12
]  # 移除十二指肠标签，因为其分布零散，难以用框指定
tumor_id = None  # 仅在有多个肿瘤时设置，将语义掩码转为实例掩码: - 语义掩码：所有肿瘤像素的值都一样，无法区分不同肿瘤。- 实例掩码：每个肿瘤区域有独立的标签，可以区分每一个肿瘤。
# 设置窗位窗宽
# https://radiopaedia.org/articles/windowing-ct
WINDOW_LEVEL = 40  # CT窗位
WINDOW_WIDTH = 400  # CT窗宽

# %% 保存预处理后的图像和掩码为npz文件
for name in tqdm(names[:40]):  # 只处理前40个样本，剩下的用于验证
    image_name = name.split(gt_name_suffix)[0] + img_name_suffix  # 获取对应图像名
    gt_name = name  # 标签名
    gt_sitk = sitk.ReadImage(join(gt_path, gt_name))  # 读取标签nii文件: <class 'SimpleITK.SimpleITK.Image'>
    gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))  # 转为numpy数组并转为uint8: shape 
    # 移除指定标签id
    for remove_label_id in remove_label_ids:
        gt_data_ori[gt_data_ori == remove_label_id] = 0
    # 若有肿瘤id，则将肿瘤掩码转为实例掩码并从原掩码中移除
    if tumor_id is not None:
        tumor_bw = np.uint8(gt_data_ori == tumor_id)  # 获取肿瘤二值掩码
        gt_data_ori[tumor_bw > 0] = 0  # 从原掩码中移除肿瘤
        # 对肿瘤掩码做连通域分析，获得实例标签: tumor_inst：每个独立肿瘤区域被赋予不同的整数标签（0为背景，1、2、3...为不同肿瘤）。tumor_n：肿瘤实例的数量。
        tumor_inst, tumor_n = cc3d.connected_components(
            tumor_bw, connectivity=26, return_N=True
        )
        # 将肿瘤实例标签加回原掩码: 把每个肿瘤实例的标签加回到原始掩码中，并且标签值要避开原有的标签（通过加上np.max(gt_data_ori) + 1），确保不会和其他器官或结构的标签冲突。
        gt_data_ori[tumor_inst > 0] = (
            tumor_inst[tumor_inst > 0] + np.max(gt_data_ori) + 1
        )

    # 移除3D体素数小于阈值的连通域: threshold=voxel_num_thre3d：体素数阈值（如1000），小于这个体素数的连通域会被移除（设为0）。
    gt_data_ori = cc3d.dust(
        gt_data_ori, threshold=voxel_num_thre3d, connectivity=26, in_place=True
    )
    # 对每个2D切片移除体素数小于阈值的小连通域
    for slice_i in range(gt_data_ori.shape[0]):
        gt_i = gt_data_ori[slice_i, :, :]  # 取出第i个切片
        # 移除小连通域
        gt_data_ori[slice_i, :, :] = cc3d.dust(
            gt_i, threshold=voxel_num_thre2d, connectivity=8, in_place=True
        )
    # 找到非零切片的索引: np.where(gt_data_ori > 0) 会返回所有大于0（即有前景/标签/器官/肿瘤等像素）的三维坐标索引，分别对应 (z, y, x)。
    z_index, _, _ = np.where(gt_data_ori > 0)
    z_index = np.unique(z_index)  # 由于一个切片里可能有多个前景像素，z_index 里会有重复的切片编号。

    if len(z_index) > 0:
        # 只保留非零切片
        gt_roi = gt_data_ori[z_index, :, :]
        # 读取对应的原始图像
        img_sitk = sitk.ReadImage(join(nii_path, image_name))
        image_data = sitk.GetArrayFromImage(img_sitk)
        # nii预处理开始: 判断当前处理的图像是否为CT。如果是CT，采用窗宽窗位进行预处理。
        if modality == "CT":
            lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2  # 计算窗下界 : 使用 WW/WL 后，超出区间的 HU 值会被截断
            upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2  # 计算窗上界 : 低于 (WL - WW/2) 的值 → 设为最小灰度; 高于 (WL + WW/2) 的值 → 设为最大灰度
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)  # 截断
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )  # 归一化到0-255
        else:
            # 计算非零像素（即前景区域）的0.5分位数和99.5分位数，分别作为下界和上界。
            # 这样做的目的是去除极端异常值（如噪声、伪影），只保留主要灰度分布区间，增强对比度。
            lower_bound, upper_bound = np.percentile(
                image_data[image_data > 0], 0.5
            ), np.percentile(image_data[image_data > 0], 99.5)

            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            # 对截断后的图像做线性归一化，将像素值缩放到0~255区间，便于后续保存和可视化。
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
            # 对于原始图像中为0的像素（通常代表背景），归一化后强制设为0，确保背景不会被拉伸成非零值。
            image_data_pre[image_data == 0] = 0  # 背景设为0

        image_data_pre = np.uint8(image_data_pre)  # 转为uint8
        img_roi = image_data_pre[z_index, :, :]  # 只保留非零切片
        np.savez_compressed(join(npy_path, prefix + gt_name.split(gt_name_suffix)[0]+'.npz'), imgs=img_roi, gts=gt_roi, spacing=img_sitk.GetSpacing())
        # 保存裁剪后的图像和标签为nii文件，便于检查
        # 便于人工检查：NIfTI是医学图像领域的标准格式，医生或研究者可以用3D Slicer、ITK-SNAP等专业软件直接打开，直观地检查预处理结果是否正确。
        img_roi_sitk = sitk.GetImageFromArray(img_roi)
        gt_roi_sitk = sitk.GetImageFromArray(gt_roi)
        sitk.WriteImage(
            img_roi_sitk,
            join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + "_img.nii.gz"),
        )
        sitk.WriteImage(
            gt_roi_sitk,
            join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + "_gt.nii.gz"),
        )
        # 将每个切片保存为npy文件:roi represents 非零切片
        for i in range(img_roi.shape[0]):
            img_i = img_roi[i, :, :]  # 取出第i个切片
            img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)  # 转为3通道: 兼容主流深度学习模型很多经典的图像分割、分类网络（如ResNet、UNet、SAM等）都是为3通道RGB自然图像设计的，输入shape要求为(H, W, 3)。如果直接输入单通道，模型结构和预训练权重都无法直接
            resize_img_skimg = transform.resize(
                img_3c,
                (image_size, image_size),
                order=3,
                preserve_range=True,
                mode="constant",
                anti_aliasing=True,
            )  # 缩放图像
            resize_img_skimg_01 = (resize_img_skimg - resize_img_skimg.min()) / np.clip(
                resize_img_skimg.max() - resize_img_skimg.min(), a_min=1e-8, a_max=None
            )  # 归一化到[0, 1]
            gt_i = gt_roi[i, :, :]  # 取出第i个标签切片
            resize_gt_skimg = transform.resize(
                gt_i,
                (image_size, image_size),  #  (1024, 1024)
                order=0,
                preserve_range=True,
                mode="constant",
                anti_aliasing=False,
            )  # 缩放标签
            resize_gt_skimg = np.uint8(resize_gt_skimg)  # 转为uint8
            assert resize_img_skimg_01.shape[:2] == resize_gt_skimg.shape  # 检查尺寸一致

            #  最终保存的文件名为：
            # 图像切片：CT_Abd_case123-005.npy，路径为{npy_path}/imgs/CT_Abd_case123-005.npy
            # 标签切片：CT_Abd_case123-005.npy，路径为{npy_path}/gts/CT_Abd_case123-005.npy
            # 每个切片的编号会依次递增（如-000、-001、-002...）。
            np.save(
                join(
                    npy_path,
                    "imgs",
                    prefix
                    + gt_name.split(gt_name_suffix)[0]
                    + "-"
                    + str(i).zfill(3)
                    + ".npy",
                ),
                resize_img_skimg_01,
            )  # 保存图像切片
            np.save(
                join(
                    npy_path,
                    "gts",
                    prefix
                    + gt_name.split(gt_name_suffix)[0]
                    + "-"
                    + str(i).zfill(3)
                    + ".npy",
                ),
                resize_gt_skimg,
            )  # 保存标签切片
