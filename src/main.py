#!usr/bin/env python
# -*- coding:utf-8 _*

"""
@File : main.py.py 
@Author : ljt
@Description: longitudal registration
@Time : 2021/5/26 10:49 
"""

# from ljt_sitk import *
# import SimpleITK as sitk
# import nibabel as nib
# import matplotlib.pyplot as plt
# from nilearn.image import new_img_like, resample_to_img
# import numpy as np
# from fsl.transform import flirt as flt
# from fsl.data.image import Image
# import io


"""
一、图像归一化
"""
# quiet_path = r"D:\ljt\Study\Madic\tmp_move_reg\data\quiet.nii"
# move_path = r"D:\ljt\Study\Madic\tmp_move_reg\data\move.nii"
#
#
# quiet_img = nii_read(quiet_path)
# move_img = nii_read(move_path)
#
# quiet_img = nii_norm(quiet_img)
# move_img = nii_norm(move_img)
#
# quiet_img = sitk.GetImageFromArray(quiet_img)
# move_img = sitk.GetImageFromArray(move_img)
#
# sitk.WriteImage(quiet_img, r"D:\ljt\Study\Madic\tmp_move_reg\data\quiet_norm.nii")
# sitk.WriteImage(move_img, r"D:\ljt\Study\Madic\tmp_move_reg\data\move_norm.nii")


"""
二、切割头部 版本一
"""

# # 静止
# """
# x：[24, 44]
# y: [32, 52]
# z: [32, 62]
# """
# path = r"D:\ljt\Study\Madic\Project\Longitudinal_Registration\data\quiet_norm.nii"
# img = nii_read(path) # z, y, x
# print(img.shape)
# tmp_img = img
#
# img = img[32:62, 32:52, 24:44]
#
# img = np.pad(img, ((32, tmp_img.shape[0]-62), (32, tmp_img.shape[1]-52), (24,tmp_img.shape[2]-44)))
# # img = np.pad(img, ((32, 46), (32, 17), (24,25)))
# print(img.shape)
# img[img < (img.mean())] = 0
# # for i in img:
# #     for j in i:
# #         for z in j:
# #             if 0 < z < (img.mean()):
# #                 print(z)
#
# # img[0:32 & 63:108, 0:32 & 53:69, 0:24 & 45:108] = 1
# # img[63:108, 53:69, 45:108] = 0
# img = sitk.GetImageFromArray(img)
# sitk.WriteImage(img, "D:\ljt\Study\Madic\Project\Longitudinal_Registration\data\quiet_norm_bet.nii")
#
# # 移动
# """
# x: [14, 34]
# y: [32, 52]
# z: [43, 73]
# """
#
# path = r"D:\ljt\Study\Madic\Project\Longitudinal_Registration\data\move_norm.nii"
# img = nii_read(path) # z, y, x
# tmp_img = img
# print(tmp_img.shape)
# img = img[43:73, 32:52, 14:34]
# img = np.pad(img, ((43, tmp_img.shape[0]-73), (32, tmp_img.shape[1]-52), (14, tmp_img.shape[2]-34)))
# img[img < (img.mean())] = 0
# # for i in img:
# #     for j in i:
# #         for z in j:
# #             if 0 < z < (img.mean()):
# #                 print(z)
# print(img.mean())
# print(img.shape)
#
# # img[0:43, 0:32, 0:14] = 1
# # img[74:108, 53:69, 35:69] = 1
# img = sitk.GetImageFromArray(img)
# sitk.WriteImage(img, "D:\ljt\Study\Madic\Project\Longitudinal_Registration\data\move_norm_bet.nii")
#
#


"""
二、切割头部
"""

# 静止
"""
x：[24, 44]
y: [32, 52]
z: [32, 62]
"""
# path = r"D:\ljt\Study\Madic\Project\Longitudinal_Registration\data\quiet_norm.nii"
# img = nii_read(path) # z, y, x
# print(img.shape)
# tmp_img = img
#
# img = img[32:73, 22:62, 4:54]
#
# # img = np.pad(img, ((32, tmp_img.shape[0]-73), (22, tmp_img.shape[1]-62), (4,tmp_img.shape[2]-54)))
# # img = np.pad(img, ((32, 46), (32, 17), (24,25)))
# print(img.shape)
# img[img < (img.mean())] = 0
# # for i in img:
# #     for j in i:
# #         for z in j:
# #             if 0 < z < (img.mean()):
# #                 print(z)
#
# # img[0:32 & 63:108, 0:32 & 53:69, 0:24 & 45:108] = 1
# # img[63:108, 53:69, 45:108] = 0
# img = sitk.GetImageFromArray(img)
# sitk.WriteImage(img, "D:\ljt\Study\Madic\Project\Longitudinal_Registration\data\quiet_norm_bet.nii")
#
# # 移动
# """
# x: [14, 34]
# y: [32, 52]
# z: [43, 73]
# """
#
# path = r"D:\ljt\Study\Madic\Project\Longitudinal_Registration\data\move_norm.nii"
# img = nii_read(path) # z, y, x
# tmp_img = img
# print(tmp_img.shape)
# img = img[32:73, 22:62, 4:54]
# # img = np.pad(img, ((32, tmp_img.shape[0]-73), (22, tmp_img.shape[1]-62), (4,tmp_img.shape[2]-54)))
# img[img < (img.mean())] = 0
# # for i in img:
# #     for j in i:
# #         for z in j:
# #             if 0 < z < (img.mean()):
# #                 print(z)
# print(img.mean())
# print(img.shape)
#
# # img[0:43, 0:32, 0:14] = 1
# # img[74:108, 53:69, 35:69] = 1
# img = sitk.GetImageFromArray(img)
# sitk.WriteImage(img, "D:\ljt\Study\Madic\Project\Longitudinal_Registration\data\move_norm_bet.nii")


"""
三、逆变换
"""

# import nibabel as nib
# from nilearn.image import new_img_like, resample_to_img
# import numpy as np
#
#
# # 移动图像路径
# path = "D:\ljt\Study\Madic\Project\Longitudinal_Registration\data\head\move_norm_bet.nii"
#
#
# #变换函数
# def scale_image(image, scale_affine):
#     # 图片仿射矩阵乘以变换矩阵
#     new_affine = image.affine.dot(scale_affine)
#     # 返回变换后新图片
#     return new_img_like(image, data=image.get_data(), affine=new_affine)
#
#
# # 使用FSL进行配准的结果
# convert_affine = flt.readFlirt(r"D:\ljt\Study\Madic\Project\Longitudinal_Registration\data\head\reg.mat")
# # convert_affine = np.array([[0.916691, 0.085028, -0.390446, 2.634863],
# #                            [-0.089665, 0.995952, 0.006373, 1.762251],
# #                            [0.389407, 0.029168, 0.920604, -3.719278],
# #                            [0, 0, 0, 1]])
#
#
#
# scale_affine = flt.readFlirt(r"D:\ljt\Study\Madic\Project\Longitudinal_Registration\data\head\invert_reg.mat")
# # scale_affine = np.array([[0.9166908604, -0.08966503066,  0.3894072834, -0.809028176],
# #                            [0.08502753445 ,0.99595162, 0.02916756409, -1.870670492 ],
# #                            [-0.3904461219 , 0.006372701772, 0.9206037409, 4.441522776],
# #                            [0, 0, 0, 1]])# 求矩阵的逆矩阵作为变换结果
# # scale_affine2 = np.linalg.inv(convert_affine)
#
#
# # scale_affine = flt.flirtMatrixToSform(convert_affine, Image(path), Image(r"D:\ljt\Study\Madic\Project\Longitudinal_Registration\data\head\quiet_norm_bet.nii"))
#
# print(scale_affine)
#
#
# nii_img = nib.load(path)
# new_img = scale_image(nii_img, scale_affine)
# # 重采样
# image_scaled = resample_to_img(new_img, nii_img, "linear")
# nib.save(image_scaled, r"D:\ljt\Study\Madic\Project\Longitudinal_Registration\data\head\image_sacled.nii")

# print("\nnii_img.affine:\n", nii_img.affine)
# print("\nnew_img.affine 逆矩阵:\n", np.linalg.inv(new_img.affine))
# print("\nscale_affine 逆矩阵:\n", np.linalg.inv(scale_affine))
# print("\nimage_scaled.affine:\n", image_scaled.affine)


"""
使用fslpy进行逆变换
"""

# from fsl.transform import flirt as flt
# from fsl.data.image import Image
#
# path1 = "D:\ljt\Study\Madic\Project\Longitudinal_Registration\data\head\move_norm_bet.nii"
# path2 = "D:\ljt\Study\Madic\Project\Longitudinal_Registration\data\head\quiet_norm_bet.nii"
#
# move_img =  Image(path1)
# quiet_img = Image(path2)
#
#
#
# reg_matrix = flt.readFlirt("D:\ljt\Study\Madic\Project\Longitudinal_Registration\data\head\invert_reg.mat")
# # print(reg_matrix)
#
# tmp = flt.sformToFlirtMatrix(move_img, quiet_img)
# print(tmp)
#
# tmp = flt.flirtMatrixToSform(reg_matrix, move_img, quiet_img)
# print(tmp)


"""
使用FSL进行时间轴配准，并保存结果为mat文件
"""

import os
import scipy.io as scio
from ljt_sitk import pre_for_fsl
import numpy as np

# 待配准的时间轴图像
move_path = r"../data/data_618/data_645_655"

# 参考图像 600s 静止 24次迭代结果
ref_path = r"../data/data_618/iter_24_subset_0.nii"

# 先对图像进行处理，使其能够FSL配准, 使用 pre_for_fsl()函数
pre = False
if pre:
    pre_for_fsl(ref_path)
    for i in range(50):
        img_path = move_path + "/{}".format(i) + "/nii/iter_09_subset_0.nii"
        pre_for_fsl(img_path)

# 配准
reg = False
if reg:
    for i in range(1):
        # 循环的时候变量名不能重复
        new_move_path = move_path + "/{}".format(i) + "/nii/iter_09_subset_0.nii"
        cmd = "flirt -in {} -ref {}  -out ../data/data_618/reg/reg_{}.nii".format(new_move_path, ref_path, i)
        # print(cmd)
        matrix = os.popen(cmd, 'r')
        print("已完成对第{}副图像的配准".format(i))
        print(matrix.readlines())

# 将矩阵合并
matrix_path_all = "../data/data_618/matrix/matrix_all.mat"
matrix_path = "../data/data_618/matrix"
all_mat = False
if all_mat:
    dic = {}
    now_time = 645.0
    frame = 0.2
    # end_time = 645.0
    for i in range(50):
        new_matrix_path = matrix_path + "/matrix_{}.mat".format(i)
        matrix = np.loadtxt(new_matrix_path).astype(np.float)
        keys = "{:.1f}_{:.1f}".format(now_time, now_time + frame)
        now_time += frame
        dic[keys] = matrix
    # print(dic)
    scio.savemat(matrix_path_all, dic)
#



# 读取mat文件中指定时间段矩阵的方法
matrix = scio.loadmat(matrix_path_all)
data = matrix["649.0_649.2"]
print(np.array(data))
