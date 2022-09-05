# coding=utf-8
import numpy as np
import torch
import time
import scipy.io as sio
from PIL import Image
import os
import models
import common_func
import Train_func
import spectral
import matplotlib.pyplot as plt

use_gpu = torch.cuda.is_available()  # true

# 训练
'''
train(input_x, input_y, un_idx_train, un_idx_valid, h1, h2,learn_rate, w_decay, ground_truth)
x:input_x矩阵
y:input_y矩阵
idx_tra:？？
idx_vld:？？？
hid_dim_1:隐藏层1层数
hid_dim_2:隐藏层2层数
lrn_rt:学习率
w_decay:衰减率
ground_truth:真实结果
'''


def train(x, y, idx_tra, idx_vld, hid_dim_1, hid_dim_2, lrn_rt, w_decay, ground_truth):
    # 定义模型1和2
    model_1 = models.AutoEncoder(in_dim=127, hid_dim1=hid_dim_1, hid_dim2=hid_dim_2)
    model_2 = models.AutoEncoder(in_dim=127, hid_dim1=hid_dim_1, hid_dim2=hid_dim_2)
    # 使用模型的apply方法对模型进行权重初始化。
    model_1.apply(common_func.initNetParams)
    model_2.apply(common_func.initNetParams)
    # 定义epoch和batch_size
    epoch, bth_sz, = 200, 256
    # 训练 预测因子
    model_1, loss_m1, predt_y, T_ls_1, V_ls_1 = Train_func.train_predict(
        x, y, model_1, idx_tra, idx_vld, epoch, bth_sz, lrn_rt, w_decay, use_gpu)
    model_2, loss_m2, predt_x, T_ls_2, V_ls_2 = Train_func.train_predict(
        y, x, model_2, idx_tra, idx_vld, epoch, bth_sz, lrn_rt, w_decay, use_gpu)
    # 将两个方向的两个损耗图的 最小值 作为最终异常变化强度图
    loss_result1 = np.minimum(loss_m1, loss_m2)
    # 画ROC图
    # loss_result1.transpose()为什么要用这个的转置
    X1, Y1, aucc = common_func.plot_roc(loss_result1.transpose(), ground_truth)
    print("最终的aucc is ", aucc, '\n')
    return loss_result1, loss_m1, loss_m2, predt_y, predt_x


if __name__ == '__main__':
    start = time.time()
    # 调用GPU的当前环境可用cuda标号是0
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("-----------------------使用gpu中--------------------------------")
    '''Step1 : Read Data读取数据'''
    path_name = 'D:/Code/B_HSI/AnomalyDetection/ACDA/'
    # 可以切换不同的数据集测试
    EX, img_2, train_smp, valid_smp = 'EX1', 'img_2', 'un_idx_train1', 'un_idx_valid1'
    ground_truth = Image.open(path_name + 'ref_EX1.bmp')
    if EX == 'EX2':
        img_2, train_smp, valid_smp = 'img_3', 'un_idx_train2', 'un_idx_valid2'
        ground_truth = Image.open(path_name + 'ref_EX2.bmp')
    # read image data
    # img_data : img_1,img_2,img_3(de-striping, noise-whitening and spectrally binning)去条化、噪声白化和频谱分箱
    data_filename = 'img_data.mat'
    data = sio.loadmat(path_name + data_filename)
    # 显示图像
    plt.subplot(1, 2, 1)
    plt.imshow(data['img_1'][:, :, 100])
    plt.subplot(1, 2, 2)
    plt.imshow(data['img_2'][:, :, 20])
    plt.show()

    img_x0 = data['img_1']
    img_y0 = data['img_2']
    '''    input_x = img_x0.transpose(2, 1, 0)改变索引值'''
    input_x = img_x0.transpose(2, 1, 0)
    input_y = img_y0.transpose(2, 1, 0)
    # read pre-train samples from pre-training result of USFA
    # for different training strategy(only replace the training samples)
    TrainSmp_filename = 'groundtruth_samples.mat'
    # groundtruth_samples random_samples pretrain_samples
    TrainSmp = sio.loadmat(path_name + TrainSmp_filename)
    # 降维
    '''un_idx_train，un_idx_valid是干嘛的？？？？'''
    un_idx_train = TrainSmp[train_smp].squeeze()
    un_idx_valid = TrainSmp[valid_smp].squeeze()
    img_channel, img_height, img_width = input_x.shape

    '''Step2 : for experiemntal result 实验结果'''
    # 用0填充
    Loss_result = np.zeros([img_height, img_width], dtype=float)
    # 超参数
    h1, h2 = 60, 40  # 127, 127
    learn_rate, w_decay = 0.001, 0.001
    # 迭代次数
    iteration = 1

    for i in np.arange(1, 1 + iteration):
        print('epoch i =', i)
        loss_result, ls_m1, ls_m2, prdt_y, prdt_x = train(input_x, input_y, un_idx_train, un_idx_valid, h1, h2,
                                                          learn_rate, w_decay, ground_truth)
        Loss_result = Loss_result + loss_result
    Loss_result = Loss_result / iteration
    X, Y, auc = common_func.plot_roc(Loss_result.transpose(), ground_truth)

    print("auc is ", auc, '\n')
    print("-------------Ending---------------")
    print("     ")
    print(EX)
    end = time.time()
    print("共用时", (end - start), "秒")
