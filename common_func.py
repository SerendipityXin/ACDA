# coding=utf-8
import numpy as np
from torch import nn
from torch.nn import init


# 初始化参数
def initNetParams(net):
    """Init net parameters."""
    for m in net.modules():
        # 如果要判断两个类型是否相同推荐使用 isinstance()。
        # isinstance(object, classinfo)
        if isinstance(m, nn.Conv2d):
            # 权重初始化
            m.weight.data.normal(0, 0.001)
            # torch.nn.init.xavier_uniform(m.weight)用一个均匀分布生成值，填充输入的张量或变量。
            # torch.nn.init.constant(tensor, val)用val的值填充输入的张量或变量
            init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            # 用1填充weight
            init.constant(m.weight, 1)
            # 用0填充bias
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # torch.nn.init.kaiming_normal(tensor, a=0, mode='fan_in')用一个正态分布生成值，填充输入的张量或变量。
            # kaiming_normal被弃用，kaiming_normal_取代
            # tensor – n维的torch.Tensor或autograd.Variable
            # a - 这层之后使用的rectifier的斜率系数（ReLU的默认值为0）
            # mode - 可以为“fan_in”（默认）或“fan_out”。“fan_in”保留前向传播时权值方差的量级，“fan_out”保留反向传播时的量级。
            init.kaiming_normal_(m.weight.data)
            # 用0充填bias
            m.bias.data.fill_(0)


# 计算ROC曲线的auc
# for evaluating the performance of the anomaly change detection result
def plot_roc(predict, ground_truth):
    """
    INPUTS:
     predict - 异常变化强度图
     ground_truth - 0or1
    OUTPUTS:
     X, Y for ROC plotting
     auc
    """
    # 寻找ground_truth的最大值
    max_value = np.max(ground_truth)
    '''？？？？'''
    if max_value != 1:
        ground_truth = ground_truth / max_value

    # initial point（1.0, 1.0）起始点
    x = 1.0
    y = 1.0
    hight_g, width_g = ground_truth.shape
    hight_p, width_p = predict.shape
    '''????目的何在'''
    if hight_p != hight_g:
        predict = np.transpose(predict)
    # -1出现在reshape就是不知道维度大小
    ground_truth = ground_truth.reshape(-1)
    predict = predict.reshape(-1)
    # np.where(condition)当where内只有一个参数时，那个参数表示条件，
    # 当条件成立时，where返回的是每个符合condition条件元素的坐标,返回的是以元组的形式
    equals_one1 = np.where(ground_truth == 1)
    # compuate the number of positive and negagtive pixels of the ground_truth
    # 计算ground_truth的正负像素个数
    pos_num = np.sum(ground_truth == 1)
    neg_num = np.sum(ground_truth == 0)
    # step in axis of  X and Y
    # X是负样本,Y是正样本 坐标的步长
    x_step = 1.0 / neg_num
    y_step = 1.0 / pos_num

    # ranking the result map 对异常结果图进行排序，得到的是一个
    index = np.argsort(list(predict))
    ground_truth = ground_truth[index]
    equals_one2 = np.where(ground_truth == 1)
    """ 
    for i in ground_truth:
     when ground_truth[i] = 1, TP minus 1，one y_step in the y axis, go down
     when ground_truth[i] = 0, FP minus 1，one x_step in the x axis, go left
    """
    # np.zeros()用0填充的数组
    X = np.zeros(ground_truth.shape)
    Y = np.zeros(ground_truth.shape)

    '''这个是用来干嘛的'''
    for idx in range(0, hight_g * width_g):
        if ground_truth[idx] == 1:
            y = y - y_step
        else:
            x = x - x_step
        X[idx] = x
        Y[idx] = y
    # np.trapz使用梯形求面积的公式沿给定轴 积分。
    auc = -np.trapz(Y, X)
    '''目的何在？？？？'''
    if auc < 0.5:
        auc = -np.trapz(X, Y)
        t = X
        X = Y
        Y = t

    return X, Y, auc
