# coding=utf-8
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset


# 训练预测
def train_predict(x_3d, y_3d, model, un_idx_train, un_idx_valid,
                  epoch, batch_sz, learn_rate, w_decay, use_gpu):
    # 1、准备数据
    # 3维数据
    img_channel, img_height, img_width = x_3d.shape
    # np.transpose是改变矩阵的索引值，二维的就相当于矩阵的转置；多维的，其第二个参数就是改变之后的索引值，正常索引（x,y,z）=(0,1,2)
    x_2d = np.transpose(x_3d.reshape(img_channel, img_height * img_width))  # reshape成[num,band]2维
    y_2d = np.transpose(y_3d.reshape(img_channel, img_height * img_width))  # reshape成[num,band]2维
    # torch.utils.data.TensorDataset(data_tensor包含样本数据, target_tensor包含样本目标)
    # 包装数据和目标张量的数据集。通过沿着第一个维度索引两个张量来恢复每个样本。
    # torch.Tensor是一种包含单一数据类型元素的多维矩阵。
    train_dataset = TensorDataset(torch.tensor(x_2d[un_idx_train, :], dtype=torch.float32),
                                  torch.tensor(y_2d[un_idx_train, :], dtype=torch.float32))
    valid_label_x = torch.tensor(x_2d[un_idx_valid, :], dtype=torch.float32)
    valid_label_y = torch.tensor(y_2d[un_idx_valid, :], dtype=torch.float32)

    # torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0,
    #                             collate_fn=<function default_collate>, pin_memory=False, drop_last=False)
    # 数据加载器。组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。
    '''
    train_dataset:加载数据的数据集
    batch_size：每个batch加载多少个样本，默认1
    shuffle：设置为true时会在每个epoch重新打乱数据
    '''
    data_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
    # 迭代iteration次数=训练集具有的batch个数=训练集总数/batchsize
    iter_num = un_idx_train.size // batch_sz
    # 2、准备网络模型
    # 3、准备损失函数和优化器
    # 损失函数 MSELoss创建一个衡量输入x(模型预测输出)和目标y之间均方误差标准。
    # 对n个元素对应的差值的绝对值求和，得出来的结果除以n。
    loss_fc = nn.MSELoss()
    # 在构建网络时，将模型和损失函数传输到 GPU（pytorch）
    if use_gpu:
        model = model.cuda()
        loss_fc = loss_fc.cuda()
    # torch.optim是一个实现了各种优化算法的库
    # 首先要构建一个optimizer，可以为每一个参数单独设置选项，
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, betas=(0.9, 0.99), weight_decay=w_decay)
    # 4、训练
    # Training loss  &  Valid loss
    Tra_ls, Val_ls = [], []
    '''轮次内部epoch：Dataloader遍历训练、save模型(间隔)、eval模型（间隔）'''
    for _epoch in range(0, epoch):
        # 训练模型
        model.train()

        tra_ave_ls = 0
        # 训练模型参数
        for i, data in enumerate(data_loader):
            train_x, train_y = data
            # While traning, transfer the data to GPU
            if use_gpu:
                train_x, train_y = train_x.cuda(), train_y.cuda()
            # predict_y = model(Variable(train_x))

            # 梯度清零 zero_grad()清空所有被优化过的Variable的梯度.
            optimizer.zero_grad()

            '''正向传播求损失'''
            loss = loss_fc(model(Variable(train_x)), train_y)

            '''反向传播求梯度 '''
            # 一旦梯度被如backward()之类的函数计算好后，我们就可以调用step()
            loss.backward()
            # 进行单次优化（参数更新），step()会更新所有的参数，
            optimizer.step()

            tra_ave_ls += loss.item()
            tra_ave_ls /= iter_num
        Tra_ls.append(tra_ave_ls)
        model.eval()
        if use_gpu:
            valid_label_x, valid_label_y = valid_label_x.cuda(), valid_label_y.cuda()
        val_ls = loss_fc(model(valid_label_x), valid_label_y).item()
        Val_ls.append(val_ls)
    #     打印
    # print('epoch [{}/{}],train:{:.4f},  valid:{:.4f}'.
    #     format(_epoch + 1, epoch, tra_ave_ls, val_ls))
    # # if _epoch % 5 == 0 :  print('epoch [{}/{}],train:{:.4f},  valid:{:.4f}'.
    #                           format(_epoch + 1, epoch, tra_ave_ls,val_ls))
    '''验证循环'''
    # Prediction
    model.eval()
    x_2d = torch.tensor(x_2d, dtype=torch.float32)
    if use_gpu:
        x_2d = x_2d.cuda()
    prediction_y = model(x_2d)  # [num, band]
    loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
    if use_gpu:
        loss_fn = loss_fn.cuda()
    input_y = torch.autograd.Variable(torch.from_numpy(y_2d)).float()  # [num, band]
    if use_gpu:
        input_y = input_y.cuda()
    loss = loss_fn(input_y, prediction_y)
    if use_gpu:
        loss, prediction_y = loss.cpu(), prediction_y.cpu()
    loss_m1 = np.sum(loss.detach().numpy(), axis=1).reshape(img_height, img_width)  # axis=1,[num, 1]
    prediction_y = prediction_y.detach().numpy().transpose(). \
        reshape([img_channel, img_height, img_width, ])

    return model, loss_m1, prediction_y, Tra_ls, Val_ls
