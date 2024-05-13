from MLPmixer_2 import *
from load_data import *
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''下面是12个月sla u v预测六个月sla的数据，经度150-270，nan很少'''
path_x_train=r'E:\chuangxun\data_devided\data_x_train_oras_for_Mm_small_6months.npy'
path_y_train=r'E:\chuangxun\data_devided\data_y_train_oras_for_Mm_small_6months.npy'
path_x_test=r'E:\chuangxun\data_devided\data_x_test_oras_for_Mm_small_6months.npy'
path_y_test=r'E:\chuangxun\data_devided\data_y_test_oras_for_Mm_small_6months.npy'

'''下面是12个月sla u v预测1个月sla u v的数据，经度150-270,nan很少,用来滑动预测'''
# path_x_train=r'E:\chuangxun\data_devided\data_x_train_oras_for_Mm_small.npy'
# path_y_train=r'E:\chuangxun\data_devided\data_y_train_oras_for_Mm_small.npy'
# path_x_test=r'E:\chuangxun\data_devided\data_x_test_oras_for_Mm_small.npy'
# path_y_test=r'E:\chuangxun\data_devided\data_y_test_oras_for_Mm_small.npy'

'''下面是12个月sla u v预测六个月sla的数据，经度130-280，nan很多'''
# path_x_train=r'E:\chuangxun\data_devided\data_x_train_oras_for_Mm_6months.npy'
# path_y_train=r'E:\chuangxun\data_devided\data_y_train_oras_for_Mm_6months.npy'
# path_x_test=r'E:\chuangxun\data_devided\data_x_test_oras_for_Mm_6months.npy'
# path_y_test=r'E:\chuangxun\data_devided\data_y_test_oras_for_Mm_6months.npy'

x_train=np.load(path_x_train)
y_train=np.load(path_y_train)
x_test=np.load(path_x_test)
y_test=np.load(path_y_test)


train_data = my_dataset(x_train, y_train)
test_data = my_dataset(x_test, y_test)

train_data_size = len(train_data)
test_data_size = len(test_data)

print(test_data_size, train_data_size)

# 添加tensorboard
writer = SummaryWriter("logs_train/")

model=MLPM(lat=30, lon=120, in_C=36, out_C=6, depth=3)

model.to(device)
train_dataloader = DataLoader(train_data, batch_size=6)
test_dataloader = DataLoader(test_data, batch_size=6)
#损失函数
loss_fn = nn.MSELoss().cuda()
# 优化器
learning_rate = 0.004
optimizer = torch.optim.Adam(model.parameters(), lr=0.0006)


# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 30

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    model.train()
    for data in train_dataloader:
        samples, targets = data
        samples = samples.type(torch.cuda.FloatTensor)
        targets = targets.type(torch.cuda.FloatTensor)
        outputs = model(samples)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 6 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            samples, targets = data
            samples = samples.type(torch.cuda.FloatTensor)
            targets = targets.type(torch.cuda.FloatTensor)
            outputs = model(samples)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            # accuracy = (outputs.argmax(1) == targets).sum()
            # total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
   # print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
   # writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1

outputs = outputs.cpu()
targets = targets.cpu()
writer.close()
torch.save(model, r"E:\py\pytorch-py39\MLPmixer\models_saved\MLPM2_small_6months_d2_move0.pth")
print("模型已保存")