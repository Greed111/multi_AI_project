import numpy as np
from load_data import *
import torch
from MLPmixer_model import *
from torch.utils.tensorboard import SummaryWriter

saved_model = torch.load(r"E:\py\pytorch-py39\MLPmixer\models_saved\MLPM_small_hdyc_d2_move0.pth")

'''下面是12个月sla u v预测六个月sla的数据，经度150-270，nan很少'''
# path_x_train=r'E:\chuangxun\data_devided\data_x_train_oras_for_Mm_small_6months.npy'
# path_y_train=r'E:\chuangxun\data_devided\data_y_train_oras_for_Mm_small_6months.npy'
# path_x_test=r'E:\chuangxun\data_devided\data_x_test_oras_for_Mm_small_6months.npy'
# path_y_test=r'E:\chuangxun\data_devided\data_y_test_oras_for_Mm_small_6months.npy'

'''下面是12个月sla u v预测1个月sla u v的数据，经度150-270,nan很少,用来滑动预测'''
path_x_train=r'E:\chuangxun\data_devided\data_x_train_oras_for_Mm_small.npy'
path_y_train=r'E:\chuangxun\data_devided\data_y_train_oras_for_Mm_small.npy'
path_x_test=r'E:\chuangxun\data_devided\data_x_test_oras_for_Mm_small.npy'
path_y_test=r'E:\chuangxun\data_devided\data_y_test_oras_for_Mm_small.npy'

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

x_ori = np.zeros((12, 36, 30, 120), dtype=np.float32)
x_now = np.zeros((12, 36, 30, 120), dtype=np.float32)
y_num = np.zeros((12, 3, 30, 120), dtype=np.float32)

time_test = 48     # 预测时间点03年1-12

writer = SummaryWriter("logs_MOVE")
# loss_func = nn.MSELoss()

# predict_data_x, truth_data_y = train_data.predict_ssh(470)
predict_data_x, truth_data_y = test_data.predict_ssh(time_test)

x_ori[0, :, :, :] = predict_data_x
predict_data_x = saved_model(torch.tensor(predict_data_x, device='cuda'))
predict_data_x = predict_data_x.cpu().detach().numpy()
# truth_data_y = np.squeeze(truth_data_y)
y_num[0, :, :, :] = predict_data_x

for i in [-1, -2, -3]:
      x_now[0, i, :, :] = y_num[0, i, :, :]
for i in range(33):
      x_now[0, i, :, :] = x_ori[0, i+3, :, :]

for j in range(1, 12):
      x_ori[j, :, :, :] = x_now[j-1, :, :, :]
      predict_data_x = x_ori[j, :, :, :].reshape(1, 36, 30, 120)
      predict_data_x = saved_model(torch.tensor(predict_data_x, device='cuda'))
      predict_data_x = predict_data_x.cpu().detach().numpy()
      y_num[j, :, :, :] = np.squeeze(predict_data_x)
      for i in [-1, -2, -3]:
            x_now[j, i, :, :] = y_num[j, i, :, :]
      for i in range(33):
            x_now[j, i, :, :] = x_ori[j, i + 3, :, :]

      # 预测时间段,注意x和y形状


predict_data_x, truth_data_y, piece = test_data.predict_ssh_area(time_test, time_test+11)
# predict_data_x, truth_data_y, piece = train_data.predict_ssh_area(470, 470+11)
#predict_data_x, truth_data_y, piece = val_data.predict_ssh_area(time_val, time_val+11)

print(truth_data_y.shape, y_num.shape)

np.save(r'E:\py\pytorch-py39\MLPmixer\data_saved\12months_hdyc/y_03_hdyc_MLPM_d2_move0.npy', truth_data_y)
np.save(r'E:\py\pytorch-py39\MLPmixer\data_saved\12months_hdyc/x_03_hdyc_MLPM_d2_move0.npy', y_num)

