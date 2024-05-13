import numpy as np
from load_data import *
import torch
from MLPmixer_model import *

saved_model = torch.load(r"E:\py\pytorch-py39\MLPmixer\models_saved\MLPM2_small_6months_d2_move0.pth")

'''下面是12个月sla u v预测六个月sla的数据，经度150-270，nan很少'''
path_x_train=r'E:\chuangxun\data_devided\data_x_train_oras_for_Mm_small_6months.npy'
path_y_train=r'E:\chuangxun\data_devided\data_y_train_oras_for_Mm_small_6months.npy'
path_x_test=r'E:\chuangxun\data_devided\data_x_test_oras_for_Mm_small_6months.npy'
path_y_test=r'E:\chuangxun\data_devided\data_y_test_oras_for_Mm_small_6months.npy'

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

#预测3个月时 time=0时 96年12月
#time = 61      # 预测时间点03年1 2 3月

#预测6个月时 time=0时 97年9月
time = 52      # 预测时间点03年1 2 3 4 5 6月

predict_data_x, truth_data_y = test_data.predict_ssh(time)
predict_data_y = saved_model(torch.tensor(predict_data_x).type(torch.cuda.FloatTensor))
predict_data_y = predict_data_y.cpu().detach().numpy()
#print(predict_data_y.shape, truth_data_y.shape)#(1, 3, 25, 180)

#np.save(r'E:\py\pytorch-py39\MLPmixer\data_saved\6months_big/y_03_big_6months_MLPM2_d2_move0.npy', truth_data_y)
np.save(r'E:\py\pytorch-py39\MLPmixer\data_saved\6months_big/x_03_small_6months_MLPM2_d2_move0.npy', predict_data_y)

