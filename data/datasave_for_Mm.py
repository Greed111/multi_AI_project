import numpy as np
from data_pre import *
import numpy as np
import xarray as xr

# 数组的坐标是纬度从负到正，经度从小到大

data_address_u = xr.open_dataset(r'E:\chuangxun\newdata\ORAS3/ORAS3taux.nc') # 58年1月到08年12月
data_address_v = xr.open_dataset(r'E:\chuangxun\newdata\ORAS3/ORAS3tauy.nc')
data_address_ssh = xr.open_dataset(r'E:\chuangxun\newdata\ORAS3/ORAS3SSH.nc')

# 读取数据，xarray中的矩阵
windu = data_address_u['TAUX'].values
windv = data_address_v['TAUY'].values
sealevel = data_address_ssh['SSH'].values

windu=windu[:,-30: ,-150:]
windv=windv[:,-30: ,-150:]
sealevel=sealevel[:,-30: ,-150:]

data_x_train, data_y_train, data_x_test, data_y_test =data_divide([sealevel,windu,windv], x_width=12, y_width=6, stride=1)

print(data_x_train.shape)
print(data_y_train.shape)
print(data_x_test.shape)
print(data_y_test.shape)

'''12个月的sla和风场预测接下来一个月的sla和风场 36→3小范围'''  #'E:\chuangxun\data_devided\data_x_train_oras_for_Mm_small.npy'
x_train=data_x_train.transpose(1,2,0,3,4).reshape(480, 36, 30, 120)
y_train=np.squeeze(data_y_train.transpose(1,0,2,3,4))
x_test=data_x_test.transpose(1,2,0,3,4).reshape(120, 36, 30, 120) #这里的测试集第一个数据的第一个月是 98年1月  #time_test = 60     # 预测时间点03年1-12
y_test=np.squeeze(data_y_test.transpose(1,0,2,3,4))

#'''12个月sla u v 预测接下来六个月sla 36→6,大范围'''    #'E:\chuangxun\data_devided\data_x_train_oras_for_Mm_6months.npy'
# x_train=data_x_train.transpose(1,0,2,3,4).reshape(476, 36, 30, 150)
# y_train=data_y_train[0]
# x_test=data_x_test.transpose(1,0,2,3,4).reshape(119, 36, 30, 150) #这里的测试集第一个数据的第一个月是 96年10月  #预测6个月时 time=0时 96年10月# time = 63      # 预测时间点03年1 2 3月
# y_test=data_y_test[0]

#'''12个月sla u v 预测接下来六个月sla 36→6，小范围'''   #'E:\chuangxun\data_devided\data_x_train_oras_for_Mm_small_6months.npy'
# x_train=data_x_train.transpose(1,0,2,3,4).reshape(476, 36, 30, 120)
# y_train=data_y_train[0]
# x_test=data_x_test.transpose(1,0,2,3,4).reshape(119, 36, 30, 120) #这里的测试集第一个数据的第一个月是 96年10月
# y_test=data_y_test[0]

#'''12个月sla u v 预测接下来三个月的sla 36→3小范围'''
# x_train=data_x_train.transpose(1,0,2,3,4).reshape(478, 36, 30, 120)
# y_train=data_y_train[0]
# x_test=data_x_test.transpose(1,0,2,3,4).reshape(120, 36, 30, 120) #这里的测试集第一个数据的第一个月是 96年12月
# y_test=data_y_test[0]

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

np.save(r'E:\chuangxun\data_devided\data_x_train_oras_for_Mm_small_6months.npy', mask(x_train))
np.save(r'E:\chuangxun\data_devided\data_y_train_oras_for_Mm_small_6months.npy', mask(y_train))
np.save(r'E:\chuangxun\data_devided\data_x_test_oras_for_Mm_big_6months.npy', mask(x_test))
np.save(r'E:\chuangxun\data_devided\data_y_test_oras_for_Mm_big_6months.npy', mask(y_test))