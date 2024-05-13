from torch.utils.data import Dataset
import numpy as np
# 主体类方法部分
class my_dataset(Dataset):
      def __init__(self, x, y):  # 这里的time_scale指的是在训练、测试情况下，总数除以每组变量数
            super(my_dataset, self).__init__()
            self.data_x = x
            self.data_y = y

      def __getitem__(self, index):
            target_ssh = self.data_y[index]
            sample = self.data_x[index]
            return sample, target_ssh

      def __len__(self):
            return len(self.data_y)

      def predict_ssh(self, time2):
            x = self.data_x[time2]
            y = self.data_y[time2]
            return np.expand_dims(x, 0), np.expand_dims(y, 0)

      def predict_ssh_area(self, time_l, time_r):  # 输入想要预测的时间片段索引
            y = self.data_y[time_l:time_r + 1]
            x = self.data_x[time_l:time_r + 1]
            m = time_r - time_l + 1
            return x, y, m