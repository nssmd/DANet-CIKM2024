import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features


class Dataset_parking_data_h(Dataset):
    def __init__(self, flag='train', size=None,
                 features='S', data_path='./data/parking_data_h',
                 target='cars_in_park', scale=True, timeenc=0, freq='h', time_mixed=True,
                 train_only=False, begin_index=0, end_index=0):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.end_index = end_index
        self.begin_index = begin_index

        self.path = data_path
        self.data = None
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.path)

        len1 = self.begin_index
        len4 = self.end_index
        len2 = int(len1 + (len4 - len1) * 0.7)
        len3 = int(len2 + (len4 - len1) * 0.1)
        # pred : val :test=7:1:2

        border1s = [len1, len2 - self.seq_len, len3 - self.seq_len]
        border2s = [len2, len3, len4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # if 0, time normalization else not
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)  # inverse

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.data = data

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_parking_data_h_at(Dataset):
    def __init__(self, flag='train', size=None, random_size=None,
                 data_path='./data/parking_data_h_new',
                 target='cars_in_park', scale=True, freq='h', time_mixed=True, features='S',
                 timeenc=0, train_only=False, begin_index=0, end_index=0, test_index=0):

        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]
        self.random_size = random_size  # 随机矩阵的长度
        self.target = target
        self.scale = scale
        self.freq = freq
        self.end_index = end_index
        self.begin_index = begin_index
        self.test_index = test_index
        self.data_train = None
        self.seq_number = self.test_index - self.begin_index
        self.path = data_path
        self.column = None  # 列数
        self.len = None  # 行数
        self.data_mean = 0
        self.data_std = 0
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_data = pd.read_csv(self.path)

        len1 = self.begin_index
        len3 = self.end_index
        len2 = self.test_index  # 这里的index指的是某几列，即训练集和测试集的列
        # pred  :test=7:3

        border1s = [len1, len2]
        border2s = [len2, len3]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.len = df_data.shape[0]
        self.column = df_data.shape[1]
        if self.scale:
            self.data_mean = df_data.values.mean()
            self.data_std = df_data.values.std()
            data = (df_data.values - self.data_mean) / self.data_std
        else:
            data = df_data.values

        self.data_x = data[:, border1:border2]
        self.data_y = data[:, border1:border2]
        self.data_train = data[:, border1s[0]:border2s[0]]

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end, :].T
        seq_y = self.data_y[r_begin:r_end, :].T

        max_start_row = self.len - self.random_size  # random_size:随机选取的矩阵长度
        start_row = np.random.randint(0, max_start_row + 1)
        stacked_array = self.data_train[start_row:start_row + self.random_size, :]

        random_matrix = torch.from_numpy(stacked_array)

        return seq_x, seq_y, random_matrix

    def __len__(self):
        return self.len - self.pred_len - self.seq_len + 1

    def extract_and_stft(self, tensor1, predlen):
        max_int_multiple_length = (len(tensor1) // predlen) * predlen
        tensor1 = tensor1[:max_int_multiple_length]

        # Set the window size for STFT to predlen as intended
        n_fft = predlen  # n_fft should be the same as predlen
        hop_length = n_fft  # No overlap, just sliding window
        win_length = n_fft  # The window length is set to predlen

        # Perform STFT
        window = torch.hann_window(n_fft, device=tensor1.device)
        stft_result = torch.stft(tensor1.clone().detach(), n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                 window=window, return_complex=True)

        # Get the magnitude of the STFT result
        magnitude = torch.abs(stft_result)

        # Sum over the time dimension (time_frames) to aggregate frequency information
        summed_magnitude = torch.sum(magnitude, dim=-1)  # Summing over the time dimension

        # Remove the first value (zero frequency component)
        summed_magnitude = summed_magnitude[1:]

        return summed_magnitude

    def fft(self):
        temp_tensors = []
        for i in range(0, self.seq_number):
            temp_tensor = self.extract_and_stft(self.get_data_range(i), self.seq_len)
            temp_tensors.append(temp_tensor)
        stacked_tensor = torch.stack(temp_tensors, dim=0)  # stack_tensor:所有傅里叶变换后的张量
        return stacked_tensor

    def get_data_range(self, i):
        return torch.tensor(self.data_train[:, i], dtype=torch.float)

    def inverse_transform(self, data):
        return data*self.data_std+self.data_mean