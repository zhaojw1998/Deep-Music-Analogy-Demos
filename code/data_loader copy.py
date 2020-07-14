import os
import numpy as np
import torch.utils.data as data

class MusicArrayLoader(data.Dataset):
    def __init__(self, data_path, ratio):
        self.datalist_full = np.load(data_path)
        np.random.shuffle(self.datalist_full)
        self.datalist = self.datalist_full[:int(ratio*len(self.datalist_full))]
        #print(self.datalist[:10])

    def __getitem__(self, index):
        file_dir = self.datalist[index]
        item = np.load(file_dir)
        melody = item[:, :130]
        chord = item[:, 130:]
        return melody, chord

    def __len__(self):
        return self.datalist.shape[0]
    
    def shuffle_data(self):
        np.random.shuffle(self.datalist_full)
        self.datalist = self.datalist_full[:len(self.datalist_full)]

if __name__ == '__main__':
    loader = MusicArrayLoader('datalist/nottingham_32beat_val.npy', 0.2)
