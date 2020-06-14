import os
import numpy as np
import torch.utils.data as data

class MusicArrayLoader(data.Dataset):
    def __init__(self, data_path):
        self.datalist = np.load(data_path)
    """
    def get_batch(self, batch_size):
        melody = np.empty((0, self.window_size, 130), dtype=np.int32)
        chord = np.empty((0, self.window_size, 12), dtype=np.int32)
        for file_dir in self.datalist[:batch_size]:
            item = np.load(file_dir)
            melody = np.concatenate((melody, item[:, :130][np.newaxis, :, :]), axis = 0)
            chord = np.concatenate((chord, item[:, 130:][np.newaxis, :, :]), axis = 0)
        self.n_epoch += 1
        return melody, chord
    
    def shuffle_samples(self):
        np.random.shuffle(self.datalist)
    
    def get_n_epoch(self):
        return self.n_epoch
    """
    def __getitem__(self, index):
        file_dir = self.datalist[index]
        item = np.load(file_dir)
        melody = item[:, :130]
        chord = item[:, 130:]
        return melody, chord

    def __len__(self):
        return self.datalist.shape[0]
