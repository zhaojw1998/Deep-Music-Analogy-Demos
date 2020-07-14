import numpy as np
import os
from loader.midi_processing import midi_processer
from tqdm import tqdm

class numpy_processor(object):
    def __init__(self, data_dir):
        self.data = 1#np.load(data_dir)

    def augment_data(self, melody, chord):
        augmented_melody = {}
        augmented_chord = {}
        augmented_melody['0'] = melody
        augmented_chord['0'] = chord
        actual_melody = melody[:, :-2]
        supplement_melody = melody[:, -2:]
        for i in range(1, 7):
            augmented_melody[str(i)] = np.concatenate((actual_melody[:, -i:], actual_melody[:, :-i], supplement_melody), axis=-1)
            augmented_chord[str(i)] = np.concatenate((chord[:, -i:], chord[:, :-i]), axis=-1)
            
        for i in range(-5, 0):
            augmented_melody[str(i)] = np.concatenate((actual_melody[:, abs(i):], actual_melody[:, :abs(i)], supplement_melody), axis=-1)
            augmented_chord[str(i)] = np.concatenate((chord[:, abs(i):], chord[:, :abs(i)]), axis=-1)
        return augmented_melody, augmented_chord

    def split_data(self, data, window_size, hop_size):
        splitted = []
        for i in range(0, data.shape[0]-window_size, hop_size):
            sample = data[i: i+window_size, :]
            splitted.append(sample)
        return splitted

    def train_test_split(self, data_dir, train_ratio):
        samples = os.listdir(data_dir)
        samples = [os.path.join(data_dir, item) for item in samples]
        random.shuffle(samples)
        train_samples = samples[:int(train_ratio*len(samples))]
        val_samples = samples[int(train_ratio*len(samples)):]
        return samples[:5]




if __name__ == '__main__':
    processor = numpy_processor('G:/data.npy')
    data=np.load('G:/data.npy')
    for i in tqdm(range(1480, data[0].shape[0])):
        melody = data[0][i]
        chord = data[1][i]
        print(melody.shape, chord.shape)"""problem remains"""
        augmented_melody, augmented_chord = processor.augment_data(melody, chord)
        for key in augmented_melody:
            melody_to_spilt = augmented_melody[key]
            chord_to_split = augmented_chord[key]
            splitted_melody = processor.split_data(melody_to_spilt, 32, 16)
            splitted_chord = processor.split_data(chord_to_split, 32, 16)
            for idx in range(len(splitted_melody)):
                savedata = np.concatenate((splitted_melody[idx], splitted_chord[idx]), axis=-1)
                if not savedata.shape == (32, 142):
                    print(savedata.shape, str(i) + '(' + key + ')' + '-' + str(idx))
                assert(savedata.shape == (32, 142))
                savename = str(i) + '(' + key + ')' + '-' + str(idx) + 'npy'
                np.save('D:/new_Nottingham_32beat_split/' + savename, savedata)
    