import pickle
import os

with open(os.path.join('data_save_root', 'batchData_shifted.npy'), 'rb') as f:
    raw_midi_set = pickle.load(f)
    tempo_set = pickle.load(f)
    minStep_set = pickle.load(f)
    start_record = pickle.load(f)
    belonging = pickle.load(f)
    num_beforeShift = pickle.load(f)
print(num_beforeShift)