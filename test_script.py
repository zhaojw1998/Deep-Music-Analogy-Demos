import pretty_midi as pyd
# import numpy as np
import os
from tqdm import tqdm

root = 'D:/Download/Program/Musicalion/solo+piano/data'
record = {}
num_malfunction = 0
for midi in tqdm(os.listdir(root)):
    # print(midi)
    try:
        midi_file = pyd.PrettyMIDI(os.path.join(root, midi))
    except ValueError:
        num_malfunction += 1
        continue
    except EOFError:
        num_malfunction += 1
        continue
    except IOError:
        num_malfunction += 1
        continue
    
    num_track = len(midi_file.instruments)
    if num_track not in record:
        record[num_track] = 0
        # print(num_track, midi)
    record[num_track] += 1
print(record)
