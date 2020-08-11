import numpy as np
import os
import random
import pretty_midi as pyd
from tqdm import tqdm

root = 'D:\Download\Program\Musicalion\solo+piano\data_temoNormalized_trackDoublized'
save_root = 'for_zehao'
if not os.path.exists(save_root):
    os.makedirs(save_root)
midis = os.listdir(root)
random.shuffle(midis)
for mid in midis[:10]:
    midi_file = pyd.PrettyMIDI(os.path.join(root, mid))
    tempo = midi_file.get_tempo_changes()[-1][0]
    midiRecon = pyd.PrettyMIDI(initial_tempo=tempo)
    midiRecon.instruments.append(midi_file.instruments[0])
    midiRecon.write(os.path.join(save_root, mid))