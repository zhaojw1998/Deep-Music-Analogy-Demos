from dataloader import MIDI_Loader, MIDI_Render
from chordloader import Chord_Loader
import numpy as np
import pickle
import os
from tqdm import tqdm
"""
loader = MIDI_Loader(datasetName = 'Nottingham', minStep = 0.125)
directory = 'nottingham_midi_dual-track/'
midi_files = loader.load(directory)
#midi_files = loader.getNoteSeq()z
midi_files = loader.getMelodyAndChordSeq(recogLevel = "Seven")
midi_files = loader.dataAugment()
_ = loader.writeFile(output = "nottingham_txt/")
"""
#render = MIDI_Render(datasetName = 'Nottingham', minStep = 0.125)
#midi = render.text2midi(text_ad="nottingham_txt/A and D-switch(-2).txt", recogLevel = "Seven", output = "test.mid")
window_size = 32
hop_size = 16
roll_size = 130
chord_size = 12
NC = 132
txt_root = "nottingham_txt"
cl = Chord_Loader(recogLevel = "Seven")
save_root = 'D:/nottingham_32beat-split_npy'
for txt in tqdm(os.listdir(txt_root)):
    txt_path = os.path.join(txt_root, txt)
    first_start = 0
    with open(txt_path, 'r') as f:
        txt_file=f.readlines()
    chord = [int(item) for item in txt_file[2].strip("'").strip(' \n').split(' ')]
    note = [int(item) for item in txt_file[4].strip("'").strip(' \n').split(' ')]
    if not len(chord) == len(note):
        print(txt_file, 'chord:', len(chord), 'note:', len(note))
    assert(len(chord) == len(note))
    while chord[first_start] == NC:
        first_start += 1
    for idx, start in enumerate(range(first_start, len(chord)-window_size, hop_size)):
        note_sample = note[start: start + window_size]
        chord_sample = chord[start: start + window_size]
        note_matrix = np.zeros((window_size, roll_size), dtype = np.int32)
        for t, roll_index in enumerate(note_sample):
            note_matrix[t][roll_index] = 1
        chord_matrix = np.zeros((window_size, chord_size), dtype = np.int32)
        for t, chord_index in enumerate(chord_sample):
            chord_name = cl.index2name(chord_index)
            chord_notes = cl.name2note(chord_name)
            if chord_notes is None:
                continue
            for roll_index in chord_notes:
                chord_matrix[t][roll_index % 12] = 1
        matrix = np.concatenate((note_matrix, chord_matrix), axis=1)
        save_name = txt.strip('.txt') + '_' + str(idx) + '_.npy'
        np.save(os.path.join(save_root, save_name), matrix)