import pretty_midi as pyd
import numpy as np
import os
from chordloader import Chord_Loader
import copy
from tqdm import tqdm
import sys

class midi_processor(object):
    def __int__(self):
        pass
    
    def dual_track_batch(self, raw_midi_root, dual_track_midi_root):
        if not os.path.exists(dual_track_midi_root):
            os.makedirs(dual_track_midi_root)
        mal_function_record = 0
        for mid in tqdm(os.listdir(raw_midi_root)):
            midi = os.path.join(raw_midi_root, mid)
            try:
                midi_data = pyd.PrettyMIDI(midi)
            except:
                mal_function_record += 1
                continue
            tempo = np.mean(midi_data.get_tempo_changes()[-1])
            if len(midi_data.instruments) == 1:
                midi_data.write(os.path.join(dual_track_midi_root, mid))
                continue    #omit mono_track samples
            melody = midi_data.instruments[0]
            new_track = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
            for i in range(1, len(midi_data.instruments)):
                track = midi_data.instruments[i]
                for note in track.notes:
                    new_track.notes.append(note)
            gen_midi = pyd.PrettyMIDI(initial_tempo=tempo)
            gen_midi.instruments.append(melody)
            gen_midi.instruments.append(new_track)
            gen_midi.write(os.path.join(dual_track_midi_root, mid))
        print('All done, with unsuccesful samples:', mal_function_record)
    
    def dual_track_single(self, raw_midi_file, dual_track_midi_file='transferred2dual_track.mid'):
        midi_data = pyd.PrettyMIDI(raw_midi_file)
        tempo = np.mean(midi_data.get_tempo_changes()[-1])
        if len(midi_data.instruments) == 1:
            print('this is mono-track')
            sys.exit()
        melody = midi_data.instruments[0]
        new_track = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
        for i in range(1, len(midi_data.instruments)):
            track = midi_data.instruments[i]
            for note in track.notes:
                new_track.notes.append(note)
        gen_midi = pyd.PrettyMIDI(initial_tempo=tempo)
        gen_midi.instruments.append(melody)
        gen_midi.instruments.append(new_track)
        gen_midi.write(dual_track_midi_file)

if __name__ == '__main__':
    processor = midi_processor()
    processor.dual_track_batch('D:/Download/Program/Musicalion/solo+piano/data', 'D:/Download/Program/Musicalion/solo+piano/data2dual_track')