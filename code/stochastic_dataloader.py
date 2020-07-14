import pretty_midi
import torch
import os
import json
import numpy as np
from model import VAE

def numpy_to_midi_with_condition(sample, condition, time_step = 0.125, output='sample/sample.mid'):
    music = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program(
        'Acoustic Grand Piano')
    chord_program = pretty_midi.instrument_name_to_program(
        'Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    chord = pretty_midi.Instrument(program=chord_program)
    t = 0
    for i in sample:
        pitch = int(np.argmax(i))
        if pitch < 128:
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=t, end=t + time_step)
            t += time_step
            piano.notes.append(note)
        elif pitch == 128:
            if len(piano.notes) > 0:
                note = piano.notes.pop()
            else:
                p = np.random.randint(60, 72)
                note = pretty_midi.Note(velocity=100, pitch=int(p), start=0, end=t)
            note = pretty_midi.Note(
                velocity=100,
                pitch=note.pitch,
                start=note.start,
                end=note.end + time_step)
            piano.notes.append(note)
            t += time_step
        elif pitch == 129:
            t += time_step
    start = 0
    end = time_step
    for index, i in enumerate(condition):
        if index < condition.shape[0]-1 and np.sum(np.abs(i-condition[index+1]))==0:
            end += time_step
            continue
        if np.sum(i) == 0:
            start = end
            end += time_step
            continue
        for idx in range(i.shape[0]):
            if i[idx] == 1:
                note = pretty_midi.Note(velocity=100, pitch=idx+4*12, start=start, end=end)
                chord.notes.append(note)
        start = end
        end += time_step
    music.instruments.append(piano)
    music.instruments.append(chord)
    music.write(output)

data = np.load('G:/data.npy')
sample_melody = data[0][1481]
sample_chord = data[1][1481]
length = min(sample_melody.shape[0], sample_chord.shape[0])
sample_melody = sample_melody[:length, :]
sample_chord = sample_chord[:length, :]
numpy_to_midi_with_condition(sample_melody, sample_chord, time_step = 0.125, output='debug.mid')
#print(sample_mdelody.shape, sample_chord.shape)
#numpy_to_midi_with_condition(sample_mdelody, sample_chord, time_step = 0.125, output='sample_reconstruct.mid')