from loader.midi_processing import midi_processor
import numpy as np
import os
import pretty_midi as pyd
import torch
import json
from model import VAE


def numpy_to_midi_with_condition(sample, condition, time_step = 0.125, output='sample/sample.mid'):
    music = pyd.PrettyMIDI()
    piano_program = pyd.instrument_name_to_program(
        'Acoustic Grand Piano')
    chord_program = pyd.instrument_name_to_program(
        'Acoustic Grand Piano')
    piano = pyd.Instrument(program=piano_program)
    chord = pyd.Instrument(program=chord_program)
    t = 0
    for i in sample:
        pitch = int(np.argmax(i))
        if pitch < 128:
            note = pyd.Note(velocity=100, pitch=pitch, start=t, end=t + time_step)
            t += time_step
            piano.notes.append(note)
        elif pitch == 128:
            if len(piano.notes) > 0:
                note = piano.notes.pop()
            else:
                p = np.random.randint(60, 72)
                note = pyd.Note(velocity=100, pitch=int(p), start=0, end=t)
            note = pyd.Note(
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
                note = pyd.Note(velocity=100, pitch=idx+4*12, start=start, end=end)
                chord.notes.append(note)
        start = end
        end += time_step
    music.instruments.append(piano)
    music.instruments.append(chord)
    music.write(output)

processor = midi_processor("Nottingham", 0.125)
file_dir = 'D:/Download/Program/right.mid'

file = processor.load_single(file_dir)
file = processor.getMelodyAndChordSeq_single("Seven")
melody, chord = processor.midi2Numpy_single()

splitted_melody = np.empty((0, 32, 130))
splitted_chord = np.empty((0, 32, 12))
for i in range(0, melody.shape[1], 32):
    splitted_melody = np.concatenate((splitted_melody, melody[:, i: i+32, :]), axis=0)
    splitted_chord = np.concatenate((splitted_chord, chord[:, i: i+32, :]), axis=0)

weight_path = 'params/Mon Jul  6 22-51-32 2020/best_fitted_params.pt'
with open('code/model_config.json') as f:
    args = json.load(f)
model = VAE(130, args['hidden_dim'], 3, 12, args['pitch_dim'], args['rhythm_dim'], args['time_step']).cuda()
model.load_state_dict(torch.load(weight_path)['model_state_dict'])
model.eval()

melody = torch.from_numpy(splitted_melody).cuda().float()
chord = torch.from_numpy(splitted_chord).cuda().float()

distr_pitch, distr_rhythm = model.encoder(melody, chord)
z_pitch = distr_pitch.mean
z_rhythm = distr_rhythm.mean

recon_rhythm = model.rhythm_decoder(z_rhythm)

recon = model.final_decoder(z_pitch, recon_rhythm, chord)

recon = recon.detach().cpu()
print(recon.shape)
recon = recon.reshape((-1, 130))
print(recon.shape)

idx = recon.max(1)[1]
out = torch.zeros_like(recon)
arange = torch.arange(out.size(0)).long()
out[arange, idx] = 1

savename = 'little_star.mid'
numpy_to_midi_with_condition(out.numpy(), chord.detach().cpu().numpy()[0], time_step = 0.125, output='./'+ savename)