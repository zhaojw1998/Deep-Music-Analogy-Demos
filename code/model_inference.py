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
            
weight_path = 'params/cdvae_2bar_overfitting.pt'
with open('code/model_config.json') as f:
    args = json.load(f)
model = VAE(130, args['hidden_dim'], 3, 12, args['pitch_dim'], args['rhythm_dim'], args['time_step']).cuda()
model.load_state_dict(torch.load(weight_path))
model.eval()

data_root = 'D:/nottingham_32beat-split_npy'
data_list = []
for txt_file in os.listdir(data_root):
    data_list.append(os.path.join(data_root, txt_file))
Melody_A = np.load(np.random.choice(data_list))
Melody_B = np.load(np.random.choice(data_list))
numpy_to_midi_with_condition(Melody_A[:, :130], Melody_A[:, 130:], time_step = 0.125, output='test_A.mid')
numpy_to_midi_with_condition(Melody_B[:, :130], Melody_B[:, 130:], time_step = 0.125, output='test_B.mid')

Melody_A = torch.from_numpy(Melody_A).cuda().float()
Melody_B = torch.from_numpy(Melody_B).cuda().float()
melody_A = Melody_A[:, :130][np.newaxis, :, :]
chord_A = Melody_A[:, 130:][np.newaxis, :, :]
melody_B = Melody_B[:, :130][np.newaxis, :, :]
chord_B = Melody_B[:, 130:][np.newaxis, :, :]

distr_A_pitch, distr_A_rhythm = model.encoder(melody_A, chord_A)
distr_B_pitch, distr_B_rhythm = model.encoder(melody_B, chord_B)
z_A_pitch = distr_A_pitch.rsample()
z_B_rhythm = distr_B_rhythm.rsample()
recon_rhythm = model.rhythm_decoder(z_B_rhythm)
condition = Melody_A[:, 130:][np.newaxis, :, :]
recon = model.final_decoder(z_A_pitch, recon_rhythm, chord_A)
recon = recon.detach().cpu()[0]
idx = recon.max(1)[1]
out = torch.zeros_like(recon)
arange = torch.arange(out.size(0)).long()
out[arange, idx] = 1
#print(out.numpy())
numpy_to_midi_with_condition(out.numpy(), chord_A.detach().cpu().numpy()[0], time_step = 0.125, output='test_analogy.mid')