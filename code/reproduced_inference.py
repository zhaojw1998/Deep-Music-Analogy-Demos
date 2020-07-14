from loader.dataloader import MIDI_Loader, MIDI_Render
from loader.chordloader import Chord_Loader
import numpy as np
import os
import pretty_midi as pyd
import torch
import json
from model import VAE

class two_bar_dataloader(MIDI_Loader):
    def __init__(self, datasetName, minStep = 0.03125):
        super(two_bar_dataloader, self).__init__(datasetName, minStep)
    
    def load(self, file_path):
        if self.datasetName == "Nottingham":
            midi_data = pyd.PrettyMIDI(file_path)
            #assert(len(midi_data.instruments)) == 2
            self.midi_file = {"name": (file_path.split("."))[0].split('/')[-1], "raw": midi_data}
            print("loading file success!")
            return self.midi_file
        print("Error: No dataset called " +  self.datasetName)
        return None
    
    def getMelodyAndChordSeq(self, recogLevel = "Mm"):
        print("start to get melody and sequences")
        self.recogLevel = recogLevel
        if self.datasetName == "Nottingham":
            # 25 one hot vectors
            # 0-11 for major
            # 12-23 for minor 
            # 24 for NC
            midi_data = self.midi_file["raw"]
            self.cl = Chord_Loader(recogLevel = self.recogLevel)
            chord_set = []
            chord_time = [0.0, 0.0]
            last_time = midi_data.instruments[0].notes[0].start # Define the chord recognition system
            chord_file = []
            pitch_file = []
            rest_pitch = 129
            hold_pitch = 128
            offset = 0
            #initialize save list for melody and chord notes
            self.midi_file["chords"] = []
            self.midi_file["pitches"] = []
            #for chord
            if len(midi_data.instruments) == 1:
                start_time = midi_data.instruments[0].notes[0].start
                end_time = midi_data.instruments[0].notes[-1].end
                self.midi_file["chords"].append({"start":start_time,"end": end_time, "chord" : "NC"})
            else:
                if not len(midi_data.instruments) == 2:
                    print(self.midi_file['name'])
                    assert(len(midi_data.instruments) == 2)
                for note in midi_data.instruments[1].notes:
                    if len(chord_set) == 0:
                        chord_set.append(note.pitch)
                        chord_time[0] = note.start
                        chord_time[1] = note.end
                        offset = chord_time[0] - last_time
                    else:
                        if note.start == chord_time[0] and note.end == chord_time[1]:
                            chord_set.append(note.pitch)
                        else:
                            if last_time < chord_time[0]:
                                self.midi_file["chords"].append({"start":last_time,"end": chord_time[0], "chord" : "NC"})
                            self.midi_file["chords"].append({"start":chord_time[0],"end":chord_time[1],"chord": self.cl.note2name(chord_set)})
                            last_time = chord_time[1]
                            chord_set = []
                            chord_set.append(note.pitch)
                            chord_time[0] = note.start
                            chord_time[1] = note.end 
                if chord_set:
                    if last_time < chord_time[0]:
                        self.midi_file["chords"].append({"start":last_time ,"end": chord_time[0], "chord" : "NC"})
                    self.midi_file["chords"].append({"start":chord_time[0],"end":chord_time[1],"chord": self.cl.note2name(chord_set)})
                    last_time = chord_time[1]
            #for melody and chord
            anchor = 0
            note = midi_data.instruments[0].notes[anchor]
            new_note = True
            for idx, c in enumerate(self.midi_file["chords"]):
                if c["end"] - c["start"] < self.minStep:
                    continue
                c1 = self.midi_file["chords"][min(idx+1, len(self.midi_file["chords"])-1)]
                if c1["end"] - c1["start"] < self.minStep:
                    steps = int(round((c1["end"] - c["start"]) / self.minStep))
                else:
                    steps = int(round((c["end"] - c["start"]) / self.minStep))
                c_index = self.cl.name2index(c["chord"])
                time_step = c["start"]
                for j in range(steps):
                    chord_file.append(c_index)
                    while note.end < time_step:
                        anchor += 1
                        note = midi_data.instruments[0].notes[anchor]
                        new_note = True
                    if note.start > time_step:
                        pitch_file.append(rest_pitch)
                    else:
                        if not new_note:
                            pitch_file.append(hold_pitch)
                        else:
                            pitch_file.append(note.pitch)
                            new_note = False
                    time_step += self.minStep
            
            if offset > 1e-1:
                steps = int(offset / self.minStep)
                for j in range(steps):
                    chord_file.append(c_index)
                    pitch_file.append(hold_pitch)
            #assert alignment
            if not (len(chord_file) == len(pitch_file)):
                print(self.midi_file['name'], 'chord file', len(chord_file), 'pitch_file', len(pitch_file))
            assert(len(chord_file) == len(pitch_file))  
            #save             
            self.midi_file["chord_seq"] = chord_file
            self.midi_file["notes"] = pitch_file
            print("calc chords and melody success!")
            return self.midi_file
        print("Error: No dataset called " +  self.datasetName)
        return None

    def to_numpy(self):
        chords = self.midi_file["chord_seq"]
        notes = self.midi_file["notes"]
        assert(len(chords) == len(notes))
        length = len(chords)
        note_array = np.zeros((1, length, 130))
        for idx, note in enumerate(self.midi_file["notes"]):
            note_array[0, idx, note] = 1
        chord_array = np.zeros((1, length, 12))
        for idx, chord in enumerate(chords):
            cname = self.cl.index2name(chord)
            cnotes = self.cl.name2note(cname)
            if cnotes is None:
                continue
            for k in cnotes:
                chord_array[0,idx,k % 12] = 1
        return note_array, chord_array
        

    def writeFile(self, output = ""):
        print("begin write file")
        midi_file = self.midi_file
        output_file = []
        if midi_file.__contains__("name"):
            output_file.append("Name: " + midi_file["name"] + "\n")
        if midi_file.__contains__("chord_seq"):
            output_file.append("Chord Sequence:\n")
            for c in midi_file["chord_seq"]:
                output_file.append(str(c) + " ")
            output_file.append("\n")
        if midi_file.__contains__("notes"):
            output_file.append("Notes:\n")
            for c in midi_file["notes"]:
                output_file.append(str(c) + " ")
            output_file.append("\n")
        with open(output + midi_file["name"] + ".txt","w") as f:
            f.writelines(output_file)
        print("finish output!")
        return True

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

def shift_tone_up(chord, num):
    chord_1 = chord[:, :, : num]
    chord_2 = chord[:, :, num:]
    shifted_chord = torch.cat((chord_2, chord_1), dim=-1)
    return shifted_chord

def load_numpy_from_midi(midi_dir = '', generate_midi = False):
    loader = two_bar_dataloader('Nottingham', 0.125)
    file = loader.load(midi_dir)
    file = loader.getMelodyAndChordSeq("Seven")
    note, chord = loader.to_numpy()
    assert(note.shape[1] == 32)
    assert(note.shape[1] == chord.shape[1])
    if generate_midi:
        numpy_to_midi_with_condition(note[0], chord[0], 0.125, 'test.mid')
    return note, chord
    
def analogy(melody_A, chord_A, melody_B, chord_B, savename = ''):
    weight_path = 'params/Mon Jul  6 22-51-32 2020/best_fitted_params.pt'
    with open('code/model_config.json') as f:
        args = json.load(f)
    model = VAE(130, args['hidden_dim'], 3, 12, args['pitch_dim'], args['rhythm_dim'], args['time_step']).cuda()
    model.load_state_dict(torch.load(weight_path)['model_state_dict'])
    model.eval()
    #data
    melody_A = torch.from_numpy(melody_A).cuda().float()
    melody_B = torch.from_numpy(melody_B).cuda().float()
    chord_A = torch.from_numpy(chord_A).cuda().float()
    chord_B = torch.from_numpy(chord_B).cuda().float()
    #encode
    distr_A_pitch, distr_A_rhythm = model.encoder(melody_A, chord_A)
    distr_B_pitch, distr_B_rhythm = model.encoder(melody_B, chord_B)
    z_A_pitch = distr_A_pitch.mean
    z_A_rhythm = distr_A_rhythm.mean
    z_B_pitch = distr_B_pitch.mean
    z_B_rhythm = distr_B_rhythm.mean
    #decode
    recon_rhythm_A = model.rhythm_decoder(z_A_rhythm)
    recon_rhythm_B = model.rhythm_decoder(z_B_rhythm)
    #Reconstruct with chord of A
    #new_chord = shift_tone_up(chord_A, 11)
    recon = model.final_decoder(z_A_pitch, recon_rhythm_A, chord_B)
    recon = recon.detach().cpu()[0]
    idx = recon.max(1)[1]
    out = torch.zeros_like(recon)
    arange = torch.arange(out.size(0)).long()
    out[arange, idx] = 1
    #generate midi
    numpy_to_midi_with_condition(out.numpy(), chord_B.detach().cpu().numpy()[0], time_step = 0.125, output='new_reproduce/'+savename)


if __name__ == '__main__':
    melody_A, chord_A = load_numpy_from_midi('samples from author/2-bar-midis/source-figure-4.mid')
    melody_B, chord_B = load_numpy_from_midi('samples from author/2-bar-midis/chord-generation-2-minor-figure9b.mid')
    analogy(melody_A, chord_A, melody_B, chord_B, 'no_sample_chord-generation-2-minor-figure9b.mid')

    

