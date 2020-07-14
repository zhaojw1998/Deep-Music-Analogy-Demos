import pretty_midi as pyd
import numpy as np
import os
import random
import music21 as m21
from chordloader import Chord_Loader
import copy
from tqdm import tqdm
import sys

class midi_processor(object):
    """
    functions for batched midis:
        load_batch(directory)
        getMelodyAndChordSeq_batch(recogLevel = "Seven")
        midiDataAugment_batch(bottom = 40, top = 85)
        writeTxtFile_batch(output = "")
        midiSaveNumpy_batch
        midi2numpy_batch
    functions  for a single midi file:
        load_single(file_path)
        getMelodyAndChordSeq_single(self, recogLevel = "Seven")
        writeTxtFile_single(output = "")
        text2midi_single(text_ad, recogLevel = "Seven",output = "test.mid")
        midi2Numpy_single()
        numpy2midiWithCondition_single(self, sample, condition, time_step = 0.125, output='sample/sample.mid')
    """
    def __init__(self, datasetName, minStep = 0.03125):
        self.datasetName = datasetName
        self.minStep = minStep

    def load_batch(self, directory):
        """currently only load dual track in batch"""
        path = os.listdir(directory)
        print("start to load mid from %s" % directory)
        # Nottingham dataset is processed by y
        if self.datasetName == "Nottingham":
            self.midi_files = [] 
            self.directory = directory
            for midi_file in tqdm(path):
                midi_data = pyd.PrettyMIDI(directory + midi_file)
                #if not (len(midi_data.instruments)) == 2:
                #    continue
                self.midi_files.append({"name": (midi_file.split("."))[0], "raw": midi_data})
            print("loading %s success! %d files in total" %(directory, len(self.midi_files)))
            return self.midi_files
        print("Error: No dataset called " +  self.datasetName)
        return None

    def load_single(self, file_path):
        """load dual or single track
        """
        if self.datasetName == "Nottingham":
            midi_data = pyd.PrettyMIDI(file_path)
            self.midi_file = {"name": (file_path.split("."))[0].split('/')[-1], "raw": midi_data}
            print("loading file success!")
            #print(midi_data.instruments[0].notes)
            return self.midi_file
        print("Error: No dataset called " +  self.datasetName)
        return None

    def getMelodyAndChordSeq_batch(self, recogLevel = "Mm"):
        print("start to get melody and sequences")
        self.recogLevel = recogLevel
        if self.datasetName == "Nottingham":
            # 25 one hot vectors
            # 0-11 for major
            # 12-23 for minor 
            # 24 for NC
            for i in tqdm(range(len(self.midi_files))):
                midi_data = self.midi_files[i]["raw"]
                self.cl = Chord_Loader(recogLevel = self.recogLevel)
                chord_set = []
                chord_time = [0.0, 0.0]
                last_time = midi_data.instruments[0].notes[0].start # Define the chord recognition system
                chord_file = []
                pitch_file = []
                rest_pitch = 129
                hold_pitch = 128
                #check chord track
                if len(midi_data.instruments) == 1:
                    start_time = midi_data.instruments[0].notes[0].start
                    end_time = midi_data.instruments[0].notes[-1].end
                    self.midi_file["chords"].append({"start":start_time,"end": end_time, "chord" : "NC"})
                else:
                    if not len(midi_data.instruments) == 2:
                        print(i, self.midi_files[i]['name'], 'more tha 2 tracks')
                    #initialize save list for melody and chord notes
                    self.midi_files[i]["chords"] = []
                    self.midi_files[i]["pitches"] = []
                    #for chord
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
                                    self.midi_files[i]["chords"].append({"start":last_time,"end": chord_time[0], "chord" : "NC"})
                                self.midi_files[i]["chords"].append({"start":chord_time[0],"end":chord_time[1],"chord": self.cl.note2name(chord_set)})
                                last_time = chord_time[1]
                                chord_set = []
                                chord_set.append(note.pitch)
                                chord_time[0] = note.start
                                chord_time[1] = note.end 
                    if chord_set:
                        if last_time < chord_time[0]:
                            self.midi_files[i]["chords"].append({"start":last_time ,"end": chord_time[0], "chord" : "NC"})
                        self.midi_files[i]["chords"].append({"start":chord_time[0],"end":chord_time[1],"chord": self.cl.note2name(chord_set)})
                        last_time = chord_time[1]
                #for melody and chord
                anchor = 0
                note = midi_data.instruments[0].notes[anchor]
                new_note = True
                for idx, c in enumerate(self.midi_files[i]["chords"]):
                    if c["end"] - c["start"] < self.minStep:
                        continue
                    c1 = self.midi_files[i]["chords"][min(idx+1, len(self.midi_files[i]["chords"])-1)]
                    if c1["end"] - c1["start"] < self.minStep:
                        steps = int(round((c1["end"] - c["start"]) / self.minStep))
                    else:
                        steps = int(round((c["end"] - c["start"]) / self.minStep))
                    c_index = self.cl.name2index(c["chord"])
                    time_step = c["start"]
                    for j in range(steps):
                        chord_file.append(c_index)
                        while note.end-self.minStep/2 <= time_step:
                            anchor += 1
                            note = midi_data.instruments[0].notes[anchor]
                            new_note = True
                        if note.start-self.minStep/2 > time_step:
                            pitch_file.append(rest_pitch)
                        else:
                            if not new_note:
                                pitch_file.append(hold_pitch)
                            else:
                                pitch_file.append(note.pitch)
                                new_note = False
                        time_step += self.minStep
                if offset > 1e-1:
                    #print(offset)
                    steps = int(offset / self.minStep)
                    for j in range(steps):
                        chord_file.append(c_index)
                        pitch_file.append(hold_pitch)
                #assert alignment
                if not (len(chord_file) == len(pitch_file)):
                    print(i, self.midi_files[i]['name'], 'chord file', len(chord_file), 'pitch_file', len(pitch_file))
                assert(len(chord_file) == len(pitch_file))  
                #save             
                self.midi_files[i]["chord_seq"] = chord_file
                self.midi_files[i]["notes"] = pitch_file
            print("calc chords success! %d files in total" % len(self.midi_files))
            return self.midi_files
        print("Error: No dataset called " +  self.datasetName)
        return None

    def getMelodyAndChordSeq_single(self, recogLevel = "Mm"):
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
                    while note.end-self.minStep/2 <= time_step:
                        anchor += 1
                        note = midi_data.instruments[0].notes[anchor]
                        new_note = True
                    if note.start-self.minStep/2 > time_step:
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
    
    def midiDataAugment_batch(self,bottom = 40, top = 85):
        print("start to augment data")
        #print("Be sure you get the chord functions before!")
        augment_data = []
        if self.datasetName == "Nottingham":
            cl = Chord_Loader(recogLevel = self.recogLevel)
            for i in tqdm(range(-5,7,1)):
                for x in self.midi_files:
                    midi_file = copy.deepcopy(x)
                    is_add = True
                    for j in range(len(midi_file["notes"])):
                        if midi_file["notes"][j] <= 127:
                            midi_file["notes"][j] += i
                            if midi_file["notes"][j] > top or midi_file["notes"][j] < bottom:
                                is_add = False
                                break
                    for j in range(len(midi_file["chord_seq"])):
                        midi_file["chord_seq"][j] = cl.chord_alu(x = midi_file["chord_seq"][j],scalar = i)
                    if is_add:
                        midi_file["name"] += "-switch(" + str(i) + ")" 
                        augment_data.append(midi_file)
                #print("finish augment %d data" % i)
            self.midi_files = augment_data
            # random.shuffle(self.midi_files)
            print("data augment success! %d files in total" % len(self.midi_files))
            return self.midi_files
        print("Error: No dataset called " +  self.datasetName)
        return False

    def writeFile_batch(self, output = ""):
        print("begin write file from %s" % self.directory)
        for midi_file in self.midi_files:
            output_file = []
            if midi_file.__contains__("name"):
                output_file.append("Name: " + midi_file["name"] + "\n")
            # if midi_file.__contains__("chords"):
            #     output_file.append("Chord:\n")
            #     for c in midi_file["chords"]:
            #         output_file.append(str(c["start"]) + " " + str(c["end"])+ " " + c["chord"] + "\n")
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
        print("finish output! %d files in total" % len(self.midi_files))
        return True

    def writeFile_single(self, output = ""):
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

    def text2midi_single(self, text_ad, recogLevel = "Mm",output = "test.mid"):
        gen_midi = pyd.PrettyMIDI()
        melodies = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
        chords = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
        if self.datasetName == "Nottingham":
            # 130 one hot vectors 
            # 0-127 for pitch
            # 128 for hold 129 for rest
            rest_pitch = 129
            hold_pitch = 128
            with open(text_ad,"r") as f:
                lines = f.readlines()
                read_flag = "none"
                for line in lines:
                    line = line.strip()
                    # if line == "Chord:":
                    #     continue
                    if line == "Chord Sequence:":
                        read_flag = "chord_seq"
                        continue
                    if line == "Notes:":
                        read_flag = "notes"
                        continue
                    if read_flag == "chord_seq":
                        cl = Chord_Loader(recogLevel = recogLevel)
                        elements = line.split(" ")
                        time_shift = 0.0
                        local_duration = 0
                        prev = "NC"
                        for chord in elements:
                            if chord == "":
                                continue
                            chord = cl.index2name(x = int(chord))
                            if chord == prev:
                                local_duration += 1
                            else:
                                if prev == "NC":
                                    prev = chord
                                    time_shift += local_duration * self.minStep
                                    local_duration = 1
                                else:
                                    i_notes = cl.name2note(name = prev, stage = 4)
                                    for i_note in i_notes:
                                        i_note = pyd.Note(velocity = 100, pitch = i_note, 
                                        start = time_shift, end = time_shift + local_duration * self.minStep)
                                        chords.notes.append(i_note)
                                    prev = chord
                                    time_shift += local_duration * self.minStep
                                    local_duration = 1
                        if prev != "NC":
                            i_notes = cl.name2note(name = prev, stage = 4)
                            for i_note in i_notes:
                                i_note = pyd.Note(velocity = 100, pitch = i_note, 
                                start = time_shift, end = time_shift + local_duration * self.minStep)
                                chords.notes.append(i_note)
                        gen_midi.instruments.append(chords)
                        continue
                    if read_flag == "notes":
                        elements = line.split(" ")
                        time_shift = 0.0
                        local_duration = 0
                        prev = rest_pitch
                        for note in elements:
                            note = int(note)
                            if note < 0 or note > 129:
                                continue
                            if note == hold_pitch:
                                local_duration += 1
                            elif note == rest_pitch:
                                time_shift += self.minStep
                            else:
                                if prev == rest_pitch:
                                    prev = note
                                    local_duration = 1
                                else:
                                    i_note = pyd.Note(velocity = 100, pitch = prev, 
                                        start = time_shift, end = time_shift + local_duration * self.minStep)
                                    melodies.notes.append(i_note)
                                    prev = note
                                    time_shift += local_duration * self.minStep
                                    local_duration = 1
                        if prev != rest_pitch:
                            i_note = pyd.Note(velocity = 100, pitch = prev, 
                                        start = time_shift, end = time_shift + local_duration * self.minStep)
                            melodies.notes.append(i_note)
                        gen_midi.instruments.append(melodies)
                        continue
                gen_midi.write(output)
                print("finish render midi on " + output)

    def midi2Numpy_single(self):
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
    
    def midiSaveNumpy_batch(self, save_root):
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        for midi_file in self.midi_files:
            chords = midi_file["chord_seq"]
            notes = midi_file["notes"]
            if not len(chords) == len(notes):
                print(midi_file['name'], len(chords), len(notes))
            assert(len(chords) == len(notes))
            length = len(chords)
            note_array = np.zeros((1, length, 130))
            for idx, note in enumerate(midi_file["notes"]):
                note_array[0, idx, note] = 1
            chord_array = np.zeros((1, length, 12))
            for idx, chord in enumerate(chords):
                cname = self.cl.index2name(chord)
                cnotes = self.cl.name2note(cname)
                if cnotes is None:
                    continue
                for k in cnotes:
                    chord_array[0,idx,k % 12] = 1
            melody_with_chord = np.concatenate((note_array, chord_array), axis=-1)
            save_name = save_root + midi_file['name'] + '.npy'
            np.save(save_name, melody_with_chord)
    
    def midi2numpy_batch(self, window_size = 32, hop_size = 16, save_root = ''):
        self.numpy_batch = np.empty((0, window_size, 130+12))
        for midi_file in self.midi_files:
            chords = midi_file["chord_seq"]
            notes = midi_file["notes"]
            if not len(chords) == len(notes):
                print(midi_file['name'], len(chords), len(notes))
            assert(len(chords) == len(notes))
            length = len(chords)
            note_array = np.zeros((1, length, 130))
            for idx, note in enumerate(midi_file["notes"]):
                note_array[0, idx, note] = 1
            chord_array = np.zeros((1, length, 12))
            for idx, chord in enumerate(chords):
                cname = self.cl.index2name(chord)
                cnotes = self.cl.name2note(cname)
                if cnotes is None:
                    continue
                for k in cnotes:
                    chord_array[0,idx,k % 12] = 1
            melody_with_chord = np.concatenate((note_array, chord_array), axis=-1)
            print(melody_with_chord.shape)
            for i in range(0, melody_with_chord.shape[1]-window_size, hop_size):
                sample = melody_with_chord[:, i: i+window_size, :]
                self.numpy_batch = np.concatenate((self.numpy_batch, sample), axis = 0)
        if not save_root == '':
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            np.save(os.path.join(save_root, 'splitted_data.npy'), self.numpy_batch)
        return self.numpy_batch
    
    def numpy2midiWithCondition_single(self, sample, condition, time_step = 0.125, output='sample/sample.mid'):
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

if __name__ == '__main__':
    #demo reading midi to numpy:
    converter = midi_processor('Nottingham', 0.125)
    converter.load_batch('../dule_track_trial/')
    converter.getMelodyAndChordSeq_batch("Seven")
    converter.midiDataAugment_batch()
    data = converter.midi2numpy_batch(save_root='splitted_data')
    print(data.shape)