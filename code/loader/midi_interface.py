import pretty_midi as pyd
import numpy as np
import os
from chordloader import Chord_Loader
import copy
from tqdm import tqdm
import sys
import pickle
from joblib import delayed
from joblib import Parallel


class midi_interface(object):
    """
    functions for batched midis:
        load_batch(directory)
        getMelodyAndChordSeq_batch(recogLevel = "Seven")
        midiDataAugment_batch(bottom = 40, top = 85)
        writeTxtFile_batch(output = "")
        midiSaveNumpy_batch
        midi2numpy_batch
        recon_midi_snippets(self, unit_idx_set, tempo=120)
    functions  for a single midi file:
        load_single(file_path)
        getMelodyAndChordSeq_single(self, recogLevel = "Seven")
        writeTxtFile_single(output = "")
        text2midi_single(text_ad, recogLevel = "Seven",output = "test.mid")
        midi2Numpy_single()
        numpy2midiWithCondition_single(self, sample, condition, time_step = 0.125, output='sample/sample.mid')
    """
    def __init__(self, minStep=0.03125):
        self.datasetName = 'Nottingham'
        self.minStep = minStep
        self.raw_midi = None
        self.discretized_midi = None
        self.tempo = None
        self.start_time_record = []
        self.pitch_shift_record = []
        self.belonging = []

    def load_batch(self, directory, record_dir=''):
        """currently only load dual track in batch""" 
        path = os.listdir(directory)
        print("start to load mid from %s" % directory)
        # Nottingham dataset is processed by y
        if self.datasetName == "Nottingham":
            self.raw_midi = [] 
            self.directory = directory
            for midi_file in tqdm(path):
                midi_data = pyd.PrettyMIDI(os.path.join(self.directory, midi_file))
                self.raw_midi.append({"name": (midi_file.split("."))[0], "raw": midi_data})
            print("loading %s success! %d files in total" %(directory, len(self.raw_midi)))
            if not record_dir == '':
                with open(os.path.join(record_dir, 'start_time_record.txt'), 'rb') as f:
                    self.start_time_record = pickle.load(f)
                #with open(os.path.join(record_dir, 'pitch_shift_record.txt'), 'rb') as f:
                #    self.pitch_shift_record = pickle.load(f)
                with open(os.path.join(record_dir, 'belonging.txt'), 'rb') as f:
                    self.belonging = pickle.load(f)
            return self.raw_midi
        print("Error: No dataset called " +  self.datasetName)
        return None

    def load_single(self, file_path):
        """midi_data = load_single(file_path)"""
        if self.datasetName == "Nottingham":
            midi_data = pyd.PrettyMIDI(file_path)
            self.raw_midi = [{"name": (file_path.split("."))[0].split('/')[-1], "raw": midi_data}]
            print("loading file success!")
            self.tempo = np.mean(midi_data.get_tempo_changes()[-1])
            #print(self.tempo)
            #print(midi_data.instruments[0].notes)
            return self.raw_midi[0]
        print("Error: No dataset called " +  self.datasetName)
        return None
    
    def getMelodyAndChordSeq(self, recogLevel = "Mm", save_root='./'):
        """[midi_datas] = getMelodyAndChordSeq(recogLevel) for batch;
            midi_data = getMelodyAndChordSeq(recogLevel) for single"""
        print("start to get melody and sequences")
        self.recogLevel = recogLevel
        # 25 one hot vectors, 0-11 for major, 12-23 for minor , 24 for NC
        self.discretized_midi = []
        for i in tqdm(range(len(self.raw_midi))):
            midi_data = self.raw_midi[i]["raw"]
            self.cl = Chord_Loader(recogLevel = self.recogLevel)
            chord_set = []
            chord_time = [0.0, 0.0]
            last_time = midi_data.instruments[0].notes[0].start # Define the chord recognition system
            chord_file = []
            pitch_file = []
            rest_pitch = 129
            hold_pitch = 128
            #check chord track
            self.discretized_midi.append({})
            self.discretized_midi[i]["name"] = self.raw_midi[i]['name']
            self.discretized_midi[i]["chords"] = []
            if not len(midi_data.instruments) == 2:
                start_time = midi_data.instruments[0].notes[0].start
                end_time = midi_data.instruments[0].notes[-1].end
                self.discretized_midi[i]["chords"].append({"start":start_time,"end": end_time, "chord" : "NC"})
                self.start_time_record.append(start_time)
            else:
                #if not len(midi_data.instruments) == 2:
                #    print(i, self.midi_files[i]['name'], 'more tha 2 tracks')
                #initialize save list for melody and chord notes
                
                #for chord
                start_time = midi_data.instruments[1].notes[0].start
                self.start_time_record.append(start_time)
                for note in midi_data.instruments[1].notes:
                    if len(chord_set) == 0:
                        chord_set.append(note.pitch)
                        chord_time[0] = note.start
                        chord_time[1] = note.end
                    else:
                        if note.start == chord_time[0] and note.end == chord_time[1]:
                            chord_set.append(note.pitch)
                        else:
                            if last_time < chord_time[0]:
                                self.discretized_midi[i]["chords"].append({"start":last_time,"end": chord_time[0], "chord" : "NC"})
                            self.discretized_midi[i]["chords"].append({"start":chord_time[0],"end":chord_time[1],"chord": self.cl.note2name(chord_set)})
                            last_time = chord_time[1]
                            chord_set = []
                            chord_set.append(note.pitch)
                            chord_time[0] = note.start
                            chord_time[1] = note.end 
                if chord_set:
                    if last_time < chord_time[0]:
                        self.discretized_midi[i]["chords"].append({"start":last_time ,"end": chord_time[0], "chord" : "NC"})
                    self.discretized_midi[i]["chords"].append({"start":chord_time[0],"end":chord_time[1],"chord": self.cl.note2name(chord_set)})
                    last_time = chord_time[1]
            #for melody and chord
            anchor = 0
            note = midi_data.instruments[0].notes[anchor]
            new_note = True
            for idx, c in enumerate(self.discretized_midi[i]["chords"]):
                if c["end"] - c["start"] < self.minStep:
                    continue
                c1 = self.discretized_midi[i]["chords"][min(idx+1, len(self.discretized_midi[i]["chords"])-1)]
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
                        try:
                            note = midi_data.instruments[0].notes[anchor]
                            new_note = True
                        except:
                            break
                    if note.start-self.minStep/2 > time_step:
                        pitch_file.append(rest_pitch)
                    else:
                        if not new_note:
                            pitch_file.append(hold_pitch)
                        else:
                            pitch_file.append(note.pitch)
                            new_note = False
                    time_step += self.minStep
            #assert alignment
            if not (len(chord_file) == len(pitch_file)):
                print(i, self.discretized_midi[i]['name'], 'chord file', len(chord_file), 'pitch_file', len(pitch_file))
            assert(len(chord_file) == len(pitch_file))  
            #save             
            self.discretized_midi[i]["chord_seq"] = chord_file
            self.discretized_midi[i]["notes"] = pitch_file
            del self.discretized_midi[i]["chords"]
        print("calc chords success! %d files in total" % len(self.discretized_midi))
        if not save_root == '':
            with open(os.path.join(save_root, 'start_time_record.txt'), 'wb') as f:
                pickle.dump(self.start_time_record, f)
        #print(self.start_time_record)
        if len(self.discretized_midi) == 1:
            return self.discretized_midi[0]
        return self.discretized_midi

    def midiDataAugment_batch(self,bottom = 40, top = 85, save_root='./'):
        print("start to augment data")
        #print("Be sure you get the chord functions before!")
        augment_data = []
        cl = Chord_Loader(recogLevel = self.recogLevel)
        for i in tqdm(range(-5,7,1)):
            for idx, x in enumerate(self.discretized_midi):
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
                    #midi_file["name"] += "-switch(" + str(i) + ")" 
                    augment_data.append(midi_file)
                    self.pitch_shift_record.append((idx, i))
            #print("finish augment %d data" % i)
        self.discretized_midi = augment_data
        if not save_root == '':
            with open(os.path.join(save_root, 'pitch_shift_record.txt'), 'wb') as f:
                pickle.dump(self.pitch_shift_record, f)
        #print(self.pitch_shift_record)
        # random.shuffle(self.midi_files)
        print("data augment success! %d files in total" % len(self.discretized_midi))
        return self.discretized_midi

    def writeFile(self, save_root = ""):
        """convert a (batch of) midi_data into .txt file"""
        #print("begin write file from %s" % self.directory)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        for midi_file in self.discretized_midi:
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
            with open(os.path.join(save_root, midi_file["name"] + ".txt"),"w") as f:
                f.writelines(output_file)
        print("finish output! %d files in total" % len(self.discretized_midi))
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
        chords = self.discretized_midi[0]["chord_seq"]
        notes = self.discretized_midi[0]["notes"]
        assert(len(chords) == len(notes))
        length = len(chords)
        note_array = np.zeros((1, length, 130))
        for idx, note in enumerate(notes):
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
        for midi_file in self.discretized_midi:
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
            save_name = os.path.join(save_root, midi_file['name'] + '.npy')
            np.save(save_name, melody_with_chord)
    
    def midi2numpy_batch_inner(self, file_idx, window_size, hop_size):
        midi_file = self.discretized_midi[file_idx]
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
        #print(melody_with_chord.shape)
        sub_numpy_batch = np.empty((0, window_size, 130+12))
        sub_belonging = []
        for i in range(0, melody_with_chord.shape[1]-window_size, hop_size):
            sample = melody_with_chord[:, i: i+window_size, :]
            sub_numpy_batch = np.concatenate((sub_numpy_batch, sample), axis = 0)
            midiBelong = self.pitch_shift_record[file_idx][0]
            shiftBelong = self.pitch_shift_record[file_idx][1]
            start = i*self.minStep
            end = (i+window_size)*self.minStep
            sub_belonging.append((midiBelong, shiftBelong, (start, end)))
        return (sub_numpy_batch, sub_belonging)
    
    def midi2numpy_batch(self, window_size = 32, hop_size = 16, save_root = './'):
        if self.pitch_shift_record == []:
            self.pitch_shift_record = [(i, 0) for i in range(len(self.start_time_record))]
        numpy_batch = np.empty((0, window_size, 130+12))
        self.belonging = []
        print('start converting midi to numpy')
        result_list = Parallel(n_jobs=4)(delayed(self.midi2numpy_batch_inner)(file_idx, window_size, hop_size) for file_idx in tqdm(range(len(self.discretized_midi))))
        for i in tqdm(range(len(result_list))):
            numpy_batch = np.concatenate((numpy_batch, result_list[i][0]), axis=0)
            print(numpy_batch.shape)
            self.belonging = self.belonging + result_list[i][1]
            print(len(self.belonging))
        if not save_root == '':
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            np.save(os.path.join(save_root, 'splitted_data.npy'), numpy_batch)
            with open(os.path.join(save_root, 'belonging.txt'), 'wb') as f:
                pickle.dump(self.belonging, f)
        #print(self.belonging, len(self.belonging))
        print('Conversion success! %d files in total' % numpy_batch.shape[0])
        return numpy_batch
    
    
    def numpy2midiWithCondition_single(self, sample, condition, time_step = 0.125, output='sample/sample.mid'):
        if not self.tempo == None:
            music = pyd.PrettyMIDI(initial_tempo=self.tempo)
        else:
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

    def recon_midi_snippets(self, unit_idx_set, tempo=120):
        if not self.tempo == None:
            gen_midi = pyd.PrettyMIDI(initial_tempo=self.tempo)
        else:
            gen_midi = pyd.PrettyMIDI(initial_tempo=tempo)
        melody = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
        accompany = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
        past = 0
        for unit_idx in unit_idx_set:
            midi_idx, shift_tone, (start, end) = self.belonging[unit_idx]
            #print(midi_idx, shift_tone, (start, end))
            start = self.start_time_record[midi_idx] + start
            end = self.start_time_record[midi_idx] + end
            #print(start, end)
            midi_file = self.raw_midi[midi_idx]['raw']
        
            for note in midi_file.instruments[0].notes:
                if note.end > start and note.start < end:
                    note_recon = pyd.Note(velocity = 100, pitch = note.pitch+shift_tone, start = max(note.start, start)-start+past, end = min(note.end, end)-start+past)
                    melody.notes.append(note_recon)
            for note in midi_file.instruments[1].notes:
                if note.end > start and note.start < end:
                    note_recon = pyd.Note(velocity = 100, pitch = note.pitch+shift_tone, start = max(note.start, start)-start+past, end = min(note.end, end)-start+past)
                    accompany.notes.append(note_recon)
            past += end-start
        gen_midi.instruments.append(melody)
        gen_midi.instruments.append(accompany)
        gen_midi.write('test_rehemerate_snippet.mid')
        return gen_midi
        


if __name__ == '__main__':
    #demo reading midi to numpy:
    """
    converter = midi_interface(0.125)
    converter.load_single("D:/Download/Program/Musicalion/solo+piano/data/ssccm581.mid")
    converter.getMelodyAndChordSeq("Seven")
    mel, chord = converter.midi2Numpy_single()
    converter.numpy2midiWithCondition_single(mel[0], chord[0], time_step = 0.125, output='sample.mid')"""
    """
    converter = midi_interface(0.125)
    midi = converter.load_single("sample.mid")["raw"]
    print(len(midi.instruments))
    #for i in range(len(midi.instruments)):
    #    print(len(midi.instruments[i].notes))
    #print(midi.instruments[0].notes[0].start, midi.instruments[0].notes[-1].end)"""
    
    """
    #test scripts
    converter = midi_interface(0.125)
    converter.load_batch('../dule_track_trial',)
    converter.getMelodyAndChordSeq(recogLevel = "Seven", save_root='./')
    #converter.midiDataAugment_batch()
    #converter.writeFile('testWriteFile')
    #converter.text2midi_single(os.path.join('testWriteFile', (os.listdir('testWriteFile')[0])), recogLevel = "Seven",output = "test_text2midi.mid")
    #converter.midiSaveNumpy_batch('testSaveNumpy')
    batch = converter.midi2numpy_batch(save_root='./')
    #batch=np.load('./splitted_data.npy')
    converter.numpy2midiWithCondition_single(batch[301, :, :130], batch[301, :, 130:], time_step = 0.125, output='testNumpy2Midi.mid')
    converter.recon_midi_snippets([301])
    """

    converter = midi_interface(0.125)
    converter.load_single('../dule_track_trial/ssccm17.mid')
    converter.getMelodyAndChordSeq(recogLevel = "Mm", save_root='')
    mel, chord = converter.midi2Numpy_single()
    converter.numpy2midiWithCondition_single(mel[0], chord[0], time_step = 0.125, output='testNumpy2Midi-single.mid')