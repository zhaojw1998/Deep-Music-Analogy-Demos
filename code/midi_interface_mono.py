import pretty_midi as pyd
import numpy as np
import os
#from chordloader import Chord_Loader
import copy
from tqdm import tqdm
import sys
import pickle
from joblib import delayed
from joblib import Parallel
import time

class midi_interface_mono(object):
    def __init__(self):
        self.hold_pitch = 128
        self.rest_pitch = 129

    def load_single(self, file_path):
        midi_data = pyd.PrettyMIDI(file_path) 
        tempo = midi_data.get_tempo_changes()[-1][0]
        melodySequence = self.getMelodySeq_byBeats(midi_data)
        #print(melodySequence)
        matrix = self.melodySeq2Numpy(melodySequence)
        batchTarget = self.numpySplit(matrix, WINDOWSIZE=32, HOPSIZE=32)
        return batchTarget, tempo
   
    
    def getMelodySeq_byBeats(self, midi_data):
        melodySequence = []
        anchor = 0
        note = midi_data.instruments[0].notes[anchor]
        start = note.start
        new_note = True
        time_stamps = midi_data.get_downbeats()
        #print(time_stamps)
        #print(midi_data.get_tempo_changes())
        for i in range(len(time_stamps)-1):
            s_curr = time_stamps[i]
            s_next = time_stamps[i+1]
            delta = (s_next - s_curr) / 16
            for i in range(16):
                while note.end < (s_curr + i * delta) and anchor < len(midi_data.instruments[0].notes)-1:
                    #if anchor >= len(midi_data.instruments[0].notes)-1:
                    #    start = 1e5
                    #    break
                    anchor += 1
                    note = midi_data.instruments[0].notes[anchor]
                    start = note.start
                    new_note = True
                if s_curr + i * delta < start:
                    melodySequence.append(self.rest_pitch)
                else:
                    if not new_note:
                        melodySequence.append(self.hold_pitch)
                    else:
                        melodySequence.append(note.pitch)
                        new_note = False
        return melodySequence

    def getMelodySeq_byTimeStamp(self, midi_data):
        tempo = np.mean(midi_data.get_tempo_changes()[-1])
        minStep = 60 / tempo / 4
        start = midi_data.instruments[0].notes[0].start
        end = midi_data.instruments[0].notes[-1].end
        #print(start, end)
        steps = int(round((end - start) / minStep))
        #print(steps)
        melodySequence = []
        anchor = 0
        note = midi_data.instruments[0].notes[anchor]
        time_step = start
        new_note = True
        for step in range(steps):
            while note.end <= time_step and time_step <= end:
                anchor += 1
                note = midi_data.instruments[0].notes[anchor]
                new_note = True

            if note.start > time_step:
                melodySequence.append(self.rest_pitch)
            else:
                if not new_note:
                    melodySequence.append(self.hold_pitch)
                else:
                    melodySequence.append(note.pitch)
                    new_note = False
            time_step += minStep
        #print(melodySequence)
        return melodySequence
    
    def midiReconFromSeq(self, sequence, tempo):
        minStep = 60 / tempo / 4
        melodyRecon = pyd.PrettyMIDI(initial_tempo=tempo)
        program = pyd.instrument_name_to_program('Violin')
        melody = pyd.Instrument(program=program)
        onset_or_rest = [i for i in range(len(sequence)) if not sequence[i]==self.hold_pitch]
        onset_or_rest.append(len(sequence))
        for idx, onset in enumerate(onset_or_rest[:-1]):
            if sequence[onset] == self.rest_pitch:
                continue
            else:
                pitch = sequence[onset]
                start = onset * minStep
                end = onset_or_rest[idx+1] * minStep
                noteRecon = pyd.Note(velocity=100, pitch=pitch, start=start, end=end)
                melody.notes.append(noteRecon)
        melodyRecon.instruments.append(melody)
        #melodyRecon.write('melodyReconTest.mid')
        return melodyRecon
        
    def melodySeq2Numpy(self, sequence, ROLL_SIZE=130):
        melodyMatrix = np.zeros((len(sequence), ROLL_SIZE))
        for idx, note in enumerate(sequence):
            melodyMatrix[idx, note] = 1
        return melodyMatrix

    def midiReconFromNumpy(self, melodyMatrix, tempo):
        noteSequence = [np.argmax(melodyMatrix[i]) for i in range(melodyMatrix.shape[0])]
        return self.midiReconFromSeq(noteSequence, tempo)
    
    def numpySplit(self, matrix, WINDOWSIZE=32, HOPSIZE=16):
        splittedMatrix = np.empty((0, WINDOWSIZE, 142))
        chord_compensate = np.zeros((1, WINDOWSIZE, 12))
        #print(matrix.shape[0])
        for idx_T in range(0, matrix.shape[0]-WINDOWSIZE, HOPSIZE):
            sample = np.concatenate((matrix[idx_T:idx_T+WINDOWSIZE, :][np.newaxis, :, :], chord_compensate), axis=-1)
            splittedMatrix = np.concatenate((splittedMatrix, sample), axis=0)
        return splittedMatrix

class accompanySelection(midi_interface_mono):
        def __init__(self, save_root='./'):
            super(accompanySelection, self).__init__()
            self.save_root = save_root
            self.raw_midi_set = []
            self.minStep_set = []
            self.tempo_set = []
            self.start_record = []
            self.melodySequence_set = []
            self.npy_midi_set = []
            self.belonging = []
            self.num_beforeShift = None
        
        def load_dataset(self, dataset_dir):
            print('begin loading dataset')
            time.sleep(.5)
            for midi in tqdm(os.listdir(dataset_dir)):
                #store raw data and relevant infomation
                midi_data = pyd.PrettyMIDI(os.path.join(dataset_dir, midi))
                self.raw_midi_set.append(midi_data)
                tempo = np.mean(midi_data.get_tempo_changes()[-1])
                self.tempo_set.append(tempo)
                minStep = 60 / tempo / 4
                self.minStep_set.append(minStep)
                start = midi_data.instruments[0].notes[0].start
                self.start_record.append(start)
                #convert to and store numpy matrix
                melodySequence = self.getMelodySeq_byBeats(midi_data)
                self.melodySequence_set.append(melodySequence)
                melodyMatrix = self.melodySeq2Numpy(melodySequence)
                self.npy_midi_set.append(melodyMatrix)
        
        def EC2_VAE_batchData(self):
            numTotal = len(self.npy_midi_set)
            print(numTotal)
            NumMiniBatch = numTotal // 10
            print('begin generating batch data for EC2-VAE')
            time.sleep(.5)
            for part, idx_B in enumerate(range(0, numTotal-NumMiniBatch, NumMiniBatch)):
                batchData = np.empty((0, 32, 142))
                chord_complement = np.zeros((1, 32, 12))
                sub_midi_set = self.npy_midi_set[idx_B: idx_B+NumMiniBatch]
                for idx in tqdm(range(len(sub_midi_set))):
                    numpyMatrix = sub_midi_set[idx]
                    for idxT in range(0, numpyMatrix.shape[0]-32, 16):
                        sample = np.concatenate((numpyMatrix[idxT:idxT+32, :][np.newaxis, :, :], chord_complement), axis=-1)
                        if sample[0, 0, 128] == 1:
                            for idx_forward in range(idxT,0, -1):
                                note = self.melodySequence_set[part*NumMiniBatch+idx][idx_forward]
                                if note != 128:
                                    break
                            sample[0, 0, 128] = 0
                            sample[0, 0, note] = 1
                        batchData = np.concatenate((batchData, sample), axis=0)
                        self.belonging.append((part*NumMiniBatch+idx, idxT))
                save_name = 'batchData_part%d.npy'%part
                np.save(os.path.join(self.save_root, save_name), batchData)
                print(batchData.shape)
                #print(len(self.belonging))
                time.sleep(.5)
            
            batchData = np.empty((0, 32, 142))
            chord_complement = np.zeros((1, 32, 12))
            sub_midi_set = self.npy_midi_set[idx_B+NumMiniBatch:]
            for idx in tqdm(range(len(sub_midi_set))):
                numpyMatrix = sub_midi_set[idx]
                for idxT in range(0, numpyMatrix.shape[0]-32, 16):
                    sample = np.concatenate((numpyMatrix[idxT:idxT+32, :][np.newaxis, :, :], chord_complement), axis=-1)
                    if sample[0, 0, 128] == 1:
                        for idx_forward in range(idxT,0, -1):
                            note = self.melodySequence_set[(part+1)*NumMiniBatch+idx][idx_forward]
                            if note != 128:
                                break
                        sample[0, 0, 128] = 0
                        sample[0, 0, note] = 1
                    batchData = np.concatenate((batchData, sample), axis=0)
                    self.belonging.append(((part+1)*NumMiniBatch+idx, idxT))
            
            save_name = 'batchData_part%d.npy'%(part+1)
            np.save(os.path.join(self.save_root, save_name), batchData)
            print(batchData.shape)
            print('begin saving auxilary information')
            time.sleep(.5)
            time1 = time.time()
            with open(os.path.join(self.save_root, 'auxilary.txt'), 'wb') as f:
                pickle.dump(self.raw_midi_set, f)
                pickle.dump(self.tempo_set, f)
                pickle.dump(self.minStep_set, f)
                pickle.dump(self.start_record, f)
                pickle.dump(self.belonging, f)
            duration = time.time() - time1
            print('finish, using time %.2fs'%duration)
            time.sleep(0.5)
            #return batchData
            
        def loadAuxilary(self, file_name):
            print('begin loading parameters')
            time.sleep(.5)
            time1=time.time()
            with open(os.path.join(self.save_root, file_name), 'rb') as f:
                self.raw_midi_set = pickle.load(f)
                self.tempo_set = pickle.load(f)
                self.minStep_set = pickle.load(f)
                self.start_record = pickle.load(f)
                self.belonging = pickle.load(f)
                try:
                    #self.num_beforeShift = pickle.load(f)
                    self.num_beforeShift = 238444
                    print(self.num_beforeShift)
                except EOFError:
                    self.num_beforeShift = None
            duration = time.time() - time1
            print('finish loading parameters, using time %.2fs'%duration)
            time.sleep(.5)

        def retriveRawMidi(self, batchIdx):
            if self.num_beforeShift == None:
                midiIdx, idxT = self.belonging[batchIdx]
                minStep = self.minStep_set[midiIdx]
                start = idxT*minStep
                end = (idxT+32)*minStep
                tempo = 60 / minStep / 4
                midi_file = self.raw_midi_set[midiIdx]
                midiRetrive = pyd.PrettyMIDI(initial_tempo=tempo)
                melody = pyd.Instrument(program = pyd.instrument_name_to_program('Violin'))
                accompany = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
                for note in midi_file.instruments[0].notes:
                    if note.end > start and note.start < end:
                        note_recon = pyd.Note(velocity = 100, pitch = note.pitch, start = max(note.start, start)-start, end = min(note.end, end)-start)
                        melody.notes.append(note_recon)
                for note in midi_file.instruments[1].notes:
                    if note.end > start and note.start < end:
                        note_recon = pyd.Note(velocity = 100, pitch = note.pitch, start = max(note.start, start)-start, end = min(note.end, end)-start)
                        accompany.notes.append(note_recon)
                midiRetrive.instruments.append(melody)
                midiRetrive.instruments.append(accompany)
            else:
                midiIdx, idxT = self.belonging[batchIdx % self.num_beforeShift]
                minStep = self.minStep_set[midiIdx]
                start = idxT*minStep
                end = (idxT+32)*minStep
                tempo = 60 / minStep / 4
                midi_file = self.raw_midi_set[midiIdx]
                midiRetrive = pyd.PrettyMIDI(initial_tempo=tempo)
                melody = pyd.Instrument(program = pyd.instrument_name_to_program('Violin'))
                accompany = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
                shift = 6 - (batchIdx // self.num_beforeShift)
                for note in midi_file.instruments[0].notes:
                    if note.end > start and note.start < end:
                        note_recon = pyd.Note(velocity = 100, pitch = note.pitch + shift, start = max(note.start, start)-start, end = min(note.end, end)-start)
                        melody.notes.append(note_recon)
                for note in midi_file.instruments[1].notes:
                    if note.end > start and note.start < end:
                        note_recon = pyd.Note(velocity = 100, pitch = note.pitch + shift, start = max(note.start, start)-start, end = min(note.end, end)-start)
                        accompany.notes.append(note_recon)
                midiRetrive.instruments.append(melody)
                midiRetrive.instruments.append(accompany)
            #print(accompany.notes)
            #midiRetrive.write('test_retrive.mid')
            #batchData = np.load(os.path.join(self.save_root, 'batchData_part0.npy'))
            #melodyRecon = self.midiReconFromNumpy(batchData[batchIdx, :, :130], tempo)
            #melodyRecon.write('test_recon.mid')
            return midiRetrive

        def tone_shift(self):
            with open(os.path.join(self.save_root, 'auxilary.txt'), 'rb') as f:
                raw_midi_set = pickle.load(f)
                tempo_set = pickle.load(f)
                minStep_set = pickle.load(f)
                start_record = pickle.load(f)
                belonging = pickle.load(f)
            original_batchData = np.load(os.path.join(self.save_root, 'batchData.npy'))
            shifted_batchData = np.zeros((original_batchData.shape[0]*12, original_batchData.shape[1], original_batchData.shape[2]))
            for idx, i in enumerate(tqdm(range(-6, 6, 1))):
                #print(idx)
                tmp = original_batchData[:, :, :128]
                #print(tmp.shape)
                tmp = np.concatenate((tmp[:, :, i:], tmp[:, :, :i]), axis=-1)
                tmp = np.concatenate((tmp, original_batchData[:, :, 128:]), axis=-1)
                (shifted_batchData[original_batchData.shape[0]*idx: original_batchData.shape[0]*(idx+1), :, :]) = tmp
            np.save(os.path.join(self.save_root, 'batchData_shifted.npy'), shifted_batchData)
            with open(os.path.join(self.save_root, 'auxilary_shifted.txt'), 'wb') as f:
                pickle.dump(raw_midi_set, f)
                pickle.dump(tempo_set, f)
                pickle.dump(minStep_set, f)
                pickle.dump(start_record, f)
                pickle.dump(belonging, f)
                pickle.dump(len(belonging), f)
            return len(belonging)

if __name__ == '__main__':
    
    
    converter1 = accompanySelection(save_root='/gpfsnyu/scratch/jz4807/musicalion_melody_batch_data/')
    #converter1.load_dataset('/gpfsnyu/home/jz4807/datasets/Musicalion/solo+piano/data2dual_track/')
    #converter1 = accompanySelection(save_root='./data_save_root')
    #converter1.load_dataset('../dule_track_trial')
    #converter1.EC2_VAE_batchData()
    #converter1.loadAuxilary()
    #midiRetrive = converter1.retriveRawMidi(90)
    #converter2 = midi_interface()
    #batch = converter2.load_single('D:/Download/Program/Musicalion/solo+piano/melody_only/RM-P003.SMF_SYNC-melody.mid')
    #print(batch.shape)
    """
    converter = midi_interface()
    converter.load_single('D:/Download/Program/Musicalion/solo+piano/data2dual_track/ssccm11.mid')
    #converter.load_single('D:/Download/Program/Musicalion/solo+piano/melody_only/RM-P003.SMF_SYNC-melody.mid')"""
    
    #test shift tone
    #length = converter1.tone_shift()
    #print(length)
    #converter1.loadAuxilary('auxilary.txt')
    #midiRetrive = converter1.retriveRawMidi(90)
    #midiRetrive.write('90_no_change.mid')
    converter1.loadAuxilary('auxilary_shifted.txt')
    midiRetrive = converter1.retriveRawMidi(90)
    midiRetrive.write('90.mid')
    midiRetrive = converter1.retriveRawMidi(90+238514)
    midiRetrive.write('90+'+str(238514)+'.mid')