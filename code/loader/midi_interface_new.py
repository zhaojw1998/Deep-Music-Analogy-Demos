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

class midi_interface(object):
    def __init__(self):
        self.hold_pitch = 128
        self.rest_pitch = 129

    def load_single(self, file_path):
        midi_data = pyd.PrettyMIDI(file_path)
        tempo = midi_data.get_tempo_changes()[-1][0]
        minStep = 60 / tempo / 4
        #melodySequence = self.getMelodySeq(midi_data, minStep)
        #self.midiReconFromSeq(melodySequence, minStep)
        #matrix = self.melodySeq2Numpy(melodySequence)
        #recon = self.midiReconFromNumpy(matrix, minStep)
    
    def getMelodySeq(self, midi_data, minStep):
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
    
    def midiReconFromSeq(self, sequence, minStep):
        tempo = 60 / minStep / 4
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

    def midiReconFromNumpy(self, melodyMatrix, minStep):
        noteSequence = [np.argmax(melodyMatrix[i]) for i in range(melodyMatrix.shape[0])]
        return self.midiReconFromSeq(noteSequence, minStep)



if __name__ == '__main__':
    class accompanySelection(midi_interface):
        def __init__(self):
            super(accompanySelection, self).__init__()
            self.raw_midi_set = []
            self.minStep_set = []
            self.start_record = []
            self.melodySequence_set = []
            self.npy_midi_set = []
            self.belonging = []
        
        def load_dataset(self, dataset_dir):
            print('begin loading dataset')
            for midi in tqdm(os.listdir(dataset_dir)):
                #store raw data and relevant infomation
                midi_data = pyd.PrettyMIDI(os.path.join(dataset_dir, midi))
                self.raw_midi_set.append(midi_data)
                tempo = midi_data.get_tempo_changes()[-1][0]
                minStep = 60 / tempo / 4
                self.minStep_set.append(minStep)
                start = midi_data.instruments[0].notes[0].start
                self.start_record.append(start)
                #convert to and store numpy matrix
                melodySequence = self.getMelodySeq(midi_data, minStep)
                self.melodySequence_set.append(melodySequence)
                melodyMatrix = self.melodySeq2Numpy(melodySequence)
                self.npy_midi_set.append(melodyMatrix)
        
        def EC2_VAE_batchData(self):
            numTotal = len(self.npy_midi_set)
            print(numTotal)
            NumMiniBatch = numTotal // 10
            print('begin generating batch data for EC2-VAE')
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
                np.save('batchData_part%d.npy'%part, batchData)
                print(batchData.shape)
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
            np.save('batchData_part%d.npy'%(part+1), batchData)
            print(batchData.shape)

            print('begin saving auxilary information')
            time1 = time.time()
            with open('auxilary.txt', 'wb') as f:
                pickle.dump(self.raw_midi_set, f)
                pickle.dump(self.minStep_set, f)
                pickle.dump(self.start_record, f)
                pickle.dump(self.belonging, f)
            duration = time.time() - time1
            print('finish, using time %.2fs'%duration)
            #return batchData

        def loadAuxilary(self):
            with open('auxilary.txt', 'rb') as f:
                self.raw_midi_set = pickle.load(f)
                self.minStep_set = pickle.load(f)
                self.start_record = pickle.load(f)
                self.belonging = pickle.load(f)

        def retriveRawMidi(self, batchIdxList):
            midiRetrive = pyd.PrettyMIDI()
            melody = pyd.Instrument(program = pyd.instrument_name_to_program('Violin'))
            accompany = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
            past = 0
            previous = 0
            for batchIdx in batchIdxList:
                midiIdx, idxT = self.belonging[previous+batchIdx]
                minStep = self.minStep_set[midiIdx]
                start = self.start_record[midiIdx] + idxT*minStep
                end = self.start_record[midiIdx] + (idxT+32)*minStep
                #tempo = 60 / minStep / 4
                midi_file = self.raw_midi_set[midiIdx]
                for note in midi_file.instruments[0].notes:
                    if note.end > start and note.start < end:
                        note_recon = pyd.Note(velocity = 100, pitch = note.pitch, start = max(note.start, start)-start+past, end = min(note.end, end)-start+past)
                        melody.notes.append(note_recon)
                for note in midi_file.instruments[1].notes:
                    if note.end > start and note.start < end:
                        note_recon = pyd.Note(velocity = 100, pitch = note.pitch, start = max(note.start, start)-start+past, end = min(note.end, end)-start+past)
                        accompany.notes.append(note_recon)
                past += (end - start)
            midiRetrive.instruments.append(melody)
            midiRetrive.instruments.append(accompany)
            midiRetrive.write('test_retrive.mid')
            batchData = np.load('batchData_part0.npy')
            melodyRecon = self.midiReconFromNumpy(batchData[batchIdx, :, :130], minStep)
            melodyRecon.write('test_recon.mid')
            return midiRetrive

    converter = accompanySelection()
    converter.load_dataset('/gpfsnyu/home/jz4807/datasets/Musicalion/solo+piano/data2dual_track/')
    #converter.load_dataset('../dule_track_trial')
    converter.EC2_VAE_batchData()
    #converter.loadAuxilary()
    midiRetrive = converter.retriveRawMidi([19, 21, 23, 25])