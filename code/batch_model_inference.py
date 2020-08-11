import numpy as np
import os
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from model import VAE
from midi_interface_mono_and_chord import midi_interface_mono_and_chord
from midi_interface_mono_and_chord import accompanySelection
from tqdm import tqdm
import time
import platform
import sys

class dataLoader(data.Dataset):
    def __init__(self, dataset_path):
        self.dataset = np.load(dataset_path)

    def __getitem__(self, index):
        item = self.dataset[index]
        melody = item[:, :130]
        chord = item[:, 130:]
        return melody, chord

    def __len__(self):
        return self.dataset.shape[0]

def computeTIV(chroma):
    #inpute size: batch_size * (32*12)
    chroma = chroma.reshape((chroma.shape[0], -1, 12))
    #print('chroma', chroma.shape)
    chroma = chroma / (np.sum(chroma, axis=-1)[:, :, np.newaxis] + 1e-10) #batch * 32 * 12
    TIV = np.fft.fft(chroma, axis=-1)[:, :, 1: 7] #batch * 32 * (6*2)
    #print(TIV.shape)
    TIV = np.concatenate((np.abs(TIV), np.angle(TIV)), axis=-1) #batch * 32 * 12
    return TIV.reshape((TIV.shape[0], -1))

def appearanceMatch(query, search, batchData):
    #query: 32 X 142
    #search: list idxs
    #batchData: numBatch X 32 X 142
    newScore = []
    #print(query.shape)
    for i in range(len(search)):
        #print(search[i])
        candidate = batchData[search[i]]
        #print(candidate.shape)
        assert(query.shape == candidate.shape)
        score = 0
        register = 0
        registerMelody = 0
        for idxT in range(candidate.shape[0]):
            MmChord = query[idxT][130:]
            #print(MmChord)
            #print(MmChord.shape, candidate[idxT].shape)
            note = np.argmax(candidate[idxT][:130])
            if note == 129:
                if not np.argmax(query[max(0, idxT)][:130]) == 129:
                    score -= 1
                    continue
            elif note == 128:
                note = register
            else:
                note = note % 12
                register = note
            #print(note)
            #print(note)
            continueFlag = 0
            if MmChord[note] == 1:
                continue
            else:
                for idxt in range(-3, 1, 1):
                    melodyNote = np.argmax(query[max(0, idxT+idxt)][:130])
                    if melodyNote == 129:
                        continue
                    elif melodyNote == 128:
                        melodyNote = registerMelody
                    else:
                        melodyNote = melodyNote % 12
                        registerMelody = melodyNote
                    #print(melodyNote, note)
                    if melodyNote == note:
                        score += (3+idxt)
                        continueFlag = 1
                        #print('add')
                        break
                if continueFlag:
                    continue
                for idxt in range(1, 4, 1):
                    melodyNote = np.argmax(query[min(idxT+idxt, candidate.shape[0]-1)][:130])
                    if melodyNote == 129:
                        continue
                    elif melodyNote == 128:
                        melodyNote = registerMelody
                    else:
                        melodyNote = melodyNote % 12
                        registerMelody = melodyNote
                    #print(melodyNote, note)
                    if melodyNote == note:
                        score += (3-idxt)
                        continueFlag = 1
                        #print('add')
                        break
                if continueFlag:
                    continue
            score -= 1
            #print(score)
        newScore.append((search[i], score))
    #print(sorted(newScore, reverse=True, key=lambda score: score[1]))
    return [item[0] for item in sorted(newScore, reverse=True, key=lambda score: score[1])]

    
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('Task operating on', platform.system(), 'system, ', "CUDA enabled:", torch.cuda.is_available())
    if platform.system() == "Windows":
        data_save_root = './data_save_root'
    else:
        data_save_root = '/gpfsnyu/scratch/jz4807/musicalion_melody_batch_data/'
    music_save_root = './music_save_root'
    if not os.path.exists(music_save_root):
        os.makedirs(music_save_root)
    """
    batchData = np.empty((0, 32, 142))
    base_name = os.path.join(data_save_root, 'batchData_withChord_part')
    print('integrating data batch')
    for i in tqdm(range(10+1)):
        sub_batchData = np.load(base_name + str(i) + '.npy')
        batchData = np.concatenate((batchData, sub_batchData), axis=0)
    np.save(os.path.join(data_save_root, 'batchData_withChord.npy'), batchData)
    """
    print('loading model')
    time1 = time.time()
    if torch.cuda.is_available():
        model = VAE(roll_dims=130, hidden_dims=1024, rhythm_dims=3, condition_dims=12, z1_dims=128, z2_dims=128, n_step=32).cuda()
        model.load_state_dict(torch.load('../params/Mon Jul  6 22-51-32 2020/best_fitted_params.pt')['model_state_dict'])
    else:
        model = VAE(roll_dims=130, hidden_dims=1024, rhythm_dims=3, condition_dims=12, z1_dims=128, z2_dims=128, n_step=32)#.cuda()
        model.load_state_dict(torch.load('../params/Mon Jul  6 22-51-32 2020/best_fitted_params.pt', map_location='cpu')['model_state_dict'])
    model.eval()
    print('done, using time:', time.time() - time1)
    """
    RP = np.empty((0, 128))
    RZ = np.empty((0, 128))
    RC = np.empty((0, 32*12))
    TIV = np.empty((0, 32*12))
    print('loading dataset')
    time1 = time.time()
    if platform.system() == "Windows":
        dataloader = DataLoader(dataLoader(os.path.join(data_save_root, 'batchData_withChord_shifted.npy')), batch_size = 16, num_workers = 4, drop_last = False)
    else:
        dataloader = DataLoader(dataLoader(os.path.join(data_save_root, 'batchData_withChord_shifted.npy')), batch_size = 256, num_workers = 16, drop_last = False)
    print('done, using time:', time.time() - time1)
    for batch, c in tqdm(dataloader):
        encode_tensor = batch.float().cuda()
        c = c.float().cuda()
        recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s = model(encode_tensor, c)
        RP = np.concatenate((RP, dis1m.cpu().detach().numpy()), axis=0)
        RZ = np.concatenate((RZ, dis2m.cpu().detach().numpy()), axis=0)
        c = c.cpu().detach().numpy().reshape((c.shape[0], -1))
        #print(c.shape)
        RC = np.concatenate((RC, c), axis=0)
        TIV = np.concatenate((TIV, computeTIV(c)), axis=0)
    
    np.save(os.path.join(data_save_root, 'RP_withChord_shifted.npy'), RP)
    np.save(os.path.join(data_save_root, 'RZ_withChord_shifted.npy'), RZ)
    np.save(os.path.join(data_save_root, 'RC_withChord_shifted.npy'), RC)
    np.save(os.path.join(data_save_root, 'TIV_withChord_shifted.npy'), TIV)
    print(RP.shape, RZ.shape, RC.shape, TIV.shape)
    """
    processor = midi_interface_mono_and_chord()
    #batchTarget_, tempo = processor.load_single('./dummy_data_withChord/ssccm16.mid')
    #batchTarget_, tempo = processor.load_single('../nottingham_database/nottingham_midi_dual-track/2nd part for B music.mid')
    batchTarget_, tempo = processor.load_single('../nottingham_database/nottingham_midi_dual-track/Boggy Brays.mid')
    if torch.cuda.is_available():
        batchTarget = torch.from_numpy(batchTarget_).float().cuda()
    else:
        batchTarget = torch.from_numpy(batchTarget_).float()#.cuda()
    recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s = model(batchTarget[:, :, :130], batchTarget[:, :, 130:])
    #print(dis1m.shape)
    RP_Target = dis1m.cpu().detach().numpy()
    RZ_Target = dis2m.cpu().detach().numpy()
    RC_Target = batchTarget_[:, :, 130:].reshape((batchTarget_.shape[0], -1))
    #print(RC_Target[0])
    TIV_Target = computeTIV(RC_Target)
    #print(TIV_Target[0])
    assert(RP_Target.shape == RZ_Target.shape)
    print('loading comparing targets')
    time1 = time.time()
    RP = np.load(os.path.join(data_save_root, 'RP_withChord_shifted.npy'))
    RZ = np.load(os.path.join(data_save_root, 'RZ_withChord_shifted.npy'))
    RC = np.load(os.path.join(data_save_root, 'RC_withChord_shifted.npy'))
    TIV = np.load(os.path.join(data_save_root, 'TIV_withChord_shifted.npy'))
    batchData = np.load(os.path.join(data_save_root, 'batchData_withChord_shifted.npy'))
    print('done, using time:', time.time() - time1)

    converter = accompanySelection(save_root=data_save_root)
    converter.loadAuxilary('auxilary_withChord_shifted.txt')
    print('begin retrieving:')
    time1 = time.time()
    a=0
    for i in tqdm(range(RP_Target.shape[0])):
        #for i in range(8):
        #melody reconstruction with chord
        recon = processor.midiReconFromNumpy(batchTarget_[i], tempo)
        recon.write(os.path.join(music_save_root, 'recon' + str(i).zfill(3) + '.mid'))
        #search accorfing to pitch and chord measurement
        result_P = np.dot(RP, RP_Target[i])/(np.linalg.norm(RP, axis=1) * np.linalg.norm(RP_Target[i]) + 1e-10)
        #result_Z = np.dot(RZ, RZ_Target[i])/(np.linalg.norm(RZ, axis=1) * np.linalg.norm(RZ_Target[i]) + 1e-10)
        #result_C = np.dot(RC, RC_Target[i])/((np.linalg.norm(RC, axis=1) * np.linalg.norm(RC_Target[i])) + 1e-10)
        result_TIV = np.dot(TIV, TIV_Target[i])/((np.linalg.norm(TIV, axis=1) * np.linalg.norm(TIV_Target[i])) + 1e-10)
        result_PC = a * result_P + (1-a) * result_TIV
        candidates = result_PC.argsort()[::-1][0:100]
        #sort by edit distance over melody
        candidates_resorted = appearanceMatch(query=batchTarget_[i], search=candidates, batchData=batchData)[0:10]
        #sort by variance between units
        if i >= 1:
            if torch.cuda.is_available():
                bridgePart = torch.from_numpy(np.concatenate((batchTarget_[i-1][16:], batchTarget_[i][:16]), axis=0)[np.newaxis, :, :]).float().cuda()
            else:
                bridgePart = torch.from_numpy(np.concatenate((batchTarget_[i-1][16:], batchTarget_[i][:16]), axis=0)[np.newaxis, :, :]).float()#.cuda()
            recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s = model(bridgePart[:, :, :130], bridgePart[:, :, 130:])
            rZ = dis2m.cpu().detach().numpy()
            #print(rZ.shape)
            batchCandidates = batchData[candidates_resorted]
            last_candidate = batchData[chosen_candidate][16:][np.newaxis, :, :].repeat(len(candidates_resorted), axis=0)
            if torch.cuda.is_available():
                bridgeTarget = torch.from_numpy(np.concatenate((last_candidate, batchCandidates[:, :16, :]), axis=1)).float().cuda()
            else:
                bridgeTarget = torch.from_numpy(np.concatenate((last_candidate, batchCandidates[:, :16, :]), axis=1)).float()#.cuda()
            #print(bridgeTarget.shape)
            recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s = model(bridgeTarget[:, :, :130], bridgeTarget[:, :, 130:])
            rZ_target = dis2m.cpu().detach().numpy()
            #print(rZ_target.shape)
            result_Z = np.dot(rZ_target, rZ[0])/(np.linalg.norm(rZ_target, axis=1) * np.linalg.norm(rZ) + 1e-10)
            candidates_resorted = list(zip(candidates_resorted, result_Z))
            #print(sorted(candidates_resorted, reverse=True, key=lambda score: score[1]))
            candidates_finallySorted = [item[0] for item in sorted(candidates_resorted, reverse=True, key=lambda score: score[1])]
            chosen_candidate = candidates_finallySorted[0]
        else:
            chosen_candidate = candidates_resorted[0]
        
        retrive = converter.retriveRawMidi(chosen_candidate, 'batchData_withChord_shifted.npy')
        retrive.write(os.path.join(music_save_root, 'retrive' + str(i).zfill(3) + '.mid'))
        retrive_chord = processor.midiReconFromNumpy(batchData[chosen_candidate, :, :], tempo)
        retrive_chord.write(os.path.join(music_save_root, 'retrive_chord' + str(i).zfill(3) + '.mid'))
        """
        for idx, pickIdx in enumerate(candidates):
            retrive = converter.retriveRawMidi(pickIdx, 'batchData_withChord_shifted.npy')
            retrive.write(os.path.join(music_save_root, 'retrive' + str(i).zfill(3) + '_' + str(idx) + '.mid'))
            retrive_chord = processor.midiReconFromNumpy(batchData[pickIdx, :, :], tempo)
            retrive_chord.write(os.path.join(music_save_root, 'retrive_chord' + str(i).zfill(3) + '_' + str(idx) + '.mid'))
        """
    print('done, using time:', time.time() - time1)
    print('All finish!')