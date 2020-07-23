import numpy as np
import os
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from model import VAE
from midi_interface_mono import midi_interface_mono
from midi_interface_mono import accompanySelection
from tqdm import tqdm

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
    
if __name__ == '__main__':
    data_save_root = './data_save_root'
    music_save_root = './music_save_root'
    """
    batchData = np.empty((0, 32, 142))
    base_name = os.path.join(data_save_root, 'batchData_part')
    print('integrating data batch')
    for i in tqdm(range(12)):
        sub_batchData = np.load(base_name + str(i) + '.npy')
        batchData = np.concatenate((batchData, sub_batchData), axis=0)
    np.save(os.path.join(data_save_root, 'batchData.npy'), batchData)
    """
    model = VAE(roll_dims=130, hidden_dims=1024, rhythm_dims=3, condition_dims=12, z1_dims=128, z2_dims=128, n_step=32).cuda()
    model.load_state_dict(torch.load('../params/Mon Jul  6 22-51-32 2020/best_fitted_params.pt')['model_state_dict'])
    model.eval()
    """
    RP = torch.from_numpy(np.empty((0, 128))).float().cuda()
    RZ = torch.from_numpy(np.empty((0, 128))).float().cuda()
    dataloader = DataLoader(dataLoader(os.path.join(data_save_root, 'batchData.npy')), batch_size = 16, num_workers = 4, drop_last = False)
    for batch, c in tqdm(dataloader):
        encode_tensor = batch.float().cuda()
        c = c.float().cuda()
        recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s = model(encode_tensor, c)
        RP = torch.cat((RP, dis1m), dim=0)
        RZ = torch.cat((RZ, dis2m), dim=0)

    np.save(os.path.join(data_save_root, 'RP.npy'), RP.cpu().detach().numpy())
    np.save(os.path.join(data_save_root, 'RZ.npy'), RZ.cpu().detach().numpy())
    print(RP.shape, RZ.shape)
    """
    processor = midi_interface_mono()
    #batchTarget_, tempo = processor.load_single('../nottingham_database/nottingham_midi_dual-track/2nd part for B music.mid')
    batchTarget_, tempo = processor.load_single('../dule_track_trial/ssccm13.mid')
    batchTarget = torch.from_numpy(batchTarget_).float().cuda()
    recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s = model(batchTarget[:, :, :130], batchTarget[:, :, 130:])
    #print(dis1m.shape)
    RP_Target = dis1m.cpu().detach().numpy()
    RZ_Target = dis2m.cpu().detach().numpy()
    assert(RP_Target.shape == RZ_Target.shape)
    RP = np.load(os.path.join(data_save_root, 'RP.npy'))
    RZ = np.load(os.path.join(data_save_root, 'RZ.npy'))
    a=1
    converter = accompanySelection(save_root=data_save_root)
    converter.loadAuxilary()
    for i in range(RP_Target.shape[0]):
        result_P = np.dot(RP, RP_Target[i])/np.linalg.norm(RP, axis=1) * np.linalg.norm(RP_Target[i])
        result_Z = np.dot(RZ, RZ_Target[i])/np.linalg.norm(RZ, axis=1) * np.linalg.norm(RZ_Target[i])
        result_Total = a * result_P + (1-a) * result_Z
        pickIdx = np.argmax(result_Total)
        retrive = converter.retriveRawMidi(pickIdx)
        retrive.write(os.path.join(music_save_root, 'retrive' + str(i).zfill(3) + '.mid'))
        recon = processor.midiReconFromNumpy(batchTarget_[i], tempo)
        recon.write(os.path.join(music_save_root, 'recon' + str(i).zfill(3) + '.mid'))
