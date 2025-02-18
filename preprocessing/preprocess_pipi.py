import numpy as np
import glob
import h5py as h5
import os
from core import Core
import matplotlib.pyplot as plt
import pickle
name_map = {
    "pi_pi": [1, 0, 0, 0, 0, 0],
    "pi_pi_MM": [1, 0, 0, 0, 0, 0],
    "pi_pi_PM": [1, 0, 0, 0, 0, 0],
    "pi_pi_MP": [1, 0, 0, 0, 0, 0],
    "pi_pi_PP": [1, 0, 0, 0, 0, 0],
    "pi_rho": [0, 0, 1, 0, 0, 0],
    "lep_pi": [0, 1, 0, 0, 0, 0],
    "lep_rho": [0, 0, 0, 1, 0, 0],
    "QCD": [0, 0, 0, 0, 0, 1],
    "rho_rho": [0, 0, 0, 0, 1, 0],
    "tt": [0, 0, 0, 0, 0, 1],
    "Wlnu": [0, 0, 0, 0, 0, 1],
    "Wtaunu": [0, 0, 0, 0, 0, 1],
    "Zll": [0, 0, 0, 0, 0, 1],
}

def process(file_path, save_name):
    import vector
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    jet_1 = data['jet_1'][:]
    jet_1 = vector.array({"pt": jet_1[:,0], "eta": jet_1[:,1], "phi": jet_1[:,2], "mass": jet_1[:,3]})
    jet_2 = data['jet_2'][:]
    jet_2 = vector.array({"pt": jet_2[:,0], "eta": jet_2[:,1], "phi": jet_2[:,2], "mass": jet_2[:,3]})
    jet_3 = data['jet_3'][:]
    jet_3 = vector.array({"pt": jet_3[:,0], "eta": jet_3[:,1], "phi": jet_3[:,2], "mass": jet_3[:,3]})
    MET = data['MET'][:]
    MET = vector.array({"pt": MET[:,0], "phi": MET[:,1], "eta": np.zeros(MET.shape[0]), "mass": np.zeros(MET.shape[0])})
    EventID_temp = data['EventID'][:]
    print("We have ", EventID_temp.shape[0], " events")
    Type_temp = np.ones((EventID_temp.shape[0], 6))
    tau_p_child1 = data['tau_p_child1'][:]
    tau_p_child1 = vector.array({"pt": tau_p_child1[:,0], "eta": tau_p_child1[:,1], "phi": tau_p_child1[:,2], "mass": tau_p_child1[:,3]})
    tau_p_child2 = data['tau_p_child2'][:]
    tau_p_child2 = vector.array({"pt": tau_p_child2[:,0], "eta": tau_p_child2[:,1], "phi": tau_p_child2[:,2], "mass": tau_p_child2[:,3]})
    tau_m_child1 = data['tau_m_child1'][:]
    tau_m_child1 = vector.array({"pt": tau_m_child1[:,0], "eta": tau_m_child1[:,1], "phi": tau_m_child1[:,2], "mass": tau_m_child1[:,3]})
    tau_m_child2 = data['tau_m_child2'][:]
    tau_m_child2 = vector.array({"pt": tau_m_child2[:,0], "eta": tau_m_child2[:,1], "phi": tau_m_child2[:,2], "mass": tau_m_child2[:,3]})
    tau_p_child1_charge = data['tau_p_child1_charge'][:]
    tau_p_child1_is_el = data['tau_p_child1_is_el'][:]
    tau_p_child1_is_mu = data['tau_p_child1_is_mu'][:]
    tau_p_child1_is_charged_pion = data['tau_p_child1_is_charged_pion'][:]
    tau_p_child1_is_neutral_part = data['tau_p_child1_is_neutral_part'][:]
    tau_p_child2_charge = data['tau_p_child2_charge'][:]
    tau_p_child2_is_el = data['tau_p_child2_is_el'][:]
    tau_p_child2_is_mu = data['tau_p_child2_is_mu'][:]
    tau_p_child2_is_charged_pion = data['tau_p_child2_is_charged_pion'][:]
    tau_p_child2_is_neutral_part = data['tau_p_child2_is_neutral_part'][:]
    tau_m_child1_charge = data['tau_m_child1_charge'][:]
    tau_m_child1_is_el = data['tau_m_child1_is_el'][:]
    tau_m_child1_is_mu = data['tau_m_child1_is_mu'][:]
    tau_m_child1_is_charged_pion = data['tau_m_child1_is_charged_pion'][:]
    tau_m_child1_is_neutral_part = data['tau_m_child1_is_neutral_part'][:]
    tau_m_child2_charge = data['tau_m_child2_charge'][:]
    tau_m_child2_is_el = data['tau_m_child2_is_el'][:]
    tau_m_child2_is_mu = data['tau_m_child2_is_mu'][:]
    tau_m_child2_is_charged_pion = data['tau_m_child2_is_charged_pion'][:]
    tau_m_child2_is_neutral_part = data['tau_m_child2_is_neutral_part'][:]
    nu_p = data['truth_nu_p'][:]
    nu_p = vector.array({"pt": nu_p[:,0], "eta": nu_p[:,1], "phi": nu_p[:,2], "mass": np.zeros(nu_p.shape[0])})
    nu_m = data['truth_nu_m'][:]
    nu_m = vector.array({"pt": nu_m[:,0], "eta": nu_m[:,1], "phi": nu_m[:,2], "mass": np.zeros(nu_m.shape[0])})
    jet_1_temp = np.stack([jet_1.px, jet_1.py, jet_1.pz, jet_1.E, np.zeros_like(jet_1.pt), np.zeros_like(jet_1.pt), np.zeros_like(jet_1.pt), np.zeros_like(jet_1.pt), np.zeros_like(jet_1.pt)], -1) 
    jet_2_temp = np.stack([jet_2.px, jet_2.py, jet_2.pz, jet_2.E, np.zeros_like(jet_2.pt), np.zeros_like(jet_2.pt), np.zeros_like(jet_2.pt), np.zeros_like(jet_2.pt), np.zeros_like(jet_2.pt)], -1)
    jet_3_temp = np.stack([jet_3.px, jet_3.py, jet_3.pz, jet_3.E, np.zeros_like(jet_3.pt), np.zeros_like(jet_3.pt), np.zeros_like(jet_3.pt), np.zeros_like(jet_3.pt), np.zeros_like(jet_3.pt)], -1)
    MET_temp = np.stack([MET.pt, MET.phi], -1)
    tau_p_child1_temp = np.stack([tau_p_child1.px, tau_p_child1.py, tau_p_child1.pz, tau_p_child1.E, tau_p_child1_charge, tau_p_child1_is_el, tau_p_child1_is_mu, tau_p_child1_is_charged_pion, tau_p_child1_is_neutral_part], -1)
    tau_p_child2_temp = np.stack([tau_p_child2.px, tau_p_child2.py, tau_p_child2.pz, tau_p_child2.E, tau_p_child2_charge, tau_p_child2_is_el, tau_p_child2_is_mu, tau_p_child2_is_charged_pion, tau_p_child2_is_neutral_part], -1)
    tau_m_child1_temp = np.stack([tau_m_child1.px, tau_m_child1.py, tau_m_child1.pz, tau_m_child1.E, tau_m_child1_charge, tau_m_child1_is_el, tau_m_child1_is_mu, tau_m_child1_is_charged_pion, tau_m_child1_is_neutral_part], -1)
    tau_m_child2_temp = np.stack([tau_m_child2.px, tau_m_child2.py, tau_m_child2.pz, tau_m_child2.E, tau_m_child2_charge, tau_m_child2_is_el, tau_m_child2_is_mu, tau_m_child2_is_charged_pion, tau_m_child2_is_neutral_part], -1)
    nu_p_temp = np.stack([nu_p.px, nu_p.py, nu_p.pz], -1)
    nu_m_temp = np.stack([nu_m.px, nu_m.py, nu_m.pz], -1)
    if 'test' in save_name:
        if os.path.exists(save_name.replace('_test.hdf5', '_train.hdf5.npy')):
            random_index = np.load(save_name.replace('_test.hdf5', '_train.hdf5.npy'))
    if 'val' in save_name:
        if os.path.exists(save_name.replace('_val.hdf5', '_train.hdf5.npy')):
            random_index = np.load(save_name.replace('_val.hdf5', '_train.hdf5.npy'))
    if 'train' in save_name:  
        random_index = np.random.permutation(np.arange(len(Type_temp)))
        np.save(save_name + '.npy', random_index)
    jet_1 = jet_1_temp[random_index]
    jet_2 = jet_2_temp[random_index]
    jet_3 = jet_3_temp[random_index]
    MET = MET_temp[random_index]
    Type = Type_temp[random_index]
    EventID = EventID_temp[random_index]
    tau_p_child1 = tau_p_child1_temp[random_index]
    tau_p_child2 = tau_p_child2_temp[random_index]
    tau_m_child1 = tau_m_child1_temp[random_index]
    tau_m_child2 = tau_m_child2_temp[random_index]
    nu_p = nu_p_temp[random_index]
    nu_m = nu_m_temp[random_index]
    X = np.concatenate((tau_p_child1.reshape(tau_p_child1.shape[0], 1, 9), tau_p_child2.reshape(tau_p_child1.shape[0], 1, 9), tau_m_child1.reshape(tau_p_child1.shape[0], 1, 9), tau_m_child2.reshape(tau_p_child1.shape[0], 1, 9)), axis=1)
    X = np.concatenate((X, jet_1.reshape(jet_1.shape[0], 1, 9), jet_2.reshape(jet_2.shape[0], 1, 9), jet_3.reshape(jet_3.shape[0], 1, 9)), axis=1)
    nu = np.concatenate((nu_p, nu_m), axis=1)
    mean_X = np.mean(X, axis=(0, 1), where=X!=0)
    std_X = np.std(X, axis=(0, 1), where=X!=0)
    mean_nu = np.mean(nu, axis=(0), where=nu!=0)
    std_nu = np.std(nu, axis=(0), where=nu!=0)
    temp_MET = np.log(MET[:,0])
    mean_MET = np.mean(temp_MET, axis=(0), where=temp_MET!=0)
    std_MET = np.std(temp_MET, axis=(0), where=temp_MET!=0)
    if 'train' in save_name:
        X_final = X[:int(X.shape[0]*0.8)]
        nu_final = nu[:int(nu.shape[0]*0.8)]
        MET_final = MET[:int(MET.shape[0]*0.8)]
        type_final = Type[:int(Type.shape[0]*0.8)]
        mean_X_final = mean_X
        std_X_final = std_X
        mean_nu_final = mean_nu
        std_nu_final = std_nu
        with h5.File(save_name, 'w') as f:
            f.create_dataset('X', data=X_final)
            f.create_dataset('nu', data=nu_final)
            f.create_dataset('MET', data=MET_final)
            f.create_dataset('label', data=type_final)
            f.create_dataset('mean_X', data=mean_X_final)
            f.create_dataset('std_X', data=std_X_final)
            f.create_dataset('mean_nu', data=mean_nu_final)
            f.create_dataset('std_nu', data=std_nu_final)
            f.create_dataset('mean_MET', data=mean_MET)
            f.create_dataset('std_MET', data=std_MET)
    elif 'test' in save_name:
        X_final = X[int(X.shape[0]*0.8):int(X.shape[0]*0.9)]
        nu_final = nu[int(nu.shape[0]*0.8):int(nu.shape[0]*0.9)]
        MET_final = MET[int(MET.shape[0]*0.8):int(MET.shape[0]*0.9)]
        Type_final = Type[int(Type.shape[0]*0.8):int(Type.shape[0]*0.9)]
        mean_X_final = mean_X
        std_X_final = std_X
        mean_nu_final = mean_nu
        std_nu_final = std_nu
        with h5.File(save_name, 'w') as f:
            f.create_dataset('X', data=X_final)
            f.create_dataset('nu', data=nu_final)
            f.create_dataset('MET', data=MET_final)
            f.create_dataset('label', data=Type_final)
            f.create_dataset('mean_X', data=mean_X_final)
            f.create_dataset('std_X', data=std_X_final)
            f.create_dataset('mean_nu', data=mean_nu_final)
            f.create_dataset('std_nu', data=std_nu_final)
            f.create_dataset('mean_MET', data=mean_MET)
            f.create_dataset('std_MET', data=std_MET)
    elif 'val' in save_name:
        X_final = X[int(X.shape[0]*0.9):]
        nu_final = nu[int(nu.shape[0]*0.9):]
        MET_final = MET[int(MET.shape[0]*0.9):]
        Type_final = Type[int(Type.shape[0]*0.9):]
        mean_X_final = mean_X
        std_X_final = std_X
        mean_nu_final = mean_nu
        std_nu_final = std_nu
        with h5.File(save_name, 'w') as f:
            f.create_dataset('X', data=X_final)
            f.create_dataset('nu', data=nu_final)
            f.create_dataset('MET', data=MET_final)
            f.create_dataset('label', data=Type_final)
            f.create_dataset('mean_X', data=mean_X_final)
            f.create_dataset('std_X', data=std_X_final)
            f.create_dataset('mean_nu', data=mean_nu_final)
            f.create_dataset('std_nu', data=std_nu_final)
            f.create_dataset('mean_MET', data=mean_MET)
            f.create_dataset('std_MET', data=std_MET)
            
def main():
    data_path = '/global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace_20250114/results'
    save_path = '/pscratch/sd/b/baihong/data/NumpyData/baseline'
    print("Start processing the data")
    sample_lists = ['pi_pi/ml_export/pi_pi_recon_boost.pkl']
    for f in sample_lists:
        file_path = data_path + '/' + f
        file_name = file_path.split('/')[-1].replace('_boost.pkl', '_total')
        if "lep" in file_name:
            real_par = f.split('/')[0].split('_')[0]
            file_name = file_name.replace('lep', real_par)
        train_name = save_path + '/' + file_name + '_train.hdf5'
        test_name = save_path + '/' + file_name + '_test.hdf5'
        val_name = save_path + '/' + file_name + '_val.hdf5'
        if os.path.exists(train_name):
            print(f"Processing {f} train.hdf5 done")
        else:
            process(file_path, train_name)
            print(f"Processing {f} train.hdf5 done")
        if os.path.exists(test_name):
            print(f"Processing {f} test.hdf5 done")
        else:
            process(file_path, test_name)
            print(f"Processing {f} test.hdf5 done")
        if os.path.exists(val_name):
            print(f"Processing {f} val.hdf5 done")
        else:
            process(file_path, val_name)
            print(f"Processing {f} val.hdf5 done")
    

if __name__ == '__main__':
    main()