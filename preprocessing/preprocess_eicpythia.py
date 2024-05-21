import h5py as h5
import os
import sys
import numpy as np
from optparse import OptionParser
# from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical
from sklearn.utils import shuffle

'''particles = [pT, eta, phi, PID, z]'''
'''NO JET dateset. These are all particles EIC events'''

PID_INDEX = 3

labels30 = {
    'Pythia_eP_10M.h5'
}

labels150 = {
    'Pythia_eP_10M.h5'
}

def Recenter(particles):
    '''particle features = [pT, eta, phi, PID, z]'''
    '''Particles for EIC event shouldn't be recentered'''

    print("\n\nWarning: Are you sure you want to RECENTER for EIC?")

    px = particles[:,:,0]*np.cos(particles[:,:,2])
    py = particles[:,:,0]*np.sin(particles[:,:,2])
    pz = particles[:,:,0]*np.sinh(particles[:,:,1])

    jet_px = np.sum(px,1)
    jet_py = np.sum(py,1)
    jet_pz = np.sum(pz,1)

    jet_pt = np.sqrt(jet_px**2 + jet_py**2)
    jet_phi = np.ma.arctan2(jet_py,jet_px).filled(0)
    jet_eta = np.ma.arcsinh(np.ma.divide(jet_pz,jet_pt).filled(0))

    particles[:,:,0]-= np.expand_dims(jet_eta,1)
    particles[:,:,1]-= np.expand_dims(jet_phi,1)


    return particles


def process(p):
    '''particles features: [pT, eta, phi, PID, z]'''

    mask = p[:,:,0]!=0.0  # mask is from pT==0
    new_p = np.zeros(shape=(p.shape[0],p.shape[1],4)) 
    
    #Modify the scattered electron pT for taking log later
    # p[:,0,0] = p[:,0,0]/np.max(p[:,0,0])
    p[:,0,0] = p[:,0,0]/49.0  # max pT of electron is ~48.0 GeV

    print("Line 55, before feature shuffle\n", p[1,:3])
    new_p[:,:,2] = np.ma.log(1.0 - p[:,:,0]).filled(0)  # pT
    new_p[:,:,0] = p[:,:,1]                             # eta
    new_p[:,:,1] = p[:,:,2]                             # phi
    new_p[:,:,3] = p[:,:,4]                             # z
    print("\n\nLine 60, AFTER feature shuffle\n", new_p[1,:3])
    # the main DataLoader class calculates disttanc/s, expecting eta and phi at 0 and 1
    # we add PID in the next line, z is now at index 3

    #convert PIDs to consecutive ints, then to category vectors
    pid = p[:,:,PID_INDEX]  #e-, pi+, K+  [11, 211, 321] -- May 2024

    # Create a mapping of unique values to indices
    unique_values = np.unique(pid)[1:]

    value_to_index = {value: index for index, value in enumerate(unique_values)}

    # Map the 'pid' values to their respective indices
    # pid_indices = np.vectorize(value_to_index.get)(pid)
    pid_indices = np.vectorize(lambda x: value_to_index.get(x, 0))(pid)

    # Convert indices to one-hot encoding
    pid_categorical = to_categorical(pid_indices)

    new_p = np.concatenate((new_p, pid_categorical), -1)
    print("\n\nLine 80, after PID CONCAT\n", new_p[1,:3])

    new_p = new_p*mask[:,:,None]

    print("\n\nLine 84, after MASK\n", new_p[1,:3])
    # print(new_p[:2])

    return new_p
    
def preprocess(path, labels, nevent_max=-1, npart_max=-1):

    train = {
        'data':[],
        'jet':[],
        'pid':[],
    }

    test = {
        'data':[],
        'jet':[],
        'pid':[],
    }

    val = {
        'data':[],
        'jet':[],
        'pid':[],
    }

    for label in labels:        
        with h5.File(os.path.join(path,label),"r") as h5f:

            ntotal = h5f['particles'][:nevent_max].shape[0]

            p = h5f['particles'][:nevent_max,:npart_max].astype(np.float32)
            print(np.shape(p))

            p = process(p)  # applies mask, saves real pid, shuffles feature indecies for training
            j = np.count_nonzero(p[:,:,2], axis=-1, keepdims=True)  #pT moved to index 2
            # For Pythia EIC, there are no jets, we generate particles 
            # for the event. Jet here will be event properties,
            # Most importantly multiplicity


            pid = np.ones(np.shape(p[: , :, PID_INDEX]))  
            # for EIC pythia, PID is a particle feature
            # not a feature to be conditioned on for generation,
            # PID is in the particle data structure, 'p'. pid here is dummy


            train['data'].append(p[:int(0.63*ntotal)])
            train['jet'].append(j[:int(0.63*ntotal)])
            train['pid'].append(pid[:int(0.63*ntotal)])

            val['data'].append(p[int(0.63*ntotal):int(0.7*ntotal)])
            val['jet'].append(j[int(0.63*ntotal):int(0.7*ntotal)])
            val['pid'].append(pid[int(0.63*ntotal):int(0.7*ntotal)])

            test['data'].append(p[int(0.7*ntotal):])
            test['jet'].append(j[int(0.7*ntotal):])
            test['pid'].append(pid[int(0.7*ntotal):])

    for key in train:        
        train[key] = np.concatenate(train[key],0)        
    for key in test:
        test[key] = np.concatenate(test[key],0)
    for key in val:
        val[key] = np.concatenate(val[key],0)


    for d in [train,test,val]:
        d['data'],d['jet'],d['pid'] = shuffle(d['data'],d['jet'],d['pid'])
        
    return train, val, test

if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--folder", type="string", default='/global/cfs/cdirs/m4662/data/', help="Folder containing input files")
    parser.add_option("--label", type="string", default='150', help="Which dataset to use")
    parser.add_option("--nevent_max", type="int", default='1_000_000', help="max number of events")
    parser.add_option("--npart_max", type="int", default='30', help="max number of particses per event")
    (flags, args) = parser.parse_args()

    if '150' in flags.label:
        label = labels150
    else:
        label = labels30
    
    train, val, test = preprocess(os.path.join(flags.folder, 'EIC_Pythia'),label, flags.nevent_max, flags.npart_max)

    with h5.File('{}/train_{}.h5'.format(os.path.join(flags.folder, 'EIC_Pythia'),flags.label), "w") as fh5:
        dset = fh5.create_dataset('data', data=train['data'])
        dset = fh5.create_dataset('pid', data=train['pid'])
        dset = fh5.create_dataset('jet', data=train['jet'])


    with h5.File('{}/test_{}.h5'.format(os.path.join(flags.folder, 'EIC_Pythia'),flags.label), "w") as fh5:
        dset = fh5.create_dataset('data', data=test['data'])
        dset = fh5.create_dataset('pid', data=test['pid'])
        dset = fh5.create_dataset('jet', data=test['jet'])


    with h5.File('{}/val_{}.h5'.format(os.path.join(flags.folder, 'EIC_Pythia'),flags.label), "w") as fh5:
        dset = fh5.create_dataset('data', data=val['data'])
        dset = fh5.create_dataset('pid', data=val['pid'])
        dset = fh5.create_dataset('jet', data=val['jet'])
