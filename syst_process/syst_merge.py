import numpy as np
import vector
import pickle
import glob
import os
import awkward as ak
name_map = {
    "pi_pi": 0,
    "pi_pi_MM": 10,
    "pi_pi_PM": 11,
    "pi_pi_MP": 12,
    "pi_pi_PP": 13,
    "pi_rho": 1,
    "lep_pi": 2,
    "lep_rho": 3,
    "QCD": 4,
    "rho_rho": 5,
    "tt": 6,
    "Wlnu": 7,
    "Wtaunu": 8,
    "Zll": 9
}
    
raw_path = "/global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace_12.14/results/pi_pi/systematics"
raw_file_list = glob.glob(f"{raw_path}/variation.*")
for i in range(len(raw_file_list)):
    merge_file_path = raw_file_list[i]
    merge_list = []
    final_merged_file = f"{merge_file_path}/merged.pkl"
    if not os.path.exists(final_merged_file):
        for file in glob.glob(f"{merge_file_path}/*.pkl"):
            print(f"Loading {file}...")
            with open(file, "rb") as f:
                data = pickle.load(f)
            data["sample"] = name_map[data["sample"]] * np.ones(len(data["weight"]))
            print(f"Merging {merge_file_path}...")
            if len(merge_list) == 0:
                merge_list = data
            else:
                for key in data:
                    if isinstance(data[key], vector.MomentumNumpy4D):
                        merge_list[key] = np.concatenate((np.array(merge_list[key]), np.array(data[key])))
                        merge_list[key] = vector.MomentumNumpy4D(merge_list[key])
                    elif isinstance(data[key], vector.MomentumNumpy2D):
                        merge_list[key] = np.concatenate((np.array(merge_list[key]), np.array(data[key])))
                        merge_list[key] = vector.MomentumNumpy2D(merge_list[key])
                    elif isinstance(data[key], np.ndarray):
                        merge_list[key] = np.concatenate((merge_list[key], data[key]))
                    elif isinstance(data[key], dict):
                        for subkey in data[key]:
                            if isinstance(data[key][subkey], np.ndarray):
                                merge_list[key][subkey] = np.concatenate((merge_list[key][subkey], data[key][subkey]))
                            if isinstance(data[key][subkey], list):
                                merge_list[key][subkey] += data[key][subkey]
        with open(final_merged_file, "wb") as f:
            pickle.dump(merge_list, f)
        print(f"Saved merged file to {final_merged_file}")
    
        
        