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
    "pi_rho_MM": 14,
    "pi_rho_PM": 15,
    "pi_rho_MP": 16,
    "pi_rho_PP": 17,
    "lep_pi": 2,
    "lep_pi_MM": 18,
    "lep_pi_PM": 19,
    "lep_pi_MP": 20,
    "lep_pi_PP": 21,
    "lep_rho": 3,
    "lep_rho_MM": 22,
    "lep_rho_PM": 23,
    "lep_rho_MP": 24,
    "lep_rho_PP": 25,
    "QCD": 4,
    "rho_rho": 5,
    "rho_rho_MM": 26,
    "rho_rho_PM": 27,
    "rho_rho_MP": 28,
    "rho_rho_PP": 29,
    "tt": 6,
    "Wlnu": 7,
    "Wtaunu": 8,
    "Zll": 9
}
    
raw_path = "/global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace_20250211_sig_ext/results/*/systematics"
# raw_path = " /global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace_20250125_new_fakeresults/*/ml_export"
raw_file_list = glob.glob(f"{raw_path}/variation.*")
# raw_file_list = glob.glob(f"{raw_path}")
for i in range(len(raw_file_list)):
    merge_file_path = raw_file_list[i]
    merge_list = []
    final_merged_file = f"{merge_file_path}/merged.pkl"
    if not os.path.exists(final_merged_file):
        for file in glob.glob(f"{merge_file_path}/*.pkl"):
            if "boost" in file:
                continue
            print(f"Loading {file}...")
            with open(file, "rb") as f:
                data = pickle.load(f)
            samples_name = file.split("/")[-1].split(".")[0].replace("_particles", "")
            data["sample"] = name_map[samples_name] * np.ones(len(data["weight"]))
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
                    elif isinstance(data[key], list):
                        merge_list[key] += data[key]
                    elif isinstance(data[key], dict):
                        for subkey in data[key]:
                            if isinstance(data[key][subkey], np.ndarray):
                                merge_list[key][subkey] = np.concatenate((merge_list[key][subkey], data[key][subkey]))
                            if isinstance(data[key][subkey], list):
                                merge_list[key][subkey] += data[key][subkey]
        with open(final_merged_file, "wb") as f:
            pickle.dump(merge_list, f)
        print(f"Saved merged file to {final_merged_file}")
    
        
        
