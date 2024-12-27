import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','no-latex'])
import glob
import vector
from core import Core

def validate_alignment(ID_sorted, ID_reference):
    is_valid = np.array_equal(ID_sorted, ID_reference[:len(ID_sorted)])
    if is_valid:
        print("Validation successful: IDs are aligned.")
    else:
        raise ValueError("Validation failed: IDs are not aligned.")

    return is_valid

def select_random_nu(arr0, arr1):
    temp_arr0 = np.array(arr0)
    temp_arr1 = np.array(arr1)
    index0 = np.random.randint(0, temp_arr0.shape[1], temp_arr0.shape[0])
    final_arr0 = np.array([temp_arr0[i][index0[i]] for i in range(len(index0))])
    final_arr1 = np.array([temp_arr1[i][index0[i]] for i in range(len(index0))])
    return final_arr0, final_arr1

def Get_truth_core(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    truth_tau_p = data['truth_tau_p']
    truth_tau_m = data['truth_tau_m']
    truth_ID = data['EventID'][:]
    truth_tau_p_constituent_1 = data['tau_p_constituent_1']
    truth_tau_p_constituent_2 = data['tau_p_constituent_2']
    truth_tau_p_constituent_3 = data['tau_p_constituent_3']
    truth_tau_m_constituent_1 = data['tau_m_constituent_1']
    truth_tau_m_constituent_2 = data['tau_m_constituent_2']
    truth_tau_m_constituent_3 = data['tau_m_constituent_3']
    truth_tau_p_child = truth_tau_p_constituent_1 + truth_tau_p_constituent_2 + truth_tau_p_constituent_3
    truth_tau_m_child = truth_tau_m_constituent_1 + truth_tau_m_constituent_2 + truth_tau_m_constituent_3
    truth_core = Core(truth_tau_p, truth_tau_m, truth_tau_p_child, truth_tau_m_child).analyze()
    return truth_core, truth_ID, truth_tau_p_child, truth_tau_m_child

def Get_truth_core_nominal(file_path):
    data = np.load(file_path)
    truth_tau_p = data['tau_p']
    truth_tau_m = data['tau_m']
    truth_ID = data['EventID'][:]
    truth_tau_p_child1 = data['tau_p_child1']
    truth_tau_p_child2 = data['tau_p_child2']
    truth_tau_m_child1 = data['tau_m_child1']
    truth_tau_m_child2 = data['tau_m_child2']
    truth_tau_p_child1 = vector.arr({
        "pt": truth_tau_p_child1[:, 0],
        "eta": truth_tau_p_child1[:, 1],
        "phi": truth_tau_p_child1[:, 2],
        "mass": truth_tau_p_child1[:, 3],
    })
    truth_tau_p_child2 = vector.arr({
        "pt": truth_tau_p_child2[:, 0],
        "eta": truth_tau_p_child2[:, 1],
        "phi": truth_tau_p_child2[:, 2],
        "mass": truth_tau_p_child2[:, 3],
    })
    truth_tau_m_child1 = vector.arr({
        "pt": truth_tau_m_child1[:, 0],
        "eta": truth_tau_m_child1[:, 1],
        "phi": truth_tau_m_child1[:, 2],
        "mass": truth_tau_m_child1[:, 3],
    })
    truth_tau_m_child2 = vector.arr({
        "pt": truth_tau_m_child2[:, 0],
        "eta": truth_tau_m_child2[:, 1],
        "phi": truth_tau_m_child2[:, 2],
        "mass": truth_tau_m_child2[:, 3],
    })
    truth_tau_p = vector.arr({
        "pt": truth_tau_p[:, 0],
        "eta": truth_tau_p[:, 1],
        "phi": truth_tau_p[:, 2],
        "mass": truth_tau_p[:, 3],
    })
    truth_tau_m = vector.arr({
        "pt": truth_tau_m[:, 0],
        "eta": truth_tau_m[:, 1],
        "phi": truth_tau_m[:, 2],
        "mass": truth_tau_m[:, 3],
    })
    truth_tau_p_child = truth_tau_p_child1 + truth_tau_p_child2
    truth_tau_m_child = truth_tau_m_child1 + truth_tau_m_child2
    truth_core = Core(truth_tau_p, truth_tau_m, truth_tau_p_child, truth_tau_m_child).analyze()
    return truth_core, truth_ID, truth_tau_p_child, truth_tau_m_child

def Get_recon_core(file_path, truth_ID, truth_tau_p_child, truth_tau_m_child):
    data = np.load(file_path)
    recon_nu_p = data['nu_p']
    recon_nu_m = data['nu_m']
    recon_ID = data['EventID'][:]
    # Rearrange the data according to the truth ID
    valid_IDs = np.intersect1d(truth_ID.astype(np.int64), recon_ID.astype(np.int64))
                
    # Create a mapping from the filtered valid_IDs
    sorted_indices = np.argsort(valid_IDs)
    order = np.searchsorted(valid_IDs, recon_ID, sorter=sorted_indices)

    # Filter out IDs that are not in valid_IDs
    valid_mask = np.isin(recon_ID, valid_IDs)

    # Apply the mask to all relevant arrays
    recon_nu_p = recon_nu_p[valid_mask]
    recon_nu_m = recon_nu_m[valid_mask]
    recon_ID = recon_ID[valid_mask]  # Update ID to match the filtered ones

    # Sort based on the new order
    order = np.argsort(np.searchsorted(valid_IDs, recon_ID))
    recon_nu_p = recon_nu_p[order]
    recon_nu_m = recon_nu_m[order]
    recon_ID = recon_ID[order]  # Update ID to match the sorted order
    validate_alignment(recon_ID, valid_IDs)
    
    recon_nu_p, recon_nu_m = select_random_nu(recon_nu_p, recon_nu_m)
    recon_nu_p = vector.arr({
        "px": recon_nu_p[:, 0],
        "py": recon_nu_p[:, 1],
        "pz": recon_nu_p[:, 2],
        "mass": np.zeros_like(recon_nu_p[:, 0]),
    })
    recon_nu_m = vector.arr({
        "px": recon_nu_m[:, 0],
        "py": recon_nu_m[:, 1],
        "pz": recon_nu_m[:, 2],
        "mass": np.zeros_like(recon_nu_m[:, 0]),
    })
    recon_tau_p = truth_tau_p_child + recon_nu_p
    recon_tau_m = truth_tau_m_child + recon_nu_m
    recon_core = Core(recon_tau_p, recon_tau_m, truth_tau_p_child, truth_tau_m_child).analyze()
    return recon_core

def hist_plot(truth_core,recon_core,valid_num):
    plot_keys = ["cos_theta_A_n", "cos_theta_A_r", "cos_theta_A_k", "cos_theta_B_n", "cos_theta_B_r", "cos_theta_B_k"]
    for key in plot_keys:
        fig, ax = plt.subplots()
        ax.hist(truth_core[key], bins=100, alpha=1, label="truth", histtype='step')
        ax.hist(recon_core[key], bins=100, alpha=0.5, label="recon", histtype='stepfilled')
        ax.set_title(key + f" syst: {valid_num}")
        ax.legend()
        print(f"Saving {key} in path: ./{valid_num}/{key}.png")
        plt.savefig(f"./{valid_num}/{key}.png", dpi=300)
        plt.close()
    plot_keys = ["cos_theta_n", "cos_theta_r", "cos_theta_k"]
    for key in plot_keys:
        fig, ax = plt.subplots()
        if key == "cos_theta_n":
            truth = truth_core["cos_theta_A_n"] * truth_core["cos_theta_B_n"]
            recon = recon_core["cos_theta_A_n"] * recon_core["cos_theta_B_n"]
        elif key == "cos_theta_r":
            truth = truth_core["cos_theta_A_r"] * truth_core["cos_theta_B_r"]
            recon = recon_core["cos_theta_A_r"] * recon_core["cos_theta_B_r"]
        else:
            truth = truth_core["cos_theta_A_k"] * truth_core["cos_theta_B_k"]
            recon = recon_core["cos_theta_A_k"] * recon_core["cos_theta_B_k"]
        ax.hist(truth, bins=100, alpha=1, label="truth", histtype='step')
        ax.hist(recon, bins=100, alpha=0.5, label="recon", histtype='stepfilled')
        ax.set_title(key + f" syst: {valid_num}")
        ax.legend()
        print(f"Saving {key} in path: ./{valid_num}/{key}.png")
        plt.savefig(f"./{valid_num}/{key}.png", dpi=300)
        plt.close()
        
def nomi_check():
    nomi_path = "/global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace.new/results/pi_pi/ml_export/OmniLearn_pi_pi_recon.npz"
    nomi_recon_path = "/global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace.new/results/pi_pi/ml_export/OmniLearn_pi_pi_recon_eval.npz"
    truth_core, truth_ID, truth_tau_p_child, truth_tau_m_child = Get_truth_core_nominal(nomi_path)
    recon_core = Get_recon_core(nomi_recon_path, truth_ID, truth_tau_p_child, truth_tau_m_child)
    hist_plot(truth_core, recon_core, "nominal")
        
def main():
    syst_path = "/global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace/results/pi_pi/systematics"
    finished_eval_list = glob.glob(f"{syst_path}/*/pi_pi_particles_eval.npz")
    check_list = []
    check_list_name = []
    for p in finished_eval_list:
        variation_path = os.path.dirname(p)
        variation_name = os.path.basename(variation_path)
        X = variation_name.split('.')[1]
        check_list.append(variation_path)
        check_list_name.append(X)
    for i in range(len(check_list)):
        if not os.path.exists(f"./{check_list_name[i]}"):
            print(f"Validating {check_list_name[i]}...")
            os.makedirs(f"./{check_list_name[i]}")
            truth_path = f"{check_list[i]}/pi_pi_particles.pkl"
            recon_path = f"{check_list[i]}/pi_pi_particles_eval.npz"
            truth_core, truth_ID, truth_tau_p_child, truth_tau_m_child = Get_truth_core(truth_path)
            recon_core = Get_recon_core(recon_path, truth_ID, truth_tau_p_child, truth_tau_m_child)
            hist_plot(truth_core, recon_core, check_list_name[i])
            print(f"Finish to validate {check_list_name[i]}.")
        else:
            print(f"{check_list_name[i]} has been validated.")
            
if __name__ == "__main__":
    main()
    # nomi_check()
    
    
    