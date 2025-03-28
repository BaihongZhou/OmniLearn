import argparse
import os
import sys
from pathlib import Path

current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_file_path)

import numpy as np
import vector
import h5py as h5
import pickle
import vector

from configs.global_config import load_config, save_config
import configs.global_config as config


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def build_condition_vector(jets, taus, y_met, jet_pt_threshold: float = 10.0):
    jets = vector.arr({
        "pt": jets[..., 0],
        "eta": jets[..., 1],
        "phi": jets[..., 2],
        "energy": jets[..., 3]
    })
    taus = vector.arr({
        "pt": taus[..., 0],
        "eta": taus[..., 1],
        "phi": taus[..., 2],
        "energy": taus[..., 3]
    })
    met = vector.arr({
        "pt": y_met[:, 0],
        "phi": y_met[:, 1],
        "eta": np.zeros_like(y_met[:, 0]),
        "mass": np.zeros_like(y_met[:, 0])
    })

    tau1, tau2 = taus[:, 0], taus[:, 1]
    # jets = [jets[:, i] for i in range(jets.shape[1])]
    tau1_pt = tau1.pt
    tau2_pt = tau2.pt
    HT_Tau = tau1.pt + tau2.pt
    deltaR_tau = tau1.deltaR(tau2)

    HT_Jet = np.sum(jets.pt, axis=1)
    N_jets = np.sum(jets.pt > jet_pt_threshold, axis=1)
    mean_jet = jets.sum(axis=1) / (N_jets + 1)

    met_deltaphi_tau1 = met.deltaphi(tau1)
    met_deltaphi_tau2 = met.deltaphi(tau2)

    MET_sig_jet = met.pt / np.sqrt(HT_Jet + 1e-6)
    MET_sig_tau = met.pt / np.sqrt(HT_Tau + 1e-6)

    met_balance = met.pt / (tau1.pt + tau2.pt + 1e-6)
    met_balance_1 = met.pt / (tau1.pt + 1e-6)
    met_balance_2 = met.pt / (tau2.pt + 1e-6)

    all_inputs = {
        "met_pt": np.log1p(met.pt),
        "met_phi": met.phi,
        # "N_jets": N_jets,
        # "HT_Tau": np.log1p(HT_Tau),
        # "MET_sig_tau": np.log1p(MET_sig_tau),
        # "tau1_pt": tau1_pt,
        # "tau2_pt": tau2_pt,
        # "deltaR_tau": deltaR_tau,
        # "met_deltaphi_tau1": met_deltaphi_tau1,
        # "met_deltaphi_tau2": met_deltaphi_tau2,
        # "HT_Jet": np.log1p(HT_Jet),
        # "MET_sig_jet": np.log1p(MET_sig_jet),

        # "met_balance": met_balance,
        # "met_balance_1": met_balance_1,
        # "met_balance_2": met_balance_2,

        # "mean_jet_pt": mean_jet.pt,
        # "mean_jet_eta": mean_jet.eta,
        # "mean_jet_phi": mean_jet.phi,
        # "mean_jet_energy": mean_jet.energy,
    }

    def corr(x, y): return np.corrcoef(x, y)[0, 1]

    # nu1_pt = nu[:, 0]
    # nu2_pt = nu[:, 3]

    selected_inputs = {
        name: arr for name, arr in all_inputs.items()
        # if name in ['met_pt', 'met_phi', 'N_jets'] or
        #    max(abs(corr(nu1_pt, arr)), abs(corr(nu2_pt, arr))) >= 0.2
    }

    condition_vector = np.stack([v for v in selected_inputs.values()], axis=1)
    all_inputs_name = list(selected_inputs.keys())

    return condition_vector, all_inputs_name


def calculate_correlations(cond_vec, nu, input_names):
    truth_nu1_pt = nu[:, 0]
    truth_nu2_pt = nu[:, 4]

    names = input_names

    print("### Pearson Correlation with ν1 pT:")
    for i, name in enumerate(names):
        corr = np.corrcoef(truth_nu1_pt, cond_vec[:, i])[0, 1]
        print(f"  {name:20}: {corr:.3f}")

    print("\n### Pearson Correlation with ν2 pT:")
    for i, name in enumerate(names):
        corr = np.corrcoef(truth_nu2_pt, cond_vec[:, i])[0, 1]
        print(f"  {name:20}: {corr:.3f}")


def build_extra_targets(
        data,
        x1_name: str = 'Tau1',
        x2_name: str = 'Tau2',
        met_name: str = 'met',
        truth_name: str = 'truth_TauTau'
):
    x1 = vector.arr({
        "pt": data[x1_name][:, 0],
        "eta": data[x1_name][:, 1],
        "phi": data[x1_name][:, 2],
        "mass": data[x1_name][:, 3]
    }).to_pxpypzenergy()
    x2 = vector.arr({
        "pt": data[x2_name][:, 0],
        "eta": data[x2_name][:, 1],
        "phi": data[x2_name][:, 2],
        "mass": data[x2_name][:, 3]
    }).to_pxpypzenergy()
    met = vector.arr({
        "pt": data[met_name][:, 0],
        "phi": data[met_name][:, 1],
        "eta": np.zeros_like(data[met_name][:, 0]),
        "mass": np.zeros_like(data[met_name][:, 0])
    }).to_pxpypzenergy()
    truth_sum = vector.arr({
        "pt": data[truth_name][:, 0],
        "eta": data[truth_name][:, 1],
        "phi": data[truth_name][:, 2],
        "mass": data[truth_name][:, 3]
    }).to_pxpypzenergy()

    # sum_metx1x2 = met + x1 + x2
    sum_metx1x2 = x1 + x2

    diff = np.array([
        truth_sum.px - sum_metx1x2.px,
        truth_sum.py - sum_metx1x2.py,
        truth_sum.pz - sum_metx1x2.pz,
        truth_sum.energy - sum_metx1x2.energy
    ])

    # diff = np.array([
    #     truth_sum.pt,
    #     truth_sum.eta,
    #     truth_sum.phi,
    #     truth_sum.energy,
    # ])

    return diff.T


def process(
        data_path: Path, save_path: Path, save_tag: str,
        sample_lists: list[str], features: dict, train_ratio: float = 0.8,
        overwrite: bool = False,
        for_training: bool = True,
):
    def save_hdf5(file_path, col_names, data, mode="train"):
        with h5.File(file_path, "w") as f:
            for name, arr in zip(col_names, data):
                if name == "RawFile":  # Check if the array is of string type
                    f.create_dataset(name, data=arr, dtype=h5.string_dtype())
                else:
                    f.create_dataset(name, data=arr)
        print(f"{mode.capitalize()} data saved to {file_path}")

    if for_training:
        train_file = save_path / f"{save_tag}_train.hdf5"
        test_file = save_path / f"{save_tag}_test.hdf5"

        if train_file.exists() and test_file.exists() and not overwrite:
            print(f"Skipping processing data: {train_file} and {test_file} already exists")
            return
    else:
        train_file = save_path / f"{save_tag}_evaluation.hdf5"
        test_file = None
        if train_file.exists() and not overwrite:
            print(f"Skipping processing data: {train_file} already exists")
            return

    X = {}
    nu = {}
    y = []
    Extra = []
    raw_file = []
    weight = []
    for file_path in sample_lists:
        sample_file_path = data_path / file_path
        with open(sample_file_path, 'rb') as f:
            data = pickle.load(f)

            n_events = len(data["eventNumber"])
            raw_file.extend(np.array([sample_file_path.stem] * n_events, dtype=object))

            print(f"{n_events} events loaded from {sample_file_path}")

            for category, attributes in features.items():
                if category == 'extra':
                    original_extra = np.column_stack([data[ext] for ext in attributes])
                    Extra.append(original_extra)
                    continue

                if category == 'weight':
                    event_weight = data[attributes['event']]
                    sample_weight = np.ones_like(event_weight) * attributes['sample'].get(sample_file_path.stem, 0)

                    weight.append(np.column_stack([event_weight, sample_weight]))
                    continue

                for particle in attributes.get('particles', []):
                    # Prepare Kinematics
                    pt, eta, phi, mass = data[particle][:, 0], np.zeros_like(data[particle][:, 0]), None, None

                    if category == 'met':
                        phi, mass = data[particle][:, 1], np.zeros_like(pt)
                    # elif category == "nu":
                    #     eta, phi, mass = data[particle][:, 1], data[particle][:, 2], np.zeros_like(pt)
                    else:
                        eta, phi, mass = data[particle][:, 1], data[particle][:, 2], data[particle][:, 3]

                    kin_par = vector.arr(dict(pt=pt, eta=eta, phi=phi, mass=mass)).to_ptphietaenergy()

                    if category == 'met':
                        y.append(np.column_stack((pt, phi)))
                        continue

                    if category in ["nu"]:
                        # kin_par = kin_par.to_pxpypzenergy()
                        pass

                    # Collect kinematic features
                    particle_features = [kin_par[kin][:, None] for kin in attributes.get('kinematic', [])]

                    if category == "nu":
                        nu.setdefault(particle, []).append(np.column_stack(particle_features))
                        continue

                    # Collect extra features
                    for extra in attributes.get('extra', []):
                        extra_par = data[particle + '_' + extra] if category == 'tau_vis' else np.zeros_like(pt)
                        particle_features.append(extra_par[:, None])

                    if attributes.get('padding', 0) > 0:
                        particle_features.append(np.zeros((len(pt), attributes['padding'])))

                    # Assign labels
                    label = attributes.get('label', 1 if category == 'tau_vis' else 0)
                    particle_features.append(np.full_like(pt[:, None], label))

                    X.setdefault(particle, []).append(np.hstack(particle_features))

            diff = build_extra_targets(data)
            nu.setdefault('diff', []).append(diff)

    X = np.concatenate([np.vstack(parts)[:, None, :] for parts in X.values()], axis=1)
    nu = np.concatenate([np.vstack(parts) for parts in nu.values()], axis=1)
    y = np.vstack(y)
    Extra = np.vstack(Extra)
    weight = np.vstack(weight)

    if not features['jet'].get('drop', False):
        jet_start_index = len(features['tau_vis']['particles'])
        if features['jet'].get('merge_jet', False):
            mask = X[:, jet_start_index:, 0] > 0
            valid_counts = np.maximum(mask.sum(axis=1), 1)
            sum_pt = np.where(mask, X[:, jet_start_index:, 0], 0).sum(axis=1)
            sum_eta = np.where(mask, X[:, jet_start_index:, 1], 0).sum(axis=1)
            sum_phi = np.where(mask, X[:, jet_start_index:, 2], 0).sum(axis=1)
            sum_E = np.where(mask, X[:, jet_start_index:, 3], 0).sum(axis=1)

            X[:, jet_start_index, :4] = np.stack([
                sum_pt / valid_counts,
                sum_eta / valid_counts,
                sum_phi / valid_counts,
                sum_E / valid_counts
            ], axis=-1)
            X = X[:, :jet_start_index + 1]

        # calculating MET-jet related variables for conditioning
        # Extract jets from X
        # Assuming first 2 particles = tau_vis → jets start from index 2
        jets_X = X[:, jet_start_index:, :4]  # shape: (n_events, n_jets, 4)
        tau_X = X[:, :jet_start_index, :4]  # shape: (n_events, n_tau_vis, 4)
        y, input_names = build_condition_vector(jets=jets_X, taus=tau_X, y_met=y)
        calculate_correlations(y, nu, input_names)
    else:
        X = X[:, :len(features['tau_vis']['particles'])]
        calculate_correlations(y, nu, ["met_pt", "met_phi"])

    # convert pt and energy to log(x + 1)
    X[:, :, 0] = np.log1p(X[:, :, 0])  # pt
    X[:, :, 3] = np.log1p(X[:, :, 3])  # energy
    nu[:, 0] = np.log1p(nu[:, 0])  # nu1 pt
    nu[:, 3] = np.log1p(nu[:, 3])  # nu1 energy
    nu[:, 4] = np.log1p(nu[:, 4])  # nu2 pt
    nu[:, 7] = np.log1p(nu[:, 7])  # nu2 energy
    nu[:, 8] = signed_log1p(nu[:, 8])  # truth_tautau - (tau1 + tau2) px
    nu[:, 9] = signed_log1p(nu[:, 9])  # truth_tautau - (tau1 + tau2) py
    nu[:, 10] = signed_log1p(nu[:, 10])  # truth_tautau - (tau1 + tau2) pz
    nu[:, 11] = signed_log1p(nu[:, 11])  # truth_tautau - (tau1 + tau2) energy

    if for_training:
        # Indices to compute mean and std
        selected_indices = [0, 3]
        particle_mean = np.zeros(X.shape[2])
        particle_std = np.ones(X.shape[2])
        for idx in selected_indices:
            particle_mean[idx] = np.mean(X[:, :, idx], axis=(0, 1), where=X[:, :, idx] != 0)
            particle_std[idx] = np.std(X[:, :, idx], axis=(0, 1), where=X[:, :, idx] != 0)

        selected_indices = [0, 3, 4, 7] + [8, 9, 10, 11]
        nu_mean = np.zeros(nu.shape[1])
        nu_std = np.ones(nu.shape[1])
        for idx in selected_indices:
            nu_mean[idx] = np.mean(nu[:, idx], axis=0, where=nu[:, idx] != 0)
            nu_std[idx] = np.std(nu[:, idx], axis=0, where=nu[:, idx] != 0)

        norm_dict = {
            "particle": {
                "mean": particle_mean.tolist(),
                "std": particle_std.tolist()
            },
            "nu": {
                "mean": nu_mean.tolist(),
                "std": nu_std.tolist(),
            }
        }

        raw_file = np.array(raw_file)

        indices = np.random.permutation(X.shape[0])  # Shuffle indices
        X, nu, y, Extra, Weight, Raw_File = (
            arr[indices] for arr in (X, nu, y, Extra, weight, raw_file)
        )  # Apply shuffle

        split_idx = int(len(X) * train_ratio)  # Compute split index
        train_data = (
            X[:split_idx], nu[:split_idx], y[:split_idx], Extra[:split_idx], Weight[:split_idx],
            list(Raw_File[:split_idx]))
        test_data = (
            X[split_idx:], nu[split_idx:], y[split_idx:], Extra[split_idx:], Weight[split_idx:],
            list(Raw_File[split_idx:]))

        # Save train & test data
        save_hdf5(train_file, ["X", "nu", "Condition", "Extra", "Weight", "RawFile"], train_data, mode="train")
        save_hdf5(test_file, ["X", "nu", "Condition", "Extra", "Weight", "RawFile"], test_data, mode="test")
    else:
        norm_dict = {}
        save_hdf5(
            train_file,
            ["X", "nu", "Condition", "Extra", "RawFile"],
            (X, nu, y, Extra, raw_file),
            mode="evaluation"
        )

    return norm_dict


def main():
    parser = argparse.ArgumentParser(description="Train the PET model on JetNet datasets.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")

    args = parser.parse_args()

    load_config(config_file=args.config)

    np.random.seed(42)

    train_norm_dict = None

    for data_type, cfg in config.cfg['preprocess'].items():
        data_path = Path(cfg['raw_folder']).absolute()
        save_path = Path(cfg['processed_folder']).absolute()
        save_tag = config.cfg['sample']['tag']
        if not save_path.exists():
            save_path.mkdir(parents=True)

        print("Start processing the data")
        sample_lists = cfg['sample_list']
        features = config.cfg['features']

        norm_dict = process(
            data_path=data_path, save_path=save_path, save_tag=save_tag,
            sample_lists=sample_lists, features=features, train_ratio=0.8,
            overwrite=cfg['overwrite'],
            for_training=cfg.get('for_training', False),
        )

        if norm_dict:
            if cfg.get('for_training', False):
                train_norm_dict = norm_dict

            config.cfg['sample']['normalization'] = train_norm_dict if train_norm_dict else norm_dict
            cfg['raw_folder'] = str(data_path)
            cfg['processed_folder'] = str(save_path)
            config.cfg['sample']['base_folder'] = str(save_path)
            config.cfg['evaluation']['processed_folder'] = str(save_path)

            save_config(save_path=save_path, save_tag=f"{save_tag}_preprocess")


if __name__ == '__main__':
    main()
