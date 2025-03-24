import argparse
import os
import sys
from pathlib import Path

current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_file_path)

import numpy as np
import glob
import h5py as h5
import pickle
import vector

from configs.global_config import load_config, save_config
import configs.global_config as config


def build_condition_vector(jets, y_met, jet_pt_threshold: float = 10.0):
    # X: (n_events, n_particles, n_features)
    # y_met: (n_events, 2) → [MET_pt, MET_phi]

    # Extract jets assuming first 2 particles are taus
    jet_pts = jets[..., 0]
    jet_phis = jets[..., 2]

    # Sort jets by pt in descending order
    sorted_indices = np.argsort(-jet_pts, axis=1)  # descending
    sorted_pts = np.take_along_axis(jet_pts, sorted_indices, axis=1)
    sorted_phis = np.take_along_axis(jet_phis, sorted_indices, axis=1)

    # Recompute observables with sorted jets
    HT = np.sum(sorted_pts, axis=1, keepdims=True)
    N_jets = np.sum(sorted_pts > jet_pt_threshold, axis=1, keepdims=True)

    # Δϕ between MET and leading jet (after sorting)
    met_pt, met_phi = y_met[:, 0], y_met[:, 1]
    delta_phi = np.abs(sorted_phis[:, 0] - met_phi)
    delta_phi = np.mod(delta_phi + np.pi, 2 * np.pi) - np.pi
    delta_phi = np.abs(delta_phi)[:, None]

    # MET cartesian
    met_px = met_pt * np.cos(met_phi)
    met_py = met_pt * np.sin(met_phi)

    # MET significance
    met_sig = met_pt / np.sqrt(HT[:, 0] + 1e-6)
    met_sig = met_sig[:, None]

    # Final conditioning vector
    condition_vector = np.hstack([
        met_px[:, None],
        met_py[:, None],
        HT,
        N_jets,
        delta_phi,
        met_sig
    ])

    return condition_vector


def process(
        data_path: Path, save_path: Path, save_tag: str,
        sample_lists: list[str], features: dict, train_ratio: float = 0.8,
        overwrite: bool = False,
        for_training: bool = True,
):
    def save_hdf5(file_path, col_names, data, mode="train"):
        with h5.File(file_path, "w") as f:
            for name, arr in zip(col_names, data):
                if isinstance(arr, list):  # Check if the array is of string type
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
    for file_path in sample_lists:
        sample_file_path = data_path / file_path
        with open(sample_file_path, 'rb') as f:
            data = pickle.load(f)

            n_events = len(data["eventNumber"])
            raw_file.extend(np.array([sample_file_path.stem] * n_events, dtype=object))

            for category, attributes in features.items():
                if category == 'extra':
                    original_extra = np.column_stack([data[ext] for ext in attributes])
                    Extra.append(original_extra)
                    continue

                for particle in attributes.get('particles', []):
                    # Prepare Kinematics
                    pt, eta, phi, mass = data[particle][:, 0], np.zeros_like(data[particle][:, 0]), None, None

                    if category == 'met':
                        phi, mass = data[particle][:, 1], np.zeros_like(pt)
                    elif category == "nu":
                        eta, phi, mass = data[particle][:, 1], data[particle][:, 2], np.zeros_like(pt)
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

    X = np.concatenate([np.vstack(parts)[:, None, :] for parts in X.values()], axis=1)
    nu = np.concatenate([np.vstack(parts) for parts in nu.values()], axis=1)
    y = np.vstack(y)
    Extra = np.vstack(Extra)

    # calculating MET-jet related variables for conditioning
    # Extract jets from X
    # Assuming first 2 particles = tau_vis → jets start from index 2
    jet_start_index = len(features['tau_vis']['particles'])
    jets_X = X[:, jet_start_index:, :4]  # shape: (n_events, n_jets, 4)
    y = build_condition_vector(jets_X, y)

    # convert pt and energy to log(x + 1)
    X[:, :, 0] = np.log1p(X[:, :, 0])  # pt
    X[:, :, 3] = np.log1p(X[:, :, 3])  # energy
    nu[:, 0] = np.log1p(nu[:, 0])  # nu1 pt
    nu[:, 3] = np.log1p(nu[:, 3])  # nu2 pt

    if for_training:
        # Indices to compute mean and std
        selected_indices = [0, 3]
        particle_mean = np.zeros(X.shape[2])
        particle_std = np.ones(X.shape[2])
        nu_mean = np.zeros(nu.shape[1])
        nu_std = np.ones(nu.shape[1])
        for idx in selected_indices:
            particle_mean[idx] = np.mean(X[:, :, idx], axis=(0, 1), where=X[:, :, idx] != 0)
            particle_std[idx] = np.std(X[:, :, idx], axis=(0, 1), where=X[:, :, idx] != 0)
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

        indices = np.random.permutation(X.shape[0])  # Shuffle indices
        X, nu, y, Extra = (arr[indices] for arr in (X, nu, y, Extra))  # Apply shuffle

        split_idx = int(len(X) * train_ratio)  # Compute split index
        train_data = (X[:split_idx], nu[:split_idx], y[:split_idx], Extra[:split_idx])
        test_data = (X[split_idx:], nu[split_idx:], y[split_idx:], Extra[split_idx:])

        # Save train & test data
        save_hdf5(train_file, ["X", "nu", "Condition", "Extra"], train_data, mode="train")
        save_hdf5(test_file, ["X", "nu", "Condition", "Extra"], test_data, mode="test")
    else:
        norm_dict = {}
        save_hdf5(train_file, ["X", "nu", "Condition", "Extra", "RawFile"], (X, nu, y, Extra, raw_file), mode="evaluation")

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
