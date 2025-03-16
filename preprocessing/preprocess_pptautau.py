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


def process(
        data_path: Path, save_path: Path, save_tag: str,
        sample_lists: list[str], features: dict, train_ratio: float = 0.8,
        overwrite: bool = False,
):
    train_file = save_path / f"{save_tag}_train.hdf5"
    test_file = save_path / f"{save_tag}_test.hdf5"

    if train_file.exists() and test_file.exists() and not overwrite:
        print(f"Skipping processing data: {train_file} and {test_file} already exists")
        return

    X = {}
    nu = {}
    y = []
    Extra = []
    for file_path in sample_lists:
        with open(data_path / file_path, 'rb') as f:
            data = pickle.load(f)

            for category, attributes in features.items():
                if category == 'extra':
                    Extra.append(np.column_stack([data[ext] for ext in attributes]))

                    continue

                for particle in attributes.get('particles', []):
                    # Prepare Kinematics
                    pt, eta, phi, mass = data[particle][:, 0], np.zeros_like(data[particle][:, 0]), None, None

                    if particle == 'MET':
                        phi, mass = data[particle][:, 1], np.zeros_like(pt)
                    elif category == "nu":
                        eta, phi, mass = data[particle][:, 1], data[particle][:, 2], np.zeros_like(pt)
                    else:
                        eta, phi, mass = data[particle][:, 1], data[particle][:, 2], data[particle][:, 3]

                    kin_par = vector.arr(dict(pt=pt, eta=eta, phi=phi, mass=mass)).to_ptphietaenergy()

                    if particle == 'MET':
                        y.append(np.column_stack((pt, phi)))
                        continue

                    if category in ["nu"]:
                        kin_par = kin_par.to_pxpypzenergy()

                    # Collect kinematic features
                    particle_features = [kin_par[kin][:, None] for kin in attributes.get('kinematic', [])]

                    if category == "nu":
                        nu.setdefault(particle, []).append(np.column_stack(particle_features))
                        continue

                    # Collect extra features
                    for extra in attributes.get('extra', []):
                        extra_par = data[particle + '_' + extra] if category == 'tau_vis' else np.zeros_like(pt)
                        particle_features.append(extra_par[:, None])

                    # Assign labels
                    label = attributes.get('label', 1 if category == 'tau_vis' else 0)
                    particle_features.append(np.full_like(pt[:, None], label))

                    X.setdefault(particle, []).append(np.hstack(particle_features))

    X = np.concatenate([np.vstack(parts)[:, None, :] for parts in X.values()], axis=1)
    nu = np.concatenate([np.vstack(parts) for parts in nu.values()], axis=1)
    y = np.vstack(y)
    Extra = np.vstack(Extra)

    # convert pt and energy to log(x + 1)
    X[:, :, 0] = np.log(X[:, :, 0] + 1)
    X[:, :, 3] = np.log(X[:, :, 3] + 1)
    y[:, 0] = np.log(y[:, 0] + 1)

    # Indices to compute mean and std
    selected_indices = [0, 3]
    particle_mean = np.zeros(X.shape[2])
    particle_std = np.ones(X.shape[2])
    for idx in selected_indices:
        particle_mean[idx] = np.mean(X[:, :, idx], axis=(0, 1), where=X[:, :, idx] != 0)
        particle_std[idx] = np.std(X[:, :, idx], axis=(0, 1), where=X[:, :, idx] != 0)

    norm_dict = {
        "particle": {
            "mean": particle_mean.tolist(),
            "std": particle_std.tolist()
        },
        "nu": {
            "mean": np.mean(nu, axis=0, where=nu != 0).tolist(),
            "std": np.std(nu, axis=0, where=nu != 0).tolist(),
        }
    }

    indices = np.random.permutation(X.shape[0])  # Shuffle indices
    X, nu, y, Extra = (arr[indices] for arr in (X, nu, y, Extra))  # Apply shuffle

    split_idx = int(len(X) * train_ratio)  # Compute split index
    train_data = (X[:split_idx], nu[:split_idx], y[:split_idx], Extra[:split_idx])
    test_data = (X[split_idx:], nu[split_idx:], y[split_idx:], Extra[split_idx:])

    # Save function
    def save_hdf5(file_path, data, mode="train"):
        with h5.File(file_path, "w") as f:
            for name, arr in zip(["X", "nu", "MET", "Extra"], data):
                f.create_dataset(name, data=arr)
        print(f"{mode.capitalize()} data saved to {file_path}")

    # Save train & test data
    save_hdf5(train_file, train_data, mode="train")
    save_hdf5(test_file, test_data, mode="test")

    return norm_dict


def main():
    parser = argparse.ArgumentParser(description="Train the PET model on JetNet datasets.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")

    args = parser.parse_args()

    load_config(config_file=args.config)

    np.random.seed(42)

    cfg = config.cfg['preprocess']

    data_path = Path(cfg['raw_folder'])
    save_path = Path(cfg['processed_folder'])
    save_tag = config.cfg['sample']['tag']
    if not save_path.exists():
        save_path.mkdir(parents=True)

    print("Start processing the data")
    sample_lists = cfg['sample_list']
    features = config.cfg['features']

    norm_dict = process(
        data_path=data_path, save_path=save_path, save_tag=save_tag,
        sample_lists=sample_lists, features=features, train_ratio=0.8,
        overwrite=cfg['overwrite']
    )

    if norm_dict:
        config.cfg['sample']['normalization'] = norm_dict
        save_config(save_path=save_path, save_tag=f"{save_tag}_preprocess")


if __name__ == '__main__':
    main()
