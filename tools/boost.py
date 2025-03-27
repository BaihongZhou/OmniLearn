import pickle
from pathlib import Path
from tqdm import tqdm

import numpy as np
import os
import vector


def is_valid_event(input_dict: dict, boost_col: list, boost_max_eta: float = 5.0, boost_min_pt: float = 0.0):
    """
    Generate a mask for events where all tau child particles have |eta| < 2.5 and |pT| > 10 GeV (if pT != 0).
    """
    valid_mask = np.ones(len(input_dict['truth_TauTau']), dtype=bool)
    for key in boost_col:
        # Ensure |eta| < 2.5
        valid_mask &= (np.abs(input_dict[key][:, 1]) < boost_max_eta)
        # Ensure 1.37 < |eta| < 1.52
        # valid_mask &= ((np.abs(input_dict[key][:, 1]) < 1.37) | (np.abs(input_dict[key][:, 1]) > 1.52))

        # Apply pT conditions
        pt_values = input_dict[key][:, 0]
        valid_mask &= ((pt_values == 0) | (pt_values > boost_min_pt))

    return valid_mask


def pre_selection(data: dict[str, vector.MomentumNumpy4D | list], select: bool = True, with_bb: bool = False) -> dict:
    sel = (data['truth_TauTau'].E > 0)
    sel &= (data['truth_TauTau'].mass < 260) & (data['truth_TauTau'].mass > 60)
    # sel &= (data['truth_TauTau'].pt > 0) & (data['truth_TauTau'].eta > -10) & (data['truth_TauTau'].phi > -50)
    sel &= (data['truth_nu1'].E > 0)
    sel &= (data['truth_nu2'].E > 0)

    if with_bb:
        sel &= (data['truth_bb'].E > 0)

        data['truth_HH'] = data['truth_TauTau'] + data['truth_bb']

    for key, value in data.items():
        if isinstance(value, list):
            if select:
                data[key] = np.array(value)[sel]
            else:
                data[key] = np.array(value)
        else:
            if key in ['Tau1', 'Tau2', 'Jet_b1', 'Jet_b2', 'Jet_0', 'Jet_1', 'Jet_2']:
                valid_mask = value.E >= 0
                data[key] = vector.arr(
                    dict(
                        pt=np.where(valid_mask, value.pt, 0),
                        eta=np.where(valid_mask, value.eta, 0),
                        phi=np.where(valid_mask, value.phi, 0),
                        mass=np.where(valid_mask, value.mass, 0),
                    )
                )

            if select:
                data[key] = data[key][sel]

    # print(f"Pre-selection: {np.sum(sel)} / {len(sel)} = {np.sum(sel) / len(sel):.2%}")

    return data


def boost(
        data: dict[str, vector.MomentumNumpy4D | np.ndarray], seed: int = 42, boost: bool = True, with_bb: bool = False
) -> dict:
    np.random.seed(seed)

    skip_columns = [
        'weight_mc', 'eventNumber', 'runNumber', 'n_jets',
        'Tau1_nProng', 'Tau1_decayMode', 'Tau1_charge',
        'Tau2_nProng', 'Tau2_decayMode', 'Tau2_charge',
        # 'truth_TauTau',
    ]
    # drop_columns = ['eventNumber', 'runNumber', 'sum_metTautau', 'mmc'] if boost else []
    drop_columns = []
    mother_column = 'truth_TauTau' if not with_bb else 'truth_HH'
    boost_columns = ['Tau1', 'Tau2', mother_column] if boost else []
    neutrino_columns = ['truth_nu1', 'truth_nu2'] if boost else []
    met_columns = ['met']

    total_events = len(data[skip_columns[0]])
    remaining_mask = np.ones(total_events, dtype=bool)  # Start with all events to process

    # Initialize the final boosted save dictionary
    final_boosted_save_dict = {key: [] for key in data.keys()}

    max_attempts = 10  # Maximum attempts to resample failed events

    while np.any(remaining_mask):
        # Step 1: Apply the boost to the remaining particles
        boosted_save_dict = {}
        for key in skip_columns:
            if key not in drop_columns:
                boosted_save_dict[key] = data[key][remaining_mask]

        vis_momentum_change = vector.array({
            'px': np.zeros(np.sum(remaining_mask)),
            'py': np.zeros(np.sum(remaining_mask)),
            'pz': np.zeros(np.sum(remaining_mask)),
            'mass': np.zeros(np.sum(remaining_mask)),
        })

        boosted_par_dict = {}
        for key, val in data.items():
            if key in [*skip_columns, *met_columns]:
                continue

            boosted_par_dict[key] = val[remaining_mask]

        # Calculate tau+tau 4-momentum
        mother = boosted_par_dict[mother_column]
        boost_to_rest = -mother.to_beta3()
        random_angle = np.random.uniform(0, 2 * np.pi, np.sum(remaining_mask))

        for key, val in boosted_par_dict.items():
            if key in [*skip_columns, *met_columns]:
                continue

            particle_original = val
            if key in [*boost_columns, *neutrino_columns]:
                particle_rest = particle_original.boost(boost_to_rest)
                particle_boosted = particle_rest.rotateZ(random_angle)
                particle_lab = particle_boosted.boost(-boost_to_rest)
            else:
                particle_lab = particle_original

            boosted_save_dict[key] = np.array(
                [particle_lab.pt, particle_lab.eta, particle_lab.phi, particle_lab.mass]
            ).T

            vis_momentum_change -= (particle_lab - particle_original).to_pxpyetamass()

        # Create a vector for the original MET
        for met in met_columns:
            met_vector = data[met][remaining_mask]

            # Adjust MET by subtracting the visible momentum change
            adjusted_met = met_vector + vis_momentum_change

            # Update MET in boosted_dict
            boosted_save_dict[met] = np.array([
                adjusted_met.pt,  # Updated MET_ET
                adjusted_met.phi,  # Updated MET_phi
                # adjusted_met.eta,  # Updated MET_eta
                # adjusted_met.mass  # Updated MET_mass
            ]).T

        # Step 3: Check validity of the events
        valid_mask = is_valid_event(boosted_save_dict, boost_columns, boost_max_eta=5.0, boost_min_pt=0.0)

        # Collect valid events into the final boosted save dictionary
        for key in boosted_save_dict.keys():
            if key in final_boosted_save_dict:
                final_boosted_save_dict[key].append(boosted_save_dict[key][valid_mask])
            else:
                final_boosted_save_dict[key] = [boosted_save_dict[key][valid_mask]]

        # Update the remaining mask for invalid events
        remaining_indices = np.where(remaining_mask)[0]
        remaining_mask[remaining_indices[valid_mask]] = False

        max_attempts -= 1
        if max_attempts <= 0:
            print("\nFailed to generate valid events after multiple attempts.")
            print(
                f"Number of remaining events: {np.sum(remaining_mask)} "
                f"[ / {total_events} = {np.sum(remaining_mask) / total_events: .2%}]\n"
            )
            break

    # drop columns
    final_boosted_save_dict = {
        key: np.concatenate(final_boosted_save_dict[key], axis=0)
        for key in final_boosted_save_dict.keys()
        if key not in drop_columns
    }

    return final_boosted_save_dict


def reweight_flat_distribution(mass: np.ndarray, bins: int = 100, range_min: float = 60.0, range_max: float = 200.0):
    hist, bin_edges = np.histogram(mass, bins=bins, range=(range_min, range_max))

    # Avoid division by zero
    hist = np.maximum(hist, 1e-8)
    target = np.ones_like(hist)

    weights_per_bin = target / hist
    bin_indices = np.digitize(mass, bin_edges) - 1
    weights = np.zeros_like(mass)

    in_range_mask = (mass >= range_min) & (mass <= range_max)
    weights[in_range_mask] = weights_per_bin[bin_indices[in_range_mask]]

    # Normalize weights so the total weight in range matches the number of events in range
    total_entries_in_range = np.sum(in_range_mask)
    total_weight_in_range = np.sum(weights[in_range_mask])
    if total_weight_in_range > 0:
        weights[in_range_mask] *= total_entries_in_range / total_weight_in_range

    return weights


def main(args):
    pass


def build_extra_variables(data: dict):
    # data['sum_metTauTau'] = data['met'] + data['Tau1'] + data['Tau2']
    # data['sum_metTauTau_diff'] = data['truth_TauTau'].to_pxpypzenergy() - data['sum_metTauTau'].to_pxpypzenergy()

    truth_tautau = vector.arr(dict(
        pt=data['truth_TauTau'][:, 0],
        eta=data['truth_TauTau'][:, 1],
        phi=data['truth_TauTau'][:, 2],
        mass=data['truth_TauTau'][:, 3],
    ))

    tau1 = vector.arr(dict(
        pt=data['Tau1'][:, 0],
        eta=data['Tau1'][:, 1],
        phi=data['Tau1'][:, 2],
        mass=data['Tau1'][:, 3],
    ))
    tau2 = vector.arr(dict(
        pt=data['Tau2'][:, 0],
        eta=data['Tau2'][:, 1],
        phi=data['Tau2'][:, 2],
        mass=data['Tau2'][:, 3],
    ))
    met = vector.arr(dict(
        pt=data['met'][:, 0],
        phi=data['met'][:, 1],
        eta=np.zeros(len(data['met'])),
        mass=np.zeros(len(data['met'])),
    ))

    sum_metTauTau = met + tau1 + tau2
    sum_metTauTau_diff = truth_tautau.to_pxpypzenergy() - sum_metTauTau.to_pxpypzenergy()
    data['sum_metTauTau_diff'] = np.array(
        [sum_metTauTau_diff.pt, sum_metTauTau_diff.eta, sum_metTauTau_diff.phi, sum_metTauTau_diff.energy]
    ).T

    return data


if __name__ == '__main__':
    base_dir = Path('./raw')
    out_dir = Path('./eval')
    out_dir.mkdir(exist_ok=True)

    files = {
        # 'hhttbbSM': {
        #     'f': [f'hhttbbSM_{i}.pkl' for i in range(3)],
        #     'boost': True,
        #     'boost_round': 3,
        #     'with_bb': True,
        # },
        'ytautau': {
            'f': [f'ytautau_{i}.pkl' for i in range(3)],
            'boost': True,
            'boost_round': 20,
            'with_bb': False,
            'reweight': True,
        },
        # 'Ztt': {
        #     'f': [f'Ztt_jets_{i}.pkl' for i in range(10)],
        #     'boost': True,
        #     'boost_round': 3,
        #     'with_bb': False,
        # },
        # 'ttbar_dilep': {
        #     'f': [f'ttbar_dilep_{i}.pkl' for i in range(10)],
        #     'boost': False,
        #     'boost_round': 1,
        #     'with_bb': False,
        # },
        # 'VBFhhttbbSM': {
        #     'f': [f'VBFhhttbbSM_{i}.pkl' for i in range(3)],
        #     'boost': False,
        #     'boost_round': 1,
        #     'with_bb': True,
        # },
    }

    # !!! for eval !!!
    files = {f'{sample}_eval': value for sample, value in files.items()}
    for samples in files.keys():
        files[samples]['boost'] = False
        files[samples]['boost_round'] = 1

    for samples in files.keys():
        files[samples]['f'] = [base_dir / f for f in files[samples]['f']]

    for sample in tqdm(files, desc='Loading files'):
        if not all(f.exists() for f in files[sample]['f']):
            raise FileNotFoundError(
                f"Missing files for {sample}: {[str(f) for f in files[sample]['f'] if not f.exists()]}")

        boost_round = files[sample]['boost_round']
        if_boost = files[sample]['boost']

        save_data = None
        for f in files[sample]['f']:
            print(f"Loading {f}")
            in_data = pickle.load(open(f, 'rb'))
            in_data = pre_selection(in_data, select=if_boost)

            for i in tqdm(range(boost_round), desc=f'Boosting {sample}', leave=False):
                out_data = boost(in_data, i, boost=if_boost)

                if save_data is None:
                    save_data = {key: [] for key in out_data.keys()}

                for key in save_data:
                    save_data[key].append(out_data[key])  # Append instead of concatenating

        # Convert lists to numpy arrays **only once**
        for key in save_data:
            save_data[key] = np.concatenate(save_data[key], axis=0)

        # save_data = build_extra_variables(save_data)

        print(f"\n {sample}: Final data length: {len(save_data[list(save_data)[0]])}")

        if files[sample].get('reweight', False): # and files[sample]['boost']:
            save_data['weight_mc'] = reweight_flat_distribution(
                mass=save_data['truth_TauTau'][:, 3], bins=200, range_min=60, range_max=260
            )

        # Save the final merged data
        out_file = out_dir / f"{sample}_boosted.pkl" if if_boost else out_dir / f"{sample}.pkl"
        print(f"\nSaving {out_file}")
        pickle.dump(save_data, open(out_file, 'wb'))
