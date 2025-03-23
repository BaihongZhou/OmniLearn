import sys
import os
from pathlib import Path
import numpy as np
import pickle
from tqdm import tqdm
import vector
from functools import partial

from stack import plot_hist


def get_neutrino_candidates(reco_nu, method='random'):
    if method == 'random':
        idx = np.random.randint(0, reco_nu.shape[1], size=reco_nu.shape[0])
        return np.array([reco_nu[i, idx[i], :] for i in range(reco_nu.shape[0])])
    elif method == 'average':
        return reco_nu.mean(axis=1)
    else:
        raise ValueError("Method must be 'random' or 'average'")


def pre_selection(data: dict[str, vector.MomentumNumpy4D | list], select: bool = True) -> dict:
    sel = (data['truth_TauTau'].E > 0)
    sel &= (data['truth_TauTau'].mass < 200) & (data['truth_TauTau'].mass > 0)
    sel &= (data['truth_nu1'].pt > 0)
    sel &= (data['truth_nu2'].pt > 0)

    for key, value in data.items():
        if isinstance(value, list):
            if select:
                data[key] = np.array(value)[sel]
            else:
                data[key] = np.array(value)
        else:
            if select:
                data[key] = data[key][sel]

    # print(f"Pre-selection: {np.sum(sel)} / {len(sel)} = {np.sum(sel) / len(sel):.2%}")

    return data


def read_data(raw_files: list[Path], ml_files: list[Path]):
    """Reads data from a list of pickle files and concat all (all is numpy array)."""

    if len(raw_files) != len(ml_files):
        raise ValueError("Raw and ML files must have the same length")

    data = {}

    for raw_file, ml_file in zip(raw_files, ml_files):
        raw_data = pickle.load(open(raw_file, "rb"))
        ml_data = np.load(ml_file)
        ml_data = {key: ml_data[key] for key in ml_data.files}

        raw_run_number = np.array(raw_data['runNumber'])
        raw_event_number = np.array(raw_data['eventNumber'])

        ml_event_number = np.array(ml_data['extra_1'])
        ml_run_number = np.array(ml_data['extra_2'])

        if not np.all((raw_event_number == ml_event_number) and (raw_run_number == ml_run_number)):
            raise ValueError("Event numbers and Run numbers do not match")

        for key in raw_data:
            if key not in data:
                data[key] = []
            data[key].append(raw_data[key])

        data['reco_nu1'] = ml_data['recon_nu1']
        data['reco_nu2'] = ml_data['recon_nu2']

    for key in data:
        data[key] = np.concatenate(data[key], axis=0)

    return data


def process_data(
        data: dict,
        get_neutrino: partial[get_neutrino_candidates] = None,
        delta_columns: list = None,
        key_columns: list = None
):
    for key in data:
        if key_columns is not None and key in key_columns:

            array_data = data[key]
            if 'reco_nu' in key:
                # data[key] = data[key.replace('reco_nu', 'truth_nu')]

                # IMPORTANT
                # pt and E is np.log1p(pt), need to revert
                array_data = get_neutrino(data[key])
                data[key] = vector.array({
                    'pt': np.expm1(array_data[:, 0]),
                    'eta': array_data[:, 1],
                    'phi': array_data[:, 2],
                    'mass': np.zeros(array_data.shape[0]),
                }).to_ptphietamass()
            else:
                data[key] = vector.array({
                    'pt': array_data[:, 0],
                    'eta': array_data[:, 1],
                    'phi': array_data[:, 2],
                    'mass': array_data[:, 3],
                }).to_ptphietamass()

    # if sum_columns is not None:
    #     for key in sum_columns:
    #         for col in sum_columns[key]:
    #             if key not in data:
    #                 data[key] = data[col]
    #             else:
    #                 data[key] += data[col]

    data['reco_Tau1'] = data['Tau1'] + data['reco_nu1']
    data['reco_Tau2'] = data['Tau2'] + data['reco_nu2']
    data['reco_TauTau'] = data['reco_Tau1'] + data['reco_Tau2']
    data['reco_HH'] = data['reco_TauTau'] + data['Jet_b1'] + data['Jet_b2']
    data['truth_HH'] = data['truth_TauTau'] + data['truth_bb']

    data = pre_selection(data)

    if delta_columns is not None:
        for key in delta_columns:
            data[f'delta_{key}'] = (data['reco_' + key] - data['truth_' + key])

    return data


def read_variable(files: dict, var: str, weight: str = 'weight'):
    """Reads a variable from a list of pickle files and concat all (all is numpy array)."""

    data = {}

    for sample in files.keys():
        data[sample] = {
            'var': files[sample]['data'][var],
            'weight': files[sample]['data'][weight]
        }

        for key in files[sample].keys():
            if key in ['data']:
                continue
            data[sample][key] = files[sample][key]

    return data


if __name__ == '__main__':
    base_dir = Path('/global/cfs/cdirs/m2616/avencast/bbtautau/tautau_reconstruction/out_20250219_eval')
    out_dir = Path('/global/cfs/cdirs/m2616/avencast/bbtautau/tautau_reconstruction/out_20250219_plots')
    out_dir.mkdir(exist_ok=True)

    files = {
        'hhttbbSM': {
            'raw': ['hhttbbSM_eval.pkl'],
            'ml': ['hhttbbSM_recon.pkl'],
            'signal': True,
            'color': '#cc7c71',
        },
        'ytautau': {
            'raw': ['ytautau_eval.pkl'],
            'ml': ['ytautau_recon.pkl'],
            'signal': False,
            'color': '#7ab656',
        },
        'Ztt': {
            'raw': ['Ztt_eval.pkl'],
            'ml': ['Ztt_recon.pkl'],
            'signal': False,
            'color': '#925eb0',
        },
        'ttbar_dilep': {
            'raw': ['ttbar_dilep_eval.pkl'],
            'ml': ['ttbar_dilep_recon.pkl'],
            'signal': False,
            'color': '#7399f4',
        },
        'VBFhhttbbSM': {
            'raw': ['VBFhhttbbSM_recon.pkl'],
            'ml': ['VBFhhttbbSM_eval.pkl'],
            'signal': False,
            'color': '#a5aeb7',
        },
    }

    key_columns = [
        'truth_TauTau',
        'truth_nu1', 'truth_nu2',
        'Tau1', 'Tau2',
        'reco_nu1', 'reco_nu2',
        'Jet_b1', 'Jet_b2',
        'truth_bb',
    ]

    delta_columns = [
        'TauTau',
        'HH',
        'nu1', 'nu2',
    ]

    delta_columns_title = [
        r'$M_{\tau\tau}$',
        r'$M_{HH}$',
        r'$M_{\nu_1}$',
        r'$M_{\nu_2}$',
    ]

    for samples in files.keys():
        files[samples]['raw'] = [base_dir / f for f in files[samples]['raw']]
        files[samples]['ml'] = [base_dir / f for f in files[samples]['ml']]
        files[samples]['data'] = read_data(files[samples]['raw'], files[samples]['ml'])

    get_neutrino = partial(get_neutrino_candidates, method='random')
    for samples in files.keys():
        files[samples]['data'] = process_data(
            files[samples]['data'],
            get_neutrino=get_neutrino,
            key_columns=key_columns,
            delta_columns=delta_columns,
        )

    for var, var_title in tqdm(zip(delta_columns, delta_columns_title), desc='Variables'):
        nominal = read_variable(files, f'reco_{var}', 'weight_mc')
        truth = read_variable(files, f'truth_{var}', 'weight_mc')
        delta_nominal = read_variable(files, f'delta_{var}', 'weight_mc')

        for kin, kin_title in zip(['pt', 'eta', 'phi', 'mass'], [r'p^T', r'\eta', r'\phi', r'M']):
            x_title = var_title.replace('M', kin_title)

            plot_hist(
                data=nominal,
                column_to_plot=kin,
                weight_col='weight',
                bins=100,
                x_range=None,
                plot_sig_percentile=(0.5, 99.5),
                fig_size=(10, 8),
                x_title=x_title,
                save_path=out_dir / f'{var}_{kin}.png'
            )

            plot_hist(
                data=truth,
                column_to_plot=kin,
                weight_col='weight',
                bins=100,
                x_range=None,
                plot_sig_percentile=(0.5, 99.5),
                fig_size=(10, 8),
                x_title=x_title,
                save_path=out_dir / f'{var}_{kin}_truth.png'
            )

            plot_hist(
                data=delta_nominal,
                column_to_plot=kin,
                delta_data=truth,
                weight_col='weight',
                bins=100,
                x_range=(-1.0, 1),
                plot_sig_percentile=None,
                fig_size=(10, 8),
                x_title=f'$\\Delta$ {x_title}',
                save_path=out_dir / f'delta_{var}_{kin}.png'
            )
