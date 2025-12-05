# This module contains utility functions and type definitions for the AstroQELM package. 
# This provides all the function to extract a couple of useful data structures.

import numpy as np
import pandas as pd
import h5py as h5
from typing import Union, tuple 
from pathlib import Path
import qiskit as qt
from itertools import product
from os import listdir

MatrixLike = Union[np.ndarray, list]

def add_zeros(result: qt.result.Result) -> dict:
    """Ensure all possible bitstrings are represented in the counts dictionary. This function is especially useful when dealing
        with experimental data obtained with real quantum hardware.
    Args:
        result (qt.result.Result): The result object obtained from executing a quantum circuit.
    Returns:
        dict: A dictionary containing counts for all possible bitstrings, with missing bitstrings assigned a count of zero.
    """
    if not isinstance(result, qt.result.Result):
        raise ValueError("Input must be a qiskit Result object.")
    try:
        counts = result.get_counts()
    except:
        counts = result
    n_qubits = len(list(dict(counts).keys())[0])
    all_bitstrings = ("".join(states) for states in product("01", repeat=n_qubits))
    counts = {key: counts.get(key, 0) for key in all_bitstrings}
    return counts

def data_from_h5(filename: h5.File) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extracts parameters and spectra from an HDF5 file produced with TauREx and returns them as pandas DataFrames.
    Args:
        filename (h5py.File): Path to the HDF5 file containing the data.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames - one for parameters and one for spectra.
    """

    f = h5.File(filename, 'r')

    values = list(f.keys())
    keys = list(f[values[0]].keys())
    headers = f[values[0]][keys[1]]
    key = list(f.keys())

    par = [[f[k][keys[1]][i][()] for i in f[values[0]][keys[1]].keys()] for k in key]
    df_parameters = pd.DataFrame(data = par, columns = headers)
    spectra = [f[k][keys[0]]['spectrum'][:] for k in key]
    df_spectra = pd.DataFrame(data = spectra, columns = [f'bin {i}' for i in range(len(spectra[0]))])

    return  df_parameters, df_spectra

def data_from_txt(path: str) -> tuple[np.ndarray, list]:
    """Extracts parameters and spectra from text files in a specified directory.
    Args:
        path (str): The path to the directory containing the text files.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames - one for parameters and one for spectra.
    """

    dirs = listdir(path)[3:]
    spectra = []
    parameters = []
    for file in dirs:
        p = Path(f"{path}/{file}")
    
        with p.open("r", errors="replace") as f:
            lines = [ln.rstrip("\n") for ln in f]
        df_lines = pd.DataFrame(lines, columns=["line"])
        spectrum = []
        with p.open("r", errors="replace") as f:
            for i, line in enumerate(lines):
                if i >= 69 and i <= 69+1404:
                    spectrum.append(line)
        name_par = []
        clean_spectrum = []
        for i, line in enumerate(spectrum):
            if i ==0:
                names = [line[3:6], line[10:16], line[23:28], line[36:39], line[47:52], line[59:64]]
                name_par.append(names)
            else:
                spectrum_value = [int(line[3:7]), float(line[9:19]), float(line[21:31]), float(line[33:43]), float(line[45:55]), float(line[57:67])]
                clean_spectrum.append(spectrum_value)
        
        with p.open("r", errors="replace") as f:
            par = []
            for i, line in enumerate(lines):
                if i >= 41 and i < 41+17:
                    par.append(line)
        
    
        names_par = []
        clean_par = []
        for line in par:
            name = line[24:32]
            name = name.replace(" ","")
            names_par.append(name)
            par_value = line[44:55]
            clean_par.append(float(par_value))
        parameters.append(clean_par)
        spectra.append(np.array(clean_spectrum)[:,5])


    df_spectra = pd.DataFrame(data = spectra, columns = [f'bin {i}' for i in range(len(spectra[0]))])   
    df_parameters = pd.DataFrame(data = parameters, columns = names_par)
    return df_parameters, df_spectra
