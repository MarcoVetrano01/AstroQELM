# This module contains all the function for the preprocessing of Astrophysical datas with techniques inspired to those of 
# Zingales, T. et Al. (2018). Exogan: Retrieving exoplanetary atmospheres using deep convolutional generative adversarial networks. 
# The Astronomical Journal, 156(6), 268.

# It also contains some functions to add artificial shot noise to the datasets as done in Cowan et al 2015 https://iopscience.iop.org/article/10.1086/680855

import numpy as np
from scipy.integrate import quad
from .utils import MatrixLike
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d

#Constants

h = 6.62607004e-34
c = 299792458.0
Rsun = 695508e3
RJUP = 6.9911e7
MJUP = 1.898e27
au = 149597870700.0
kB = 1.38064852e-23
pc = 3.086e16
errfloor = 30e-6

#Stellar Temperature and Radius

Ts = 6460.0
Rs = 1.458 * Rsun

#Planetary Radius, Orbital Period, Distance, Semi-Major Axis (WASP-121 b)

Rp = 1.807 * RJUP
P = 1.2749255 * 86400.0
d = 270 * pc
ax = 0.02544 * au


# Telescope parameters (diameter and efficiency)
D = 16
tau = 0.4


norm_factors = np.array([8,8,8,8,2*MJUP,1.5*RJUP, 2e3])

# Functions for Exoplanetary Datasets

# ======================================================
#                       SHOT NOISE 
# ======================================================

def Bl(wav: float, T: float = Ts) -> float:

    """Black Body Radiation Function for a given wavelength and temperature.
    Args:
        wav (float): Wavelength in meters.
        T (float): Temperature in Kelvin. Default is Ts = 6460.0 K.
        Returns:
        float: Black Body Radiation in W/sr/m^3.
    """

    non_negativity_checks = np.any([T < 0, wav < 0])
    if non_negativity_checks:
        raise ValueError("Wavelength and Temperature must be greater than zero.")
    
    scalar_checks = np.any([len(np.shape(T)), len(np.shape(wav))])
    if scalar_checks:
        raise ValueError("Wavelength and Temperature must be scalars.")

    fac = 2.0 * h * c**2 / (wav) ** 5.0
    bose = 1.0 / (np.exp(h * c / (wav * kB * T)) - 1.0)
    return fac * bose * wav

def tran_dur(P: float, ax: float, Rs: float, Rp: float) -> float:

    """Calculate the transit duration for a given set of parameters.
    Args:
        P (float): Orbital period in seconds.
        ax (float): Semi-major axis in meters.
        Rs (float): Stellar radius in meters.
        Rp (float): Planetary radius in meters.
    Returns:
        float: Transit duration in seconds.
    """
    controls = np.any([len(np.shape(P)), len(np.shape(ax)), len(np.shape(Rs)), len(np.shape(Rp))])
    if controls:
        raise ValueError("All inputs must be scalars.")

    b = ax / Rs
    return P / np.pi * np.arcsin(np.sqrt((Rs + Rp) ** 2 - b**2) / ax)

Dt = 2 * tran_dur(P, ax, Rs, Rp)

def Nphot(l1: float, l2: float, τ: float = tau, dt: float = Dt, R: float = Rs, D_tel: float = D, distance: float = d) -> float:

    """Calculate the number of photons received during a transit in a given wavelength range. Considering a Black Body Radiation for the star.
    Args:
        l1 (float): Lower wavelength limit in meters.
        l2 (float): Upper wavelength limit in meters.
        dt (float): Transit duration in seconds. Default is Dt calculated for WASP-121 b.
        R (float): Stellar radius in meters. Default is Rs = 1.458 * Rsun.
        D_tel (float): Telescope diameter in meters. Default is D = 16 m.
        distance (float): Distance to the star in meters. Default is d = 270 pc.
        τ (float): Telescope efficiency. Default is tau = 0.4.
    Returns:
        float: Number of photons received during the transit.
    """
    non_negativity_checks = np.any([l1 < 0, l2 < 0, dt < 0, R < 0, D_tel < 0, distance < 0, τ < 0])
    if non_negativity_checks:
        raise ValueError("All inputs must be greater than zero.")
    
    scalar_checks = np.any([len(np.shape(l1)), len(np.shape(l2)), len(np.shape(dt)), len(np.shape(R)), len(np.shape(D_tel)), len(np.shape(distance)), len(np.shape(τ))])
    if scalar_checks:
        raise ValueError("All inputs must be scalars.")
    
    fac = np.pi * τ * dt / (h * c) * (R * D_tel / (2.0 * distance)) ** 2
    integ = quad(Bl, l1, l2)
    # print(integ)
    return fac * integ[0]

def add_shot_noise(spectra: MatrixLike, wavs: MatrixLike, τ: float = tau, dt: float = Dt, R: float = Rs, D_tel: float = D, distance: float = d) -> np.ndarray:

    """Add shot noise to a set of spectra based on the number of photons received in each wavelength bin.
    Args:
        spectra (np.ndarray): Array of spectra to which shot noise will be added.
        wavs (np.ndarray): Array of wavelength bins corresponding to the spectra.
        dt (float): Transit duration in seconds. Default is Dt calculated for WASP-121 b.
        R (float): Stellar radius in meters. Default is Rs = 1.458 * Rsun.
        D_tel (float): Telescope diameter in meters. Default is D = 16 m.
        distance (float): Distance to the star in meters. Default is d = 270 pc.
        τ (float): Telescope efficiency. Default is tau = 0.4.
    Returns:
        np.ndarray: Spectra with added shot noise.
    """
    
    if not isinstance(spectra, MatrixLike):
        raise ValueError("Spectra must be a list or numpy array.")
    if not isinstance(wavs, MatrixLike):
        raise ValueError("Wavelengths must be a list or numpy array.")
    
    spectra = np.array(spectra)
    wavs = np.array(wavs)

    bins = np.append(wavs, 2 * wavs[-1] - wavs[-2])
    # Assumes just one target
    err = []
    tol = 1
    for i in range(len(bins) - 1):
        N = Nphot(bins[i], bins[i + 1], τ, dt, R, D_tel, distance)
        # print(N)
        binerr = np.sqrt(2.0) / np.sqrt(N)
        # print(binerr)
        if binerr > errfloor and binerr<tol:
            err.append(binerr)  # Use it for an accurate calculation of the SN errobars
        else:
            err.append(errfloor)  # constant error floor
    err = np.array(err)
    spectra_scattered = spectra + np.random.normal(0, err, size=spectra.shape)
    return spectra_scattered

# ======================================================
#                   PREPROCESSING 
# ======================================================

def JWST_interpolation(synth_data: MatrixLike, dense_wl: MatrixLike, synth_wl: MatrixLike):
    
    """Preprocess synthetic spectra by interpolating them onto a more dense spectral range contained in the synthetic one.
    Args:
        synth_data (MatrixLike): Synthetic spectra data.
        real_l (MatrixLike): Wavelength grid of real observations.
        synth_l (MatrixLike): Wavelength grid of synthetic spectra.
    Returns:
        tuple[np.ndarray, np.ndarray]: Preprocessed synthetic spectra and the new wavelength grid.
    """
    
    if not isinstance(synth_data, MatrixLike):
        raise ValueError("Synthetic data must be a list or numpy array.")
    if not isinstance(dense_wl, MatrixLike):
        raise ValueError("Dense wavelength grid must be a list or numpy array.")
    if not isinstance(synth_wl, MatrixLike):
        raise ValueError("Synthetic wavelength grid must be a list or numpy array.")

    synth_data = np.array(synth_data)
    dense_wl = np.array(dense_wl)
    synth_wl = np.array(synth_wl)

    spectral_range = synth_wl[(synth_wl >= dense_wl[0]) & (synth_wl <= dense_wl[-1])]
    start = list(synth_wl).index(spectral_range[0])
    end = list(synth_wl).index(spectral_range[-1])
    spectral_range = np.sort(np.concatenate((spectral_range, dense_wl)))
    synth_data = synth_data[:,start:end]
    
    spectra_int = []
    for i in range(np.shape(synth_data)[0]):
        interp_func = interp1d(synth_wl[start:end], synth_data[i], kind='linear', fill_value="extrapolate")
        spectra_int.append(interp_func(spectral_range))

    return np.array(spectra_int), spectral_range

def _spectra_normalization(spectra, idx):
    
    '''
    Function to divide the spectra into patches and normalize each in a range [0,1] as done in the work of ExoGAN (Zingales & Waldmann 2018)
    Args:
        spectra: MatrixLike
            Matrix containing the spectra to be normalized
        norm_idx: MatrixLike
            List containing the starting and ending indices of each patch to be normalized
            Example for ExoGAN:
            norm_idx_exogan = [0, 162, 195, 208, 241, 255, 318, 334, 371, 384, 394, 406, 420, 440, -1]
            Example for JWST wavelengths taken from HAT-P-18b 
            (FU, Guangwei et al. Water and an escaping helium tail detected in the hazy and methane-depleted atmosphere of HAT-P-18b from JWST NIRISS/SOSS.)
            norm_idx_JWST: [4,147,255,349,429,532,827,944,1167]
    Returns:
        mean_spectra: MatrixLike: average of the spectra before normalization
        norm_spectra: List[MatrixLike]: list containing the normalized patches of the spectra
    '''
    if not isinstance(idx[0], list):
        norm_idx = [[idx[i], idx[i + 1]] for i in range(len(idx) - 1)]
    else:
        norm_idx = idx

    patch = len(norm_idx)
    mean_spectra = np.mean(spectra, 1)
    norm_spectra = []

    for i in range(patch):
        frag = spectra[:,norm_idx[i][0]:norm_idx[i][1]]
        mm = MinMaxScaler()
        norm_spectra.append(mm.fit_transform(frag.T).T)
    return  mean_spectra, norm_spectra

def _feature_extraction(frag_spectra, comps):
    """
    Apply PCA to each fragment of the spectra and reduce its dimensionality. Principal components are normalized in [0,1]
    Args:
        frag_spectra (list): List of fragmented spectra
        comps (int): Number of principal components to keep for each fragment
    Returns:
        reduced_data (np.ndarray): Array of shape (Patches, samples, comps) containing the reduced spectra
    """
    patch = len(frag_spectra)
    p_comp = PCA(n_components=comps)
    reduced_data = []
    for i in range(patch):
        p_comp.fit(frag_spectra[i])
        reduced_data.append(p_comp.transform(frag_spectra[i]))
    reduced_data = np.array(reduced_data)
    reduced_data += np.abs(np.min(reduced_data))
    reduced_data = np.transpose(reduced_data, [1, 0, 2])
    reduced_data = np.transpose(reduced_data/(np.max(reduced_data,0)),[1,0,2])
    return reduced_data

def preprocessing_pipeline(spectra, norm_idx, comps):
    """
    Wrapper function to preprocess the spectra: normalization and feature extraction with PCA.
    Args:
        spectra (MatrixLike): Matrix containing the spectra to be preprocessed
        norm_idx (MatrixLike): List containing the starting and ending indices of each patch to be normalized
        comps (int): Number of principal components to keep for each fragment
    Returns:
        reduced_data (np.ndarray): Array of shape (Patches, samples, comps) containing the reduced spectra
        mean_spectra (MatrixLike): average of the spectra before normalization normalized in [0,1]
    """
    mean_spectra, frag_spectra = _spectra_normalization(spectra, norm_idx)
    reduced_data = _feature_extraction(frag_spectra, comps)
    mm = MinMaxScaler()
    mean_spectra = mm.fit_transform(mean_spectra.reshape(-1,1)).flatten()
    return reduced_data, mean_spectra