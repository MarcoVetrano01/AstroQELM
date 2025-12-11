from .utils import MatrixLike
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.linear_model import RidgeCV, Ridge
from tqdm import tqdm

def map_angle(data: MatrixLike):

    """Maps data in [0, 1] to angles in [0, pi].
     Args:
         data (MatrixLike): Input data to be mapped.

     Returns:
         MatrixLike: Mapped data.
     """

    data_mapped = data * 2 * np.pi

    return data_mapped


def AngleEncoding(nq: int, data: MatrixLike) -> QuantumCircuit:

    """Encodes data into a quantum circuit using RX gates. The number of encoded features must be n_features <= nq.
     Args:
         nq (int): Number of qubits.
         data (MatrixLike): Input data to be encoded.
         encoding_gate (str): Type of encoding gate to be used. Default is "RX".
     Returns:
         QuantumCircuit: Quantum circuit with encoded data.
     """
    
    if not isinstance(data, MatrixLike):
        raise TypeError("Data must be of type MatrixLike.")
    
    enc_dim = data.shape[0]
    
    quantum_circuits = QuantumCircuit(nq)

    for i in range(enc_dim):
        quantum_circuits.rx(data[i], i)


    return quantum_circuits

def ReservoirLayer(qc: QuantumCircuit, par: np.ndarray,depth: int = 1) -> QuantumCircuit:

    """Adds a reservoir layer to the quantum circuit using random CZ gates.
     Args:
         qc (QuantumCircuit): Input quantum circuit.
         depth (int): Depth of the reservoir layer.
     Returns:
         QuantumCircuit: Quantum circuit with reservoir layer.
     """

    nq = qc.num_qubits
    
    for d in range(depth):

        for i in range(nq):
            qc.ry(par[d, i], i)

        for i in range(nq - 1):
            qc.cx(i, i + 1)
        
        for i in range(nq):
            qc.ry(par[d, i + nq], i)

        qc.barrier()

    return Statevector(qc).probabilities()

def FiniteStatistics(probabilities: MatrixLike, shots: int) -> np.ndarray:

    """Inserts statistical shot noise in the results.
     Args:
         probabilities (np.ndarray): Input probabilities.
         shots (int): Number of shots (samples) to simulate.
     Returns:
         np.ndarray: Finite statistics.
     """

    stats = np.random.multinomial(shots, probabilities) / shots

    return stats

def Reservoir(nq: int, data: MatrixLike, par: np.ndarray, depth: int = 1, shots: int = 1024, disable_progress_bar: bool = True) -> np.ndarray:

    """Reservoir Wrapper.
     Args:
         nq (int): Number of qubits.
         data (MatrixLike): Input data to be encoded.
         depth (int): Depth of the reservoir layer.
         shots (int): Number of shots (samples) to simulate.
     Returns:
         np.ndarray: QELM output statistics.
     """

    dim = data.shape[0]

    # Create quantum circuit
    result_si = np.zeros((dim, 2 ** nq))        

    for i in tqdm(range(dim), disable = disable_progress_bar):
        qc = AngleEncoding(nq, data[i])

        # Add reservoir layer
        result_si[i] = ReservoirLayer(qc, par, depth)

    if shots > 1:
        result_sf = np.zeros((dim, 2 ** nq))
        # Compute finite statistics
        for i in range(dim):
            result_sf[i] = FiniteStatistics(result_si[i], shots)
        result_sf = result_sf.T
    else:
        result_sf = np.zeros((2**nq, dim))

    result_si = result_si.T

    return result_si, result_sf

def training(x_train: np.ndarray, y_train: MatrixLike, regularize: bool = True) -> Ridge | RidgeCV:

    """Training Wrapper.
     Args:
         x_train (np.ndarray): Training data.
         y_train (MatrixLike): Training targets.
         regularize (bool): Whether to use regularization. Default is True.
     Returns:
         Ridge | RidgeCV: Trained Ridge regression model.
     """

    if regularize:
        model = RidgeCV(alphas=np.logspace(-7, 2, 100))
    else:
        model = Ridge(alpha=0.0, solver = 'svd')

    model.fit(x_train, y_train)

    return model

def FactorizedQELM(data: MatrixLike, targets: MatrixLike, nq: int , global_properties: np.ndarray | None = None, patchwise_properties: np.ndarray | None = None, depth: int = 1, shots: int = 1024, train_size: float = 0.8, regularize: bool = True, disable_progress_bar: bool = False) -> Ridge | RidgeCV:

    """Factorized QELM Wrapper.
     Args:
         data (MatrixLike): Input data to be encoded of shape (patches, n_samples, n_features).
         targets (MatrixLike): Target values of shape (n_samples, n_outputs).
         nq (int): Number of qubits.
         means (np.ndarray | None): Mean values for data normalization. 
         depth (int): Depth of the reservoir layer.
         shots (int): Number of shots (samples) to simulate.
         train_size (float): Proportion of data to be used for training. Default is 0.8.
         regularize (bool): Whether to use regularization. Default is True.
     Returns:
         Ridge | RidgeCV: Trained Ridge regression model.
     """

    data = np.array(data)
    data_mapped = map_angle(data)

    patches = data.shape[0]
    dim = data.shape[1]
    enc_dim = data.shape[2]
    par = np.random.uniform(0, np.pi, size=(patches, depth,  2 * nq))
    if enc_dim > nq:
        raise ValueError("Number of encoded features must be less than or equal to number of qubits.")
    if shots < 1:
        raise ValueError("Number of shots must be at least 1.")
    if not (0 < train_size < 1):
        raise ValueError("Train size must be between 0 and 1.")
    
    split_idx = int(dim * train_size)
    y_train = targets[:split_idx]
    y_test = targets[split_idx:]

    results_si = np.zeros((patches, 2 ** nq, dim))
    results_sf = np.zeros((patches, 2 ** nq, dim))
    for i in tqdm(range(patches), desc="Processing patches", disable = disable_progress_bar):
        x_patch = data_mapped[i]
        results_si[i], results_sf[i] = Reservoir(nq, x_patch, par[i], depth, shots)
    results_sf = np.concatenate(results_sf, axis=0).T
    results_si = np.concatenate(results_si, axis=0).T
    
    if global_properties is not None:
        print("Processing global properties...")
        par_mean = np.random.uniform(0, np.pi, size=(depth,  2 * nq))
        global_properties_mapped = map_angle(global_properties)
        result_global_si, result_global_sf = Reservoir(nq, global_properties_mapped, par_mean, depth, shots, False)
        results_si = np.concatenate((results_si, result_global_si.T), axis=1)
        results_sf = np.concatenate((results_sf, result_global_sf.T), axis=1)
    
    if patchwise_properties is not None:
        par_patch = np.random.uniform(0, np.pi, size=(patches, depth,  2 * nq))
        patchwise_properties_mapped = map_angle(patchwise_properties)
        results_local_si = np.zeros((patches, 2 ** nq, dim))
        results_local_sf = np.zeros((patches, 2 ** nq, dim))
        for i in tqdm(range(patches), desc="Processing local properties", disable = disable_progress_bar):
            x_patch_prop = patchwise_properties_mapped[i]
            results_local_si[i], results_local_sf[i] = Reservoir(nq, x_patch_prop, par_patch[i], depth, shots)
        results_local_sf = np.concatenate(results_local_sf, axis=0).T
        results_local_si = np.concatenate(results_local_si, axis=0).T
        results_si =  np.concatenate((results_si, results_local_si), axis=1)
        results_sf =  np.concatenate((results_sf, results_local_sf), axis=1)

    x_train = results_si[:split_idx]
    x_test = results_si[split_idx:]
    W_si = training(x_train, y_train, regularize)
    y_pred_si = W_si.predict(x_test)

    if shots > 1:
        x_train_sf = results_sf[:split_idx]
        x_test_sf = results_sf[split_idx:]
        W_sf = training(x_train_sf, y_train, regularize)
        y_pred_sf = W_sf.predict(x_test_sf)
    else:
        y_pred_sf = None
    
    return y_pred_si, y_pred_sf