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

def Reservoir(nq: int, data: MatrixLike, par: np.ndarray, depth: int = 1, shots: int | MatrixLike = 1024, disable_progress_bar: bool = True) -> np.ndarray:

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
    stat = len(shots)
    for i in tqdm(range(dim), disable = disable_progress_bar):
        qc = AngleEncoding(nq, data[i])

        # Add reservoir layer
        result_si[i] = ReservoirLayer(qc, par, depth)

    if np.any(shots > 1):
        result_sf = np.zeros((stat, dim, 2 ** nq))
        # Compute finite statistics
        for k in range(stat):
            for i in range(dim):
                result_sf[k, i] = FiniteStatistics(result_si[i], shots[k])
        result_sf = np.transpose(result_sf, (2, 0, 1))
        
    else:
        result_sf = np.zeros((2 ** nq, stat, dim))

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
        model = RidgeCV(alphas=np.logspace(-4, 2, 100))
    else:
        model = Ridge(alpha=0.0, solver = 'svd')

    model.fit(x_train, y_train)

    return model

def _output(data: MatrixLike, nq: int , global_properties: np.ndarray | None = None, patchwise_properties: np.ndarray | None = None, depth: int = 1, shots: int | MatrixLike = 1024, disable_progress_bar: bool = False) -> MatrixLike:
    """Factorized QELM Wrapper
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
         MatrixLike: Predicted values.
     """
    
    patches = data.shape[0]
    dim = data.shape[1]
    enc_dim = data.shape[2]
    stat = len(shots)
    
    if enc_dim > nq:
        raise ValueError("Number of encoded features must be less than or equal to number of qubits.")
    
    data_mapped = map_angle(data)
    par = np.random.uniform(0, np.pi, size=(patches, depth,  2 * nq))
    
    results_si = np.zeros((patches, 2 ** nq, dim))
    results_sf = np.zeros((patches, 2 ** nq, stat, dim))
    for i in tqdm(range(patches), desc="Processing patches", disable = disable_progress_bar):
        x_patch = data_mapped[i]
        results_si[i], results_sf[i] = Reservoir(nq, x_patch, par[i], depth, shots)
    results_sf = np.concatenate(results_sf, axis=0)
    results_si = np.concatenate(results_si, axis=0).T
    
    if global_properties is not None:
        print("Processing global properties...")
        par_mean = np.random.uniform(0, np.pi, size=(depth,  2 * nq))
        global_properties_mapped = map_angle(global_properties)
        result_global_si, result_global_sf = Reservoir(nq, global_properties_mapped, par_mean, depth, shots, disable_progress_bar)
        results_si = np.concatenate((results_si, result_global_si.T), axis=1)
        results_sf = np.concatenate((results_sf, result_global_sf), axis=0)
    
    if patchwise_properties is not None:
        par_patch = np.random.uniform(0, np.pi, size=(patches, depth,  2 * nq))
        patchwise_properties_mapped = map_angle(patchwise_properties)
        results_local_si = np.zeros((patches, 2 ** nq, dim))
        results_local_sf = np.zeros((patches, 2 ** nq, stat, dim))
        for i in tqdm(range(patches), desc="Processing local properties", disable = disable_progress_bar):
            x_patch_prop = patchwise_properties_mapped[i]
            results_local_si[i], results_local_sf[i] = Reservoir(nq, x_patch_prop, par_patch[i], depth, shots)
        results_local_sf = np.concatenate(results_local_sf, axis=0)
        results_local_si = np.concatenate(results_local_si, axis=0).T
        results_si =  np.concatenate((results_si, results_local_si), axis=1)
        results_sf =  np.concatenate((results_sf, results_local_sf), axis=0)
    
    results_sf = np.transpose(results_sf, (1, 2, 0))

    return results_si, results_sf

def _windows_testing(x: MatrixLike, xtest: MatrixLike, n_windows: int, y_special: MatrixLike, y_train_windows: MatrixLike, regularize: bool, test_size: float, features: int, W_special: Ridge | RidgeCV, special_index: int) -> MatrixLike:

    window = 1 / n_windows
    slide = window / 4
    W_windows_si = []
    negative_outlayers_si = y_special < - slide
    over_outlayers_si = y_special > 1 + slide
    for i in range(n_windows):
        mask = y_special <= (i + 1) * window + slide
        mask = (y_special >= i * window - slide) * mask
        if i == 0:
            mask = mask + negative_outlayers_si
        if i == n_windows - 2:
            mask = mask + over_outlayers_si
        W_windows_si.append(training(x[mask], y_train_windows[mask], regularize))
    
    ytest_si = np.zeros((test_size, features - 1))
    ytest_special_si = np.zeros((test_size))
    ytest_special_si = W_special.predict(xtest)
    negative_outlayers_si = ytest_special_si < - slide
    over_outlayers_si = ytest_special_si > 1 + slide
    for i in range(n_windows):
        mask = ytest_special_si <= (i + 1) * window + slide
        mask = (ytest_special_si >= i * window - slide) * mask
        if i == 0:
            mask = mask + negative_outlayers_si
        if i == n_windows - 2:
            mask = mask + over_outlayers_si
        ytest_si[mask] = W_windows_si[i].predict(xtest[mask])
    ytest_si = np.insert(ytest_si, special_index, ytest_special_si, axis = 1)

    return ytest_si


def FactorizedQELM(data: MatrixLike, targets: MatrixLike, nq: int , global_properties: np.ndarray | None = None, patchwise_properties: np.ndarray | None = None, depth: int = 1, shots: int | MatrixLike = 1024, train_size: float = 0.5, train_size_special: float | None = 0.3, n_windows:int | None = 5, special_index: int | None = 6, regularize: bool = True, disable_progress_bar: bool = False) -> MatrixLike:

    """Factorized QELM Wrapper
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
         MatrixLike: Predicted values.
     """

    data = np.array(data)
    targets = np.array(targets)
    dim = data.shape[1]

    if isinstance(shots, int):
        shots = np.array([shots], dtype = int)
    else:
        shots = np.array(shots, dtype = int)
        
    stat = len(shots)
    split_idx = int(dim * train_size)
    
    if train_size_special is None or n_windows is None:
        n_windows = None
        train_size_special = None
        split_idx_special = 0
        total_train_size = split_idx
        y_train_special = targets[: total_train_size]
        a = total_train_size
        b = dim
    else: 
        split_idx_special = int(dim * train_size_special)
        total_train_size = split_idx_special + split_idx
        y_train_special = targets[: split_idx_special, special_index].reshape(-1, 1)
        y_train_windows = np.delete(targets[split_idx_special : total_train_size], special_index, axis = 1)
        a = split_idx_special
        b = total_train_size
    
    if np.any(shots < 1):
        raise ValueError("All shots must be at least 1.")
    if not (0 < total_train_size < dim):
        raise ValueError("Train size must be between 0 and 1.")
    
    features = targets.shape[1]

    results_si, results_sf = _output(data, nq, global_properties, patchwise_properties, depth, shots, disable_progress_bar)
    W_special_si = training(results_si[: a], y_train_special, regularize)
    y_special_si = W_special_si.predict(results_si[a : b])

    if np.any(shots > 1):
        try:
            y_special_sf = np.zeros((stat, b - a, y_special_si.shape[1]))
        except:
            y_special_sf = np.zeros((stat, b - a))
        
        for i in tqdm(range(stat), desc="Finite statistics training", disable = disable_progress_bar):
            W_special_sf = training(results_sf[i, : a], y_train_special, regularize)
            y_special_sf[i] = W_special_sf.predict(results_sf[i, a : b])
    else:
        y_special_sf = np.zeros_like(y_special_si)
    
    if isinstance(n_windows, int):
        x = results_si[a : b]      
        xtest = results_si[total_train_size :]
        test_size = dim - total_train_size
        ypred_si = _windows_testing(x, xtest, n_windows, y_special_si, y_train_windows, regularize, test_size, features, W_special_si, special_index)
        if np.any(shots > 1):
            ypred_sf = np.zeros((stat, dim - total_train_size, features))
            for i in tqdm(range(stat), desc="Finite statistics training", disable = disable_progress_bar):
                x = results_sf[i, a : b]
                xtest = results_sf[i, total_train_size :]

                ypred_sf[i] = _windows_testing(x, xtest, n_windows, y_special_sf[i], y_train_windows, regularize, test_size, features, W_special_sf, special_index)
        else:
            ypred_sf = np.zeros_like(ypred_si)
    
    else:
        ypred_si = y_special_si
        ypred_sf = y_special_sf
    
    return ypred_si, ypred_sf