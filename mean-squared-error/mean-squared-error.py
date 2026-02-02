import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Convert inputs to NumPy arrays
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Ensure shapes match (return None if mismatch)
    if y_pred.shape != y_true.shape:
        return None
    
    # Calculate MSE: (1/N) * Σ(y_pred - y_true)²
    mse = np.mean((y_pred - y_true) ** 2)
    
    # Return as float
    return float(mse)