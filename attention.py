import numpy as np

def softmax(x):
    """
    Calcula a função softmax com truque de estabilidade numérica.

    Args:
        x (np.ndarray): Matriz de entrada 2D.

    Returns:
        np.ndarray: Matriz com softmax aplicado linha a linha.
    """
    # Subtrai o máximo de cada linha para estabilidade numérica
    row_maxes = np.max(x, axis=-1, keepdims=True)
    shifted_values = x - row_maxes
    exponentials = np.exp(shifted_values)
    row_sums = np.sum(exponentials, axis=-1, keepdims=True)
    return exponentials / row_sums