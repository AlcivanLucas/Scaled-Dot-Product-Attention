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

def scaled_dot_product_attention(Q, K, V):
    """
    Implementa o mecanismo de Scaled Dot-Product Attention.

    Fórmula: Attention(Q,K,V) = softmax(QK^T / √d_k) V

    Args:
        Q (np.ndarray): Matriz de Query.
        K (np.ndarray): Matriz de Key.
        V (np.ndarray): Matriz de Value.

    Returns:
        tuple: (output, attention_weights)
    """
    # 1. Extrair d_k
    d_k = K.shape[-1]
    
    # 2. Calcular scores = QK^T
    scores = np.matmul(Q, K.swapaxes(-2, -1))
    
    # 3. Dividir por √d_k
    scaling_factor = np.sqrt(d_k)
    scaled_scores = scores / scaling_factor
    
    # 4. Aplicar softmax
    attention_weights = softmax(scaled_scores)
    
    # 5. Multiplicar por V
    output = np.matmul(attention_weights, V)
    
    # 6. Retornar
    return output, attention_weights