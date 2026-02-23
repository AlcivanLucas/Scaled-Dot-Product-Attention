import numpy as np

def softmax(x):

    # Subtrai o máximo de cada linha para estabilidade numérica
    row_maxes = np.max(x, axis=-1, keepdims=True)
    shifted_values = x - row_maxes
    exponentials = np.exp(shifted_values)
    row_sums = np.sum(exponentials, axis=-1, keepdims=True)
    return exponentials / row_sums

def scaled_dot_product_attention(Q, K, V):

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