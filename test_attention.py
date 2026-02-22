import numpy as np
import numpy.testing as npt
from attention import scaled_dot_product_attention

def test_weights_sum_to_one():
    print("\n--- Teste: Soma dos pesos de atenção por linha ---")
    Q = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 1, 0, 0]], dtype=np.float64)
    K = np.array([[0, 1, 1, 0],
                  [1, 0, 0, 1],
                  [1, 1, 0, 0]], dtype=np.float64)
    V = np.array([[1, 2],
                  [3, 4],
                  [5, 6]], dtype=np.float64)

    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    row_sums = np.sum(attention_weights, axis=-1)
    try:
        npt.assert_array_almost_equal(row_sums, np.ones_like(row_sums), decimal=6)
        print("PASSED")
    except AssertionError:
        print("FAILED")

def test_output_shape():
    print("\n--- Teste: Shape do output ---")
    Q = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 1, 0, 0]], dtype=np.float64)
    K = np.array([[0, 1, 1, 0],
                  [1, 0, 0, 1],
                  [1, 1, 0, 0]], dtype=np.float64)
    V = np.array([[1, 2],
                  [3, 4],
                  [5, 6]], dtype=np.float64)

    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    expected_shape = (Q.shape[0], V.shape[1])
    if output.shape == expected_shape:
        print(f"PASSED: Shape {output.shape} == {expected_shape}")
    else:
        print(f"FAILED: Shape {output.shape} != {expected_shape}")

def test_numerical_correctness():
    print("\n--- Teste: Correção numérica ---")
    Q = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 1, 0, 0]], dtype=np.float64)
    K = np.array([[0, 1, 1, 0],
                  [1, 0, 0, 1],
                  [1, 1, 0, 0]], dtype=np.float64)
    V = np.array([[1, 2],
                  [3, 4],
                  [5, 6]], dtype=np.float64)

    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    # Cálculo manual esperado
    d_k = K.shape[-1]
    scores = np.matmul(Q, K.T)
    scaled_scores = scores / np.sqrt(d_k)
    expected_attention_weights = softmax(scaled_scores)
    expected_output = np.matmul(expected_attention_weights, V)

    print(f"Input Q:\n{Q}")
    print(f"Input K:\n{K}")
    print(f"Input V:\n{V}")
    print(f"Attention weights:\n{attention_weights}")
    print(f"Output:\n{output}")

    try:
        npt.assert_array_almost_equal(output, expected_output, decimal=6)
        npt.assert_array_almost_equal(attention_weights, expected_attention_weights, decimal=6)
        print("PASSED")
    except AssertionError:
        print("FAILED")

def softmax(x):
    row_maxes = np.max(x, axis=-1, keepdims=True)
    shifted_values = x - row_maxes
    exponentials = np.exp(shifted_values)
    row_sums = np.sum(exponentials, axis=-1, keepdims=True)
    return exponentials / row_sums

if __name__ == "__main__":
    test_weights_sum_to_one()
    test_output_shape()
    test_numerical_correctness()
    print("\nTodos os testes concluídos!")
