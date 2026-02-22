# Scaled Dot-Product Attention — LAB P1-01

## Descrição
Este repositório contém uma implementação do mecanismo de **Scaled Dot-Product Attention**, um componente fundamental da arquitetura Transformer, conforme descrito no artigo seminal "Attention Is All You Need" [1]. O objetivo é demonstrar a lógica de transformação das matrizes de Query (Q), Key (K) e Value (V) para calcular os pesos de atenção e a saída final, utilizando exclusivamente a biblioteca NumPy para operações de álgebra linear.

## Como Rodar
Para executar os testes e verificar a implementação, siga os passos abaixo:

1. **Crie um ambiente virtual** (para evitar conflitos com o Python do sistema):
   ```bash
   python3 -m venv venv
   ```

2. **Ative o ambiente virtual**:
   ```bash
   source venv/bin/activate
   ```

3. **Instale NumPy**:
   ```bash
   pip install numpy
   ```

4. **Execute o script de testes**:
   ```bash
   python test_attention.py
   ```
   O script `test_attention.py` irá imprimir os inputs, outputs e o status de cada teste, verificando a correção dimensional, a soma dos pesos de atenção e a precisão numérica dos resultados.

5. **Desative o ambiente virtual** (opcional, quando terminar):
   ```bash
   deactivate
   ```

## Explicação do Scaling Factor
No mecanismo de Scaled Dot-Product Attention, os scores de similaridade entre as queries e as keys são divididos por $\sqrt{d_k}$, onde $d_k$ é a dimensão dos vetores de chave (Key). Este fator de escala é crucial para a estabilidade do treinamento de modelos de atenção. Sem essa normalização, para valores grandes de $d_k$, o produto escalar $Q K^T$ pode resultar em valores muito grandes. Isso, por sua vez, empurraria a função softmax para regiões onde seus gradientes são extremamente pequenos (próximos de zero), dificultando o aprendizado do modelo. A divisão por $\sqrt{d_k}$ ajuda a manter a variância dos produtos escalares em uma faixa mais controlada, garantindo que o softmax opere em uma região com gradientes mais significativos e, consequentemente, promovendo um treinamento mais estável e eficaz [1].

## Exemplo de Input/Output
Utilizamos as seguintes matrizes de exemplo para Q, K e V:

**Query (Q)**:
```
[[1. 0. 1. 0.]
 [0. 1. 0. 1.]
 [1. 1. 0. 0.]]
```

**Key (K)**:
```
[[0. 1. 1. 0.]
 [1. 0. 0. 1.]
 [1. 1. 0. 0.]]
```

**Value (V)**:
```
[[1. 2.]
 [3. 4.]
 [5. 6.]]
```

Com base nesses inputs, os **Attention Weights** calculados são:
```
[[0.33333333 0.33333333 0.33333333]
 [0.33333333 0.33333333 0.33333333]
 [0.27406862 0.27406862 0.45186276]]
```

E o **Output da Atenção** resultante é:
```
[[3.         4.        ]
 [3.         4.        ]
 [3.35558829 4.35558829]]
```

## Referência
[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems, 30. [Link para o paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)