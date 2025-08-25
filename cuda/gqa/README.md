# Grouped Query Attention (GQA) 计算流程

Grouped Query Attention（GQA）在多头自注意力的基础上，将查询头分为 \(G\) 组，每组共享同一套键（Key）和值（Value），以减小计算和内存开销。

---

## 1. 输入张量与分组

令批量大小为 \(B\)、序列长度为 \(L\)、查询头数为 \(H_q\)、键/值头数为 \(H_{kv}\)、每头维度为 \(d\)。  

将 \(H_q\) 个查询头分为 \(G\) 组，每组有 \(\tfrac{H_q}{G}\) 个查询头，所有组共享同一套 \(H_{kv}\) 个键/值头。

- 查询张量  
  \[
    Q \in \mathbb{R}^{B \times H_q \times L \times d}
  \]

- 键张量  
  \[
    K \in \mathbb{R}^{B \times H_{kv} \times L \times d}
  \]

- 值张量  
  \[
    V \in \mathbb{R}^{B \times H_{kv} \times L \times d}
  \]

---

## 2. 组内点积与缩放

对每个查询组内的所有头，与共享的键头做点积，并按 \(\sqrt{d}\) 进行缩放：  

\[
  A = \frac{Q \times K^{\mathsf{T}}}{\sqrt{d}}
  \quad\in\quad
  \mathbb{R}^{B \times H_q \times L \times L}
\]

这里的乘法沿序列维度完成，每个查询位置会与所有键位置进行交互。

---

## 3. Softmax 归一化

对缩放后的注意力分数 \(A\) 在最后一维做 Softmax 归一化，使每个查询位置的权重和为 1：  

\[
  D = \mathrm{softmax}(A)
  \quad\in\quad
  \mathbb{R}^{B \times H_q \times L \times L}
\]

其中  
\[
  D_{b,h,i,j}
  = \frac{\exp\bigl(A_{b,h,i,j}\bigr)}
         {\sum_{k=1}^{L} \exp\bigl(A_{b,h,i,k}\bigr)}
\]

---

## 4. 加权和输出

将归一化权重 \(D\) 与共享的值张量 \(V\) 做矩阵乘法，得到最终每头的输出：  

\[
  O = D \times V
  \quad\in\quad
  \mathbb{R}^{B \times H_q \times L \times d}
\]

不同查询组均使用同一 \(V\) 进行加权和，但因组内查询头数减少，整体计算和内存占用有所降低。

---

## 5. 性能与精度权衡

| 特性          | 多头注意力 (MHA)                         | 分组查询注意力 (GQA)                                 |
|--------------|-----------------------------------------|------------------------------------------------------|
| 查询头数     | \(H_q\)                                 | \(H_q\)                                              |
| 键/值头数    | \(H_q\)                                 | \(\tfrac{H_q}{G}\)                                   |
| 计算复杂度   | \(\mathcal{O}(H_q \cdot L^2 \cdot d)\)   | \(\mathcal{O}(H_q \cdot L^2 \cdot d)\)                |
| 内存占用     | 较高                                    | 随 \(G\) 增大而显著降低                              |
| 精度影响     | 无                                      | 与 \(G\) 成正比，通常仅有轻微下降                    |

通过调整分组数 \(G\)，可以在推理速度、内存占用和模型准确度之间取得平衡。

---


