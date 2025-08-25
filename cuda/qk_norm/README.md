# QK Normalization 计算流程

QK Normalization 是在标准缩放点积注意力（Scaled Dot-Product Attention）的基础上，对查询 (Q) 与键 (K) 进行归一化处理，以抑制注意力分数的任意饱和现象并提升模型稳定性和性能。

---

## 1. 输入张量

- 批量大小：\(B\)  
- 序列长度：\(L\)  
- 头数：\(H\)  
- 每头维度：\(d\)  

输入张量：  
\[
Q \in \mathbb{R}^{B \times H \times L \times d},\quad
K \in \mathbb{R}^{B \times H \times L \times d},\quad
V \in \mathbb{R}^{B \times H \times L \times d}
\]

---

## 2. 对 Q 和 K 进行归一化

在每个头的特征维度上，分别对查询向量和键向量做 \(L_2\) 归一化（或均方根归一化），公式如下：

\[
\hat{Q}_{b,h,i,:}
= \frac{Q_{b,h,i,:}}{\sqrt{\sum_{c=1}^{d}Q_{b,h,i,c}^2 + \varepsilon}}
\quad,\quad
\hat{K}_{b,h,j,:}
= \frac{K_{b,h,j,:}}{\sqrt{\sum_{c=1}^{d}K_{b,h,j,c}^2 + \varepsilon}}
\]

其中 \(\varepsilon\) 是防止除零的小常数。

---

## 3. 计算归一化后点积

将归一化后的 \(\hat{Q}\) 与 \(\hat{K}\) 做点积，并引入可学习缩放参数 \(\gamma\)（替代原先的 \(1/\sqrt{d}\)）：

\[
A_{b,h,i,j}
= \gamma_h \;\bigl(\hat{Q}_{b,h,i,:}\cdot \hat{K}_{b,h,j,:}^{\mathsf{T}}\bigr)
\quad\in\quad
\mathbb{R}^{B \times H \times L \times L}
\]

每个头拥有独立的缩放系数 \(\gamma_h\)。

---

## 4. Softmax 归一化注意力权重

对每个查询位置 \(i\) 在最后一维做 Softmax：

\[
D_{b,h,i,j}
= \frac{\exp\bigl(A_{b,h,i,j}\bigr)}
       {\sum_{k=1}^{L}\exp\bigl(A_{b,h,i,k}\bigr)}
\quad\in\quad
\mathbb{R}^{B \times H \times L \times L}
\]

保证每个查询位置的注意力权重和为 1。

---

## 5. 加权和值输出

使用归一化权重 \(D\) 对原始值张量 \(V\) 加权求和：

\[
O_{b,h,i,:}
= \sum_{j=1}^{L} D_{b,h,i,j}\;V_{b,h,j,:}
\quad\in\quad
\mathbb{R}^{B \times H \times L \times d}
\]

生成最终注意力输出。

---

## 6. Mirage 示例代码

下面示例展示了在 Mirage 中，如何将 RMS 归一化应用于 Q 与 K 后，再执行注意力计算的流程：

```python
import torch
import mirage as mi

graph = mi.new_kernel_graph()
Q = graph.new_input(dims=(2, 256, 64), dtype=mi.float16)
K = graph.new_input(dims=(2, 64, 4096), dtype=mi.float16)
V = graph.new_input(dims=(2, 4096, 64), dtype=mi.float16)

# 对 Q 和 K 做均方根归一化
Q_norm = graph.rms_norm(Q)
K_norm = graph.rms_norm(K)

# 计算点积注意力
A = graph.matmul(Q_norm, K_norm)
E = graph.exp(A)
S = graph.reduction(E, 2)
D = graph.div(E, S)

# 加权和值并输出
O = graph.matmul(D, V)
graph.mark_output(O)

optimized_graph = mi.superoptimize(graph)
```

---

# TODO

了解 QK Normalization 在不同任务（如低资源翻译、视觉-语言模型等）中性能提升的具体实验结果，可以进一步阅读 QKNorm 原始论文和相关实现。

