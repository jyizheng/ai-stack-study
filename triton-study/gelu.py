import matplotlib.pyplot as plt
import numpy as np

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

# 生成输入数据
x = np.linspace(-5, 5, 100)

# 计算 GELU
y = gelu(x)

# 绘制图像
plt.plot(x, y, label="GELU")
plt.xlabel("x")
plt.ylabel("GELU(x)")
plt.title("GELU Activation Function")
plt.legend()
plt.grid(True)
plt.show()