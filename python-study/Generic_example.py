# 1. 导入 Generic 和 TypeVar
from typing import Generic, TypeVar

# 2. 创建一个类型变量 T
#    TypeVar("T") 就像一个占位符，代表“任何类型”
T = TypeVar("T")

# 3. 定义一个泛型类 Box[T]
#    我们告诉 Python，Box 是一个泛型类，它与类型 T 相关
class Box(Generic[T]):
    def __init__(self, item: T) -> None:
        """
        构造函数接收一个类型为 T 的物品。
        """
        self._item = item

    def get_item(self) -> T:
        """
        这个方法返回一个类型为 T 的物品。
        """
        return self._item

    def set_item(self, new_item: T) -> None:
        """
        这个方法只能接收类型为 T 的新物品。
        """
        self._item = new_item

# --- 现在我们来使用这个泛型盒子 ---

# 4. 创建一个“专门装整数”的盒子
#    我们将占位符 T 替换为具体的 int 类型
int_box: Box[int] = Box(123)

# 我们可以安全地取出整数并进行运算
# IDE 和类型检查器都知道 .get_item() 返回的是 int
value = int_box.get_item()
print(f"从整数盒子里取出的值: {value}")
print(f"进行数学运算: {value + 1}")

# 5. 创建一个“专门装字符串”的盒子
str_box: Box[str] = Box("hello")

# IDE 和类型检查器都知道 .get_item() 返回的是 str
text = str_box.get_item() + 1
print(f"从字符串盒子里取出的值: {text}")
print(f"进行字符串操作: {text.upper()}")


# 6. Generic 带来的好处：类型安全
#    如果我们试图往“整数盒子”里放一个字符串，类型检查器会立刻警告我们！
#    下面这行代码会（在 mypy 等工具中）报错：
#    Argument 1 to "set_item" of "Box" has incompatible type "str"; expected "int"
int_box.set_item("a string") # <--- 这里会产生类型错误警告

