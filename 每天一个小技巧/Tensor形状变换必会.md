# Tensor 变换形状

在 pytorch 中，我们都知道 Tensor 的形状必须 Match 在一起，才可以让整个代码跑通起来。

比如：对 `nn` 的一些操作而言

- `nn.Linear(in, out)` 要求输入 `x` 的 shape 必须是 `[..., in]`，对应输出则为 `[..., out]`
- `nn.Conv2d(out_channels, in_channels, kernel_size)` 要求输入 `x` 的 shape 必须是 `[batch_size, in_channels, height, width]`，对应输出则为 `[batch_size, out_channels, height, width]`

下面说一说必会的一些

## 前置必会：

首先我们要知道：pytorch 在多数reshape 操作中都要用上 -1，表示自动推算dim，就是只剩下一个不确定的时候，就可以用-1自动算出这一维度上变换后是几维。

### 存储空间连续问题

```py
import torch
x = torch.arange(15).reshape(3,5)
x
# tensor([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14]])
x.stride()
# out: (5, 1)
```

`stride` 的意思是指张量的步长。是初始创建张量后，存储空间与外界访问张量用的索引的联系之一。以上面为例

- `stride[0]` 就是元素 `(0, 0)` 和 `(1, 0)` 之间间隔几个元素。上面结果显示就是 0。
- `stride[1]` 就是元素 `(0, 0)` 和 `(0, 1)` 之间间隔几个元素。上面结果显示就是 1。

如果我们把 storage 打出来：

```py
x.storage()
# out: 0
# 1
# 2
# ...
# 14
```

也就是说，你放到底层，这个Tensor是按照这个顺序去储存的。

我们经常会听到一个例子，就是是否连续。这里面也就是 `is_contiguous()`。那在这里什么叫连续？我们仍然以上面的例子：

```py
x = torch.arange(15).reshape(3, 5)
x.shape: torch.Size([3, 5])
x.stride(): (5, 1)
x.is_contiguous(): True
```

实际上判断连续条件：`stride[i] = stride[i + 1] * size[i + 1]`。这里面
`stride[0] = 5, stride[1] = 1, size[0] = 3, size[1] = 5`，所以 `stride[0] = stride[1] * size[1]`，所以 `stride[0] = 1 * 5 = 5`，所以是连续的。

现在我们颠倒一下顺序：

```py
y = x.permute(1, 0)
y.shape: torch.Size([5, 3])
y.stride(): (1, 3)
stride[0] != stride[0 + 1] * size[0 + 1]
```

也就是因此，`y` 就不是连续 contiguous() 的了。实际上我们把 `y` 打印出来看看：

```py
y
# tensor([[ 0,  5, 10],
#         [ 1,  6, 11],
#         [ 2,  7, 12],
#         [ 3,  8, 13],
#         [ 4,  9, 14]])
y.storage()
# 0 
# 1
# ..
# 14
```

可以看到，`y` 仍然按照 `x` 的存储 storage 顺序去存的。

### 改变维度

**`view` 和 `reshape`的区别**

`view` 和 `reshape`：实际上这二者是相似的，但对于底层的不同在于：`view` 只能对 `is_contiguous()` 连续的 tensor 进行操作。如果不连续，则 `view` 就会报错。不过 `reshape` 比 `view` 更通用些，`view` 可以的 `reshape` 都可以做。

如果遇到一个 tensor，其不连续，`view` 就会报错，`reshape` 就不会。但是 `reshape` 会新开辟一个数据的存储空间。属于对数据做了一个拷贝，而不是赋值 A = B 只是让 A 和 B 指向了同一个引用。

```py
x = torch.arange(15).reshape(3, 5)
y = x.permute(1, 0)
x.is_contiguous() # True
y.is_contiguous() # False'

x.view(15) # 不报错
y.view(15) # 报错

z = y.reshape(15) # 不报错
z.is_contiguous() # True
```

实际上你可以看存储空间的地址就发现不一样了！

```py
x.data_ptr(), y.data_ptr(), z.data_ptr(), y.contiguous().data_ptr()
# (5059886716224, 5059886716224, 5059886716416, 5059886716608)
```

也就是数说，你的 `permute` 转换视图，并没有改变内部的存储空间顺序，而reshape就改变了，变得连续了。实际上，你也可以

```py
y.contiguous().view(15)
```

如果 tensor 连续就返回原来存储地址的，如果不连续的时候就会开辟一个新的。

虽然说 `reshape` 更灵活，优先使用 `reshape` 以避免手动处理连续性。

- `.contiguous().view()` 若原张量不连续就会改变存储地址，须显式处理连续性
- `.reshape()` 自动处理，虽然可以适用更灵活，但可能存在隐式无意间更多存储开销。

如果输入的不连续，前者显式复制数据到连续布局，再返回视图，隐式自动隐式执行contiguous()

**`transpose` 和 `permute` 的区别**

```py
x = torch.arange(15).reshape(3, 5)
x.transpose(0, 1) # x改变连续
x.permute(1, 0) # 改变连续

x.transpose(0, 1).is_contiguous() # False
x.permute(1, 0).is_contiguous() # Fals3

x.transpose(0, 1).data_ptr() == x.permute(1, 0).data_ptr() # True
```

实际上，会发现 `transpose` 和 `permute` 这两者都会把原来连续的 `tensor` 变得不连续。一般都是交换了维度。

不过值得一提的是，`transpose` 仅限于交换 2 个维度，且 `transpose(a, b)` 和 `transpose(b, a)` 没有区别，都是交换第 `a` 维 和第 `b` 维。而 `permute` 可以一次按顺序把多个维度重排，但是里面必须是全部的维度，且必须写排后的顺序才行！前者适合简单数据，后者更适合复杂数据。
