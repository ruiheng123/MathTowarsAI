# llama 1

---

[![arXiv](https://img.shields.io/badge/arXiv-2406.09246-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2302.13971)  [![Github Pages](https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white)](https://github.com/facebookresearch/llama)[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/meta-llama)

每个人对大语言模型的 github 代码学习，想必八成都是从 llama 系列开始的！毕竟当初，为了对抗 OpenAI 在 GPT 上的闭源。

很多人如果想去大模型公司实习，那简历里见过的最多的写进入的内容多半是：**微调了一个 llama 7b**。然而还有不少人不论是让手撕 RMSNorm、RoPE 这些 llama 的trick，仍然存在“一不知公式，二不会手撕，三不知好处”

这里简称新“一问三不知”。对于训练过程更是只见过interface或者仅仅调API，却不知道扒一下里面的.....这里我们将扫盲一下！

(这种新“一问三不知”，是非常严肃的问题。比如在 pytorch 里不知道 `model.train()` 和 `model.eval()` 是为了通知 `Dropout` 和 `BatchNorm` 变成训练模式/测试模式；或者学 RL 却不知道 PG 公式推导过程......)

我们先从 Llama 1 开始，来了解一下整个语言模型，结合公式，在代码实现上是什么底层逻辑：

## Tokenizer 

大语言模型都有一个 Tokenizer，将自然语言文本转化为词元 Tokens 为 Encode；将 Tokens 转化为自然语言文本为 Decode。

llama 采用的 Tokenizer 是 `SentencePiece` 的。下面我们看看 Tokenizer 的部分代码（为了减少行数，我把原作者的注释都删去，改成自己写的注释，这样显得行数不那么多。另外源代码仅在注释上有改动，为明白过程，不会改变底层逻辑。

```py
import os
from logging import getLogger
from typing import List
logger = getLogger()

from sentencepiece import SentencePieceProcessor

class Tokenizer:
    def __init__(self, model_path: str):
        # 加载模型
        assert os.path.isfile(model_path), f"你的 Tokenizer 从路径：{model_path} 失败，请检查！"
        logger.info(f"Reloaded SentencePiece model from {model_path}")
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # n_words: 总共单词有多少
        # BOS：开始符 Beginning of Sentence
        # EOS：结束符 End of Sentence
        # pad_id：填充符
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")

        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        # 这里源代码为 type(s) is str，isinstance也可。isinstance 可以判断子类，而 type 不能。
        assert isinstance(s, str), f"输入的文本 s 必须是字符串！"
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t  # 遇到开始符，就在 t 前面加上开始符
        if eos:
            t = t + [self.eos_id]  # 遇到结束符，就在 t 后面加上结束符
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

```

这个也就是 llama 1 的 Tokenizer 了。可以看出，其用的是 SentencePiece，这个是由Google开发的一种通用的 Tokenizer。SentencePiece 的特点是：
- 将句子拆分为小的piece，然后合并成 token。
- 特点：直接从句子中去训练 Encode 和 Decode，无需预先分词，句子直接视为 Unicode 字符序列，分割速度快且轻量。

## Model 部分

### RMSNorm 

不少大模型公司让你手撕代码，现在也肯定有一个是 RMSNorm。往常多是 LayerNorm，毕竟在序列性的数据里，LayerNorm 效果更好。RMSNorm 是一种新的归一化方法。

RMSNorm主要面向Layer Norm改进，