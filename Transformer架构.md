# 简介

此笔记讲解了最初的Transformer模型，即Encoder-Decoder Transformer模型，最初用来作为翻译器。

Encoder丢入待翻译的英文token序列，Decoder丢入目标语言如法语的token序列。

# Transformer架构

<img src="/Users/wmyh0416/Library/Application Support/typora-user-images/image-20260204145630220.png" alt="image-20260204145630220" style="zoom:20%;" />



​	这张图片展示了Transformer的架构：Encoder+Decoder，是2017年用来解决了翻译的核心问题。它解决了长距离依赖，对齐能力更强，更好扩展，后来衍生出了Decoder-Only的GPT、LLaMA、Gemini。

​	阅读下面[《Attention注意力机制是什么？》](#Attention注意力机制是什么？)我们就已经了解了Tokenizer、Embedding、PE和Attention是什么了。

​	下面我们将进入[Add&Norm](#Add & Norm（Add & Normalization）)阶段。阅读完此部分，我们就可以知道Transformer Encoder Block是什么了，并且一个Transformer Encode Block的可训练参数总数为：
$$
\begin{aligned}
\mathrm{num}(\text{Attention} + \text{Norm} + \text{FFN})
&= H \times (4 \times d_{\text{model}} \times d_k)  + 2 \times 2 d_{\text{model}}  + 2 \times d_{\text{model}}\times d_{ff} \\
&= 12 d_{\text{model}}^2 + 4 d_{\text{model}}
\end{aligned}
$$
​	现在我们就要学习架构右边的Transformer Decoder Block了，这里也拥有自己的Embedding矩阵和自己的Vocab（即目标翻译语言的词表，也是经过Byte-level BPE习得的）。因为是作为翻译器，当前法语token**（本文我将目标语言定位法语）**不能看到下一个法语token是什么，Decoder要学习的是：
$$
P(y_t) =P(y_t \mid y_{<t},\ x)
$$
 	即第t个目标词，只能依赖：1.之前生成的目标词语；2.Encoder编码的源句；3.不能依赖未来的目标词语。则这里我们就需要利用[Masked Multi-Head Attention机制](#Masked Multi-Head Attention)，使得法语token只能对自己即之前的词语打分，即让其对自己之后的token打分降低到最小，这样模型就看不到自己位置之后的词语，这样模型就学不会推未来，只能学会从过去推导未来。

​	在Decoder的法语token序列经过一层Masked Multi-Head Attention和Add&Norm之后，就到了Encoder和Decoder序列的交汇处了，这里我们是利用法语token序列的查询矩阵Q和来自于Encoder的K、V矩阵，进行信息交互，也就是下一个法语token的预测，只能依赖于当前即之前的法语token和所有的英文序列。这就是著名的[Cross-Attention](#Cross Attention Module)。

​	了解完这些之后我们就可以对数据进行训练了，左右各丢进去一个一样语义的英文句子和法语句子，但是模型中的参数是通过什么进行调节的呢？我们通过什么指标来知道模型[参数该怎么样调节](#Transformer参数调节)？

​	最终训练完模型之后，我们得到了属于该模型的权重分布，就可以利用这个权重开始预测了，预测的本质就是一个数学公式：$P(y_i|y_1,y_2,...,y_{i-1})$， 即最终得到了是针对一个vocab大小的概率分布，但是如何从这个分布中选择一个token，也是我们要研究的方向，这个研究方向称为[Decoding Strategy](#Decoding)。



# Attention注意力机制是什么？

​	Attention里面最重要的公式是
$$
Attention(Q,K,V)=softmax(\frac {QK^T}{\sqrt{d_k}})V
$$
​	但是在讲Attention之前我们需要先知道Tokenization、Embedding和Positional Encoding（PE）是什么？因为数据要经过这些操作之后才送入Attention层进行操作。

​	在Transformer模型中，**Inputs其实是一段token id序列**，上面的Transformer架构图中省略了关键的**原始数据输入经过Tokenizer进行查表（Vocabulary）得到token id序列的Tokenization过程**。下面我们就先学习什么是[Tokenizer](#Tokenizer)？

​	分词训练完成之后我们得到Tokenizer就可以实施encode和decode两大功能了。文本经过Tokenizer的encode之后会变成一串id序列，这个id序列就是我们embedding层的原始输入，下面我们需要先学习什么是[Embedding](#Embedding)？

​	知道Embedding是什么后，我们就知道先前我们得到的id序列经过embedding层之后会得到 seq_len * d_model的矩阵，但是数据真正进入到attention之前还需要经过[PE（Positional Encoding）](#PE（Positional Encoding）)，我们需要先知道什么是PE？

​	下面就真正进入到Attention了，经过PE处理过的$X$ 矩阵就是我们的attention层输入数据。然后我们可以计算出$Q$，$K$，$V$三个矩阵。

## 单头注意力机制（Single-Head Attention）

给定输入：$X\in\mathbb{R}^{n\times d_{model}}$ ，$n$ 为序列长度，即 $n$ 个token。
$$
\mathbf{X}
=
\begin{bmatrix}
\mathbf{x}_1 \\
\mathbf{x}_2 \\
\vdots \\
\mathbf{x}_n
\end{bmatrix}
=
\begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1d_{model}} \\
x_{21} & x_{22} & \cdots & x_{2d_{model}} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \cdots & x_{nd_{model}}
\end{bmatrix}
\quad
\text{(each row corresponds to one token)}
$$
计算:$Q=XW_Q$ ，$K=XW_K$ ， $V=XW_V$ ,其中:$W_Q,W_K,W_V\in \mathbb{R}^{d_{model} \times d_{model}}$，则 $Q,K,V\in \mathbb{R}^{n\times d_{model}}$。

 $Q,K,V\in \mathbb{R}^{n\times d_{model}}$三个矩阵的第i行都代表第i个token的向量。

然后attention输出：$Attention(Q,K,V)=softmax(\frac {QK^T}{\sqrt{d_k}})V$（$d_k$是K矩阵向量的维度，即key向量的维度，也是query向量的维度，$d_q=d_k=d_v$。）

1. **原始注意力得分矩阵（Raw Attention Score Matrix）**是$QK^T$（又称为原始相似度矩阵Similarity Matrix），其维度是$n\times n$代表每两个token间的原始打分（Raw Score）。**（RoPE就是作用在Q和K上的）**

> 如第m个token对第n个token的原始相似度打分\原始打分就是$Q[m]\cdot K[n]^T$，这是一个常数，即$1\times 1$ 维。

2. **缩放注意力得分矩阵（Scaled Attention Score Matrix）**是$\frac {QK^T}{\sqrt{d_k}}$矩阵。

对Raw Attention Score Matrix矩阵里的每一个元素都除以$\sqrt{d\_k}$，就会得到Scaled Attention Score Matrix。

3. **注意力矩阵（Attention Matrix）**是经过softmax层后的$softmax(\frac {QK^T}{\sqrt{d_k}})$矩阵，每一行是一个概率分布，每一行的和等于1，每一个元素>=0。

softmax函数对Scaled Attention Score Matrix的处理如下：

$Scaled\,Attention\,Score\,Matrix=S=\frac{QK^T}{\sqrt{d_k}}\quad( S\in\mathbb{R}^{n\times n})$

$softmax(S)$ 是对$S$ 中的每一个元素$S_{ij}$ 做$A_{ij}=\frac{e^{S_{ij}}}{\sum_{k=1}^{N}e^{S_{ik}}}\quad$，N是$seq\_len$ 即是一句话中的token总数。 

综上最终我们的到的$O=Attention\ matrix=softmax(S)_{(n\times n)}\cdot V_{(n\times d_k)}=A_{(n\times n)}\cdot V_{(n\times d_k)}=O_{(n\times d_k)}=O_{(n\times d_{model})}$

则$O_{(n\times d_{model})}$就是attention层的输出！

> 综上Single-Head Attention中的可训练矩阵有3个，分别是$W_Q,W_K,W_V$ 。

## 多头注意力机制（Multi-Head Attention）

我们会有$H$ 个head，每一个head会拥有自己的可训练参数$W_Q^{i}\in\mathbb{R}^{d_{model}\times d_k},W_K^i\in\mathbb{R}^{d_{model}\times d_k},W_V^i\in\mathbb{R}^{d_{model}\times d_k}$其中$d_k=d_q=d_v=\frac{d_{model}}{H}$。

这些参数相当于把X映射到每一个头里

即：$Q^i=X_{(n\times d_{model})}\cdot W_Q^i=Q^i_{(n\times d_k)}$

$K^i=X_{(n\times d_{model})}\cdot W_K^i=K^i_{(n\times d_k)}$

$V^i=X_{(n\times d_{model})}\cdot W_V^i=V^i_{(n\times d_k)}$

分别在各自头里计算后会得到各自头$head(i)$ 里的attnetion输出$O^i=O^i_{(n\times d_k)}$

最后再把这些$O^i_{(n\times d_k)}$ 横向拼接就得到了$Concat(O^1_{(n\times d_k)} \quad O^2_{(n\times d_k)}\quad  ... \quad O^H_{(n\times d_k)})\in\mathbb{R}^{n\times d_{model}}$ ，但是每个头之间的$O^i$没有交互信息，这里还要再设计一个交互矩阵$W_O\in\mathbb{R}^{d_{model}\times d_{model}}$ 。

$Multi\text{-}Head\ attention\ output=MultiHead(X)=Concat(O^1_{(n\times d_k)} \quad O^2_{(n\times d_k)}\quad  ... \quad O^H_{(n\times d_k)})\cdot W_O$

> 综上Multi-Head Attention中的可训练矩阵有4个，分别是$W_Q,W_K,W_V,W_O$

## Masked Multi-Head Attention（Causal Mask）机制

原self attention机制是：Raw Attention Score Matrix $\to$ Scaled Attention Score Matrix $\to$ Attention Matrix

Masked是对缩放注意力得分矩阵（Scaled Attention Score Matrix）进行操作：
$$
S'=S+M,\quad M=\begin{bmatrix}0&-\infty&-\infty&-\infty\\0&0&-\infty&-\infty\\0&0&0&-\infty\\0&0&0&0\end{bmatrix}
$$
这样Scaled Attention Score Matrix在经过Softmax层之后，注意力矩阵（Attention Matrix）中，每一个token对自身位置之后的token的关注度就变成了一个接近于0的数，即当前token看不到之后是什么东西了，这样就会避免Decoder中的token从自身位置之后的token序列中学习到信息，强制让他们只根据**过去**和**现在**去**推导未来**。

## Cross-Attention Module

Encoder和Dcoder虽然具有不同大小的词表，但他们的token向量维度是一样的，都是$d_{model}$ ，从Encoder来的矩阵是$H_{encoder}\in\mathbb{R}^{N_s\times d_{model}}$ ，Decoder中经过Masked Multi-Head Attention出来矩阵是$H_{decoder}\in\mathbb{R}^{N_t\times d_{model}}$，其中$N_s$ 是源句token序列长度，$N_t$ 是目标句token序列长度。

计算步骤：

1. $Q$ 矩阵来自于Decoder：$Q=H_{decoder}\cdot W_Q\in\mathbb{R}^{N_t\times d_k}$
2. $K、V$ 矩阵来自于Encoder：$K=H_{encoder}\cdot W_K\in\mathbb{R}^{N_s\times d_k},\ V=H_{encoder}\cdot W_V\in\mathbb{R}^{N_s\times d_k}$
3. Raw Attention Score Matrix：$Q\cdot K_T\in\mathbb{R}^{N_t\times N_s}$ ，这是每一个法语token对每一个英文token的原始打分
4. Attention Matrix：$softmax(\frac{QK^T}{\sqrt{d_k}})V\in\mathbb{R}^{N_t\times d_k}$
5. 横向拼接H个Head的输出：$O=Concat(O^1,O^2,...,O^H)\cdot W_O\in\mathbb{R}^{N_t\times d_{model}}$

# Tokenizer

Tokenizer的目的是对语料库/原始数据进行分词（分Token）,得到词表Vocabulary，这样以后我们的输入就可以直接**经过Tokenizer做encode得到Token id序列**，也就是Embedding层的输入。

<img src="/Users/wmyh0416/Library/Application Support/typora-user-images/image-20260213205201914.png" alt="image-20260213205201914" style="zoom:50%;" />

Vocabulary的数据类型其实就是（Token，id）的集合。

>	1. Tokenizer training / Vocabulary construction: We train the tokenizer on a corpus to learn a subword vocabulary. 即在现有数据中学习训练如何分token，这里会涉及到不同的Tokenizer算法
>	1. Tokenization / Mapping text to token IDs: The trained tokenizer is used to encode input text into token IDs. 即用我们训练好的Tokenizer对输入文本做映射得到Token id序列
>
>（注：Tokenizer指The thing that does tokenization;  Tokenization指The process of converting text into tokens）

Tokenizer的经典算法有很多，也就是说学习分词有很多不同策略，下面我会介绍一些经典的Tokenizer算法。

1. Word-level Tokenizer：最早最直观的tokenizer，把文本按“词”直接分开，如“I love NLP”可以分成“I”、“love”、“NLP”三个token，即**先分词，建立词表Vocabulary，每个词分配一个id**。但是会面临Out-of-Vocabulary（OOV）问题，即输入里面出现了一个token/字符串但是我们的vocab里面根本没有他，找不到与之对应的id。
2. Character-level BPE（字符级BPE）：**以字符为最小原子，在字符序列上运行BPE算法**，如对“banana”进行Character-level BPE算法，我们先把"banana"拆分成["b", "a", "n", "a", "n", "a"]，第一步合并出现频率最高的pair即("a","n") → "an"，合并后序列变为["b", "an", "an", "a"]，对新的序列继续统计相邻pair：("b", "an"),("an", "an"),("an", "a")，直到达到设定的 vocab size 或 merge 次数，最终我们会得到一个subword vocabulary。<u>但现实世界中不止有26个英文单词的unicode，还会有中文、日文、阿拉伯语、数字、emoji等各种character符号，这样的话基础词表会很大，所以我们舍弃Character-level BPE而采用Byte-level BPE。</u>
3. Unicode-level BPE（码点级BPE）：和Character- level BPE本质是是一个东西，Unicode-level BPE 以 Unicode code point 作为最小符号，在码点序列上运行 BPE 算法。

上面出现了[Byte-Pair Encoding(BPE)](# Byte-Pair Encoding (BPE))名词，我们需要先知道这个是什么算法！

但这里我们重点关注[Byte-Level BPE Tokenizer ](#Byte-Level BPE Tokenizer)算法！

当代Transformer架构中我们采用Byte-Level BPE Tokenizer，得到token-to-id mapping，最终输入到transformer第一层的embedding层的时候，其实是输入的文本从左到右对UTF-8 bytes扫描匹配出来的id序列，最终transformer预测的也是id序列，这就是为什么transformer最后我们还需要decoder解码器，并且编码和解码都是从左到右进行的。

但是我们有没有想到一件事情就是如果针对任意输入文本的UTF-8的bytes序列进行BPE操作，我们难免可能会分出跨单词的token，比如some bread这个单词出现的多了，我们可能会把"me_bre"这种无意义的内容收录到我们的vocabulary里面，这里我们就需要用[Pre-Tokenization](#Pre-Tokenization)，在了解完Pre-tokenization算法之后我们发现他可以节省BPE tokenization的时间，并且避免BPE在bytes序列上分出无意义的token。

其实基础词表里面有256个bytes，**还会有一些[Special token](#Special Token)！**，最经典的Special token就是**<|endoftext|>**

## Special Token

我们已经知道，**Byte-level BPE tokenizer** 的基础词表由 **256 个字节级别（byte-level）的 token** 构成，这保证了任意 Unicode 文本都可以被表示和编码。

在真实的 LLM 系统中，除了这 256 个基础 byte token 之外，**词表中还会额外包含一些 Special Token**，其中最经典、最重要的一个就是 **`<|endoftext|>`**。

**`<|endoftext|>` 是一种控制型（control）token，用于标记文本或文档的边界。**
 在实际训练过程中，我们通常不会将文本一条一条地单独喂给 LLM，而是将大量彼此不相关的文本拼接成一条连续的 token 流进行训练。为了防止不同文本之间产生不合理的上下文依赖，需要在两个文本块之间插入 **`<|endoftext|>`** 作为显式边界。

在 tokenizer 中，**`<|endoftext|>` 在词表中对应一个独立、不可拆分的 token ID**，不会参与 BPE 的拆分或合并过程。这样，模型在训练时会学习到如下两类统计关系：

- 文本内部的语言规律（如词汇、语法、搭配等）
- 文本结构层面的规律，例如
   **`text₁ → <|endoftext|>`** 和 **`<|endoftext|> → text₂`**

因此，训练完成后的模型在看到一段文本 `text₁` 时，确实可能预测出 `<|endoftext|>`，而在 `<|endoftext|>` 之后，也可能预测出新文本的起始 token。

但在**实际生成（inference）阶段**，我们**人为规定 `<|endoftext|>` 作为终止或分段信号**：一旦模型生成该 token，生成过程就会停止，或者显式开始一个新的文本。这并不是修改模型内部的概率分布，而是通过生成策略（stopping rule）阻止跨文本的语义延续。

通过这种方式，LLM 可以在**参数共享的前提下**，有效避免不同文本之间的上下文干扰，从而稳定地学习大规模、互不相关文本中的语言规律。

实际如下图：

![Untitled](/Users/wmyh0416/Pictures/Untitled.png)

## Byte-Pair Encoding(BPE)

BPE是一种贪心的压缩/合并算法。核心是：反复把序列中出现频率最高的相邻符号对(Pair)合并成一个新符号。这里的符号可以是：Byte、Character、Unicode code point甚至word。因此BPE本身是一种与符号类型无关的**通用合并算法**。

在现代大规模语言模型中，通常采用 **byte-level BPE**。
具体而言，模型首先将 Unicode 文本（字符串）通过 UTF-8 编码转换为字节序列（bytes），然后直接在整个字节序列上应用 BPE 算法进行 token 的学习与合并。
这种设计使得 tokenizer 以 byte 作为最小原子，从而避免 OOV 问题并具备良好的语言无关性。

## Byte-Level BPE Tokenizer

Unicode code point（Unicode码点）是Unicode标准为每一个“字符（又称Unicode Character）”分配的一个唯一整数ID，通常写成**U+XXXX**（XXXX代表Unicode code point是整数编号，一般以十六进制表示，也可以用十进制表示，长度没有限制！U+XXXX只是常见写法不是固定长度！）。无论是 UTF-8、UTF-16 还是 UTF-32，同一个字符对应的 Unicode code point 永远是相同的，世界上所有已收录的字符（Unicode Character）大约有150K。

> 今天就要揭开一个困扰我多年的ASCII码疑问了，ASCII码其实就是Unicode code point的子集。
>
> ASCII码是0-127的Unicode code point，只定义了128个字符，包含了英文字母A-Z, a-z, 数字0-9, 常见标点符号, 一些控制符（如NULL、换行等等）

Unicode string是由多个unicode code points组成的序列。

```text
Unicode string = [code point, code point, code point, ...]
```

Unicode encoding是一种规则：规定如何把一个Unicode code point转换成一串bytes，常见的Unicode encoding包含[UTF-8](#UTF-8)、UTF-16、UTF-32。

> Unicode encoding 规则带来了诸如节省存储空间、提高兼容性等诸多好处，这些细节在这里不再展开。我们只需要知道，Unicode encoding 是一种将字符对应的 Unicode code point 转换为字节序列（bytes）的协议，而这些 bytes 才是真正存储在计算机中的数据形式。

- Byte-level tokenization 是一种以字节（byte）作为最小原子单位的分词方法。

- 在该方法中，输入的 Unicode 文本首先通过 UTF-8 编码转换为字节序列（每个字节为 8 bits，取值 0–255）。

- 由于一个 Unicode 字符在 UTF-8 中可能由 1–4 个字节表示，tokenizer 实际操作的是字节序列而非字符。

- 在现代大规模语言模型中，通常在该字节序列上应用 BPE（Byte Pair Encoding）算法，以学习更长的 subword token，从而提高建模效率并避免 OOV 问题。

也就是说，我们可以把输入的Unicode文本通过UTF-8编码转化为字节（byte）序列。每一个byte由8bits构成，其取值范围为0-255（或者十六进制的00-FF）。**由于UTF-8的完备性，任何输入文本都可以表示成由这256中byte组成的序列。**在byte-level tokenization中，我们以byte作为最小原子单位，并在字节序列上应用BPE（Byte-Pair Encoding）算法，通过合并高频相邻byte序列来学习更长的token，最终得到token-ti-id的映射关系，即词表。其本质也相当于Subword tokenization，因为在BPE过程中我们肯定会学习到subword。

## UTF-8

一种最常用的把Unicode code points转化成Bytes的协议。UTF-8里的8表示$1\,byte = 8\,bits$。即$1\,byte$可以表示0-255个Unicode code points，$2\,bytes$就可以表示 $256^2$个Unicode code points,最多$4\,bytes$即可以表示$256^4=2^{32}$（40多亿）个Unicode code points，这样就把所有的Unicode code points全覆盖了！

## Pre-Tokenization

在前面我们已经知道了Byte-level BPE tokenization是针对任意文本（corpus），我们都可以得到其对应的整一串UTF-8编码的bytes序列，但是对这个东西直接做Byte-Pair Encoding(BPE)的话可能会导致vocabulary学到没有意义的subword或者学到具有high semantic similarity的 b'dog!' 和 b'dog.', even though they just differ in punctuation and will get different token IDs. 这是棘手的问题，我们需要对每一个单词（Word）之间设立边界（Boundary）所以我们需要用到Pre-tokenization去解决。

Pre-tokenization = 用正则表达式（Regex=Regular expression）把文本切成“类词单元”，作为BPE的训练边界与计数单位。

> 具体做法是：1. 用一个regex pattern从左到右扫描文本；2. 每次match得到一个pre-token（字符串片段，可能包含前导空格，又名leading whitespace）； 3. 用哈希表统计：pretoken->count； 4. 训练BPE时，对每一个pre-token：bytes = pretoken.encode("utf-8") ---> 在bytes内部统计相邻byte pair ---> 在pair频次加上count
>
> **注意，BPE merge发生在pretoken的byte序列上，边界由pre-tokenization决定**

下面是对应python处理示例：

```python
import re
from collections import Counter

# 这是一个“示意”的 pre-tokenization regex（接近 GPT-2 思路：保留前导空格）
# 实际作业可能给你固定 pattern，你要用作业提供的那个。
pattern = r"\s+|[A-Za-z]+|\d+|[^\w\s]"

def pretokenize_and_count(text: str) -> Counter:
    """
    用 re.finditer 按顺序扫描 text，每个 match.group() 是一个 pre-token。
    返回 pre-token 的频次统计。
    """
    counts = Counter()
    for m in re.finditer(pattern, text):
        tok = m.group()
        # 关键：不要 tok.strip()，否则会丢掉前导空格信息
        counts[tok] += 1
    return counts

text = "some bread is bad.\nsome bread is good!"
counts = pretokenize_and_count(text)

for k, v in counts.most_common(20):
    print(repr(k), v)

```

## Unicode是什么？

Unicode code point（Unicode码点）是Unicode标准为每一个“字符（又称Unicode Character）”分配的一个唯一整数ID，通常写成**U+XXXX**（XXXX代表Unicode code point是整数编号，一般以十六进制表示，也可以用十进制表示，长度没有限制！U+XXXX只是常见写法不是固定长度！）。无论是 UTF-8、UTF-16 还是 UTF-32，同一个字符对应的 Unicode code point 永远是相同的，世界上所有已收录的字符（Unicode Character）大约有150K。

> 今天就要揭开一个困扰我多年的ASCII码疑问了，ASCII码其实就是Unicode code point的子集。
>
> **ASCII码**是0-127的Unicode code point，只定义了128个字符，包含了英文字母A-Z, a-z, 数字0-9, 常见标点符号, 一些控制符（如NULL、换行等等）。
>
> **ASCII码**在UTF-8标准下，可以直接用一个字节，即两个十六进制位表示，我们训练的英文文本大多数都是**ASCII-heavy**的，所以用UTF-8去做tokenizer的话效率更高！！！

Unicode string是由多个unicode code points组成的序列。

```text
str = Unicode string #其实这里的str就是由字符，也就是Unicode(Unicode code point)组成的。
Unicode string = [code point, code point, code point, ...]
```

Unicode encoding是一种规则：规定如何把一个Unicode code point转换成一串bytes，常见的Unicode encoding包含[UTF-8](#UTF-8)、UTF-16、UTF-32。不同Unicode encoding方案有不同的Code unit即用于储存和处理文本的最小固定宽度单位，如UTF-8的Code unit是8bits并且一个其一个字符用1-4个Code unit表示。UTF-16的Code unit有16bits。

> Unicode encoding 规则带来了诸如节省存储空间、提高兼容性等诸多好处，这些细节在这里不再展开。我们只需要知道，Unicode encoding 是一种将字符对应的 Unicode code point 转换为字节序列（bytes）的协议，而这些 bytes 才是真正存储在计算机中的数据形式。

一些更好辅助我们理解Unicode的代码，如下：

```python
#chr()是把括号中的unicode code point转化成其对应的字符，即函数参数必须是integer
chr(65)=='A'
chr(29275)=='牛'

#ord()是把括号中的str字符变量转化成其对应的unicode code point，即函数参数必须是一个字符，也就是长度为1的str
ord('A')==65
ord('牛')==29275

#repr()打印出一串字符串的string representation（string representation != printed representation）
#string's printed representatin是调用print()函数去打印出一串字符串的结果，但是如果这类打印中有换行、空格等等Escape Sequence(转义序列)的话，打印的结果中就会出现换行、空格，让我们不能够清楚的知道这串我们打印出来的str里面究竟是什么，比如“先换行后空格”与“先空格后换行”打印出来的视觉效果完全相同，让我们无法分辨数据的真实结构。
#string's string representation是调用repr()函数去把字符串转化成string representation形式，1.如果里面涉及到不可见Escape Sequence如Control sequence的话，我们会直接打印出他的Escape Literal(转义字面量)/Slashed Representation(斜杠表示法)；2.如果里面涉及到文本，用print()打印出来我们能直接看到的字符的话，会按照原形式输出；3.如果里面涉及到如NULL等低位的一字节(0-255 bits)的不可直观显示字符的话，会打印出来x\NN(NN即其Unicode code point对应的十六进制表示)，如果涉及到高位的多字节(>255 bits)的不可直观显示字符的话，会打印出来u\NNNN或者U\NNNNNNN(其中NNNN或者NNNNNN代表该字符的Unicode code point对应的十六进制表示)
print('chr(0)') #打印出NULL，但是视觉效果是没有任何东西。
print(repr('chr(0)')) #对于print打印出的东西如果视觉效果没有东西，就会打印出其escape sequence
print('123\n') #会打印出123和换行操作，换行操作是视觉效果没有任何东西
print(repr('123\n'))  #会打印出第一行“123\n“，打印出了control character的escape sequence
print(repr('12\'3')) #会打印出"12'3"，因为'是一个视觉效果可见的字符，不会打印出其escaoe sequence
```

# Embedding

Embedding层本质上是一个 $vocab\_size * d\_model$ 的矩阵。

`embedding[i]`表示 $token\_id = i$ 的token对应的向量。

```python
embedding[0]  → token_id = 0 的向量
embedding[1]  → token_id = 1 的向量
embedding[2]  → token_id = 2 的向量
		·							·						·
		·							·						·
		·							·						·
embedding[vocab_size]  → token_id = vocab_size 的向量
```

利用我们训练好的tokenizer对输入句子encode后得到token id序列，输入到embedding层映射之后会获得一个形状为 $(seq\_len,d\_model)$ 的向量矩阵。这是我们之后要放入到attention机制中的东西。

$d\_model$ 的含义：每一个token被表示称一个多少维度的向量。$d$ 即是dimension。

还有一些需要知道的定义如下：

1. $batch\_size$ ：一次同时送进模型的句子数量。
2. $seq\_len$ ：每句话token的数量。
3. $d\_model$ ：每个token向量的维度。

> `(batch_size, seq_len, d_model) = (8, 128, 768)`即意味着：8句话，每句话128个token，token的维度是768。

<div style="border:2px solid #4CAF50; padding:10px; border-radius:8px;">
<strong> 每一句话的token数量肯定不一样，那怎么保证都是128个token呢？</strong><br>
  padding和truncation方法：<br>
  <br>
句子A → 10 tokens<br>
句子B → 25 tokens<br>
句子C → 7 tokens<br>
这样我们会在句子后面补PAD token<br>
句子A → [t1 ... t10, PAD, PAD, ..., PAD] (补到128)<br>
&lt;PAD&gt;实际上是一个special token，&lt;PAD&gt;也有自己的token id<br>
<br>
但当一句话的token数大于128时，我们会有三种截断（truncation）方式：<br>
  1.保留前面（最常见），即 tokens[:128]<br>
  2.保留后面，即 tokens[-128:]<br>
  3.双边截断（用于某些特殊任务，不常见）<br>
  因为GPT不是按照句子训练的。它是把整本预料拼成一个超长token流，所以模型不关心句子的边界。

在Decoder中实际上Embedding矩阵$E$ 一共使用了两次：

1. 一次是token序列进入系统流通前，我们需要利用Embedding矩阵把id序列转化为$n\times d_{model}$ 维度的数据
2. 第二次是数据从系统出来时，我们需要取最后一个token位置的$1\times d_{model}$向量，使其映射到$1\times vacabsize$ 的打分矩阵，再经过softmax层计算当前token对下一个token的预测概率，这里使用的是$E^T_{(d_{model}\times vocabsize)}$

Embedding矩阵在通过backpropagation操作就可以更新其参数了。

# PE（Positional Encoding）

PE（Positional Encoding）的目的是给模型“顺序信息”，因为embedding出来的数据是没有位置信息的，然后这些数据进入到的attention层本身对顺序不敏感。

PE有三种主流方法：

| 方法      | 是否可训练         | 是否支持外推 |
| --------- | :----------------- | ------------ |
| Sin/Cos   | 否                 | 是           |
| Learnable | 是                 | 否           |
| RoPE      | 否（旋转规则固定） | 是           |

## Self-Attention的数学形式：

给定输入：$X\in\mathbb{R}^{n\times d}$ ，$n$ 为序列长度，即 $n$ 个token；$d$ 为$d\_model$。
$$
\mathbf{X}
=
\begin{bmatrix}
\mathbf{x}_1 \\
\mathbf{x}_2 \\
\vdots \\
\mathbf{x}_n
\end{bmatrix}
=
\begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1d} \\
x_{21} & x_{22} & \cdots & x_{2d} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \cdots & x_{nd}
\end{bmatrix}
\quad
\text{(each row corresponds to one token)}
$$
计算：$Q=XW_Q$ ，$K=XW_K$ ， $V=XW_V$ ，其中：$W_Q,W_K,W_V\in \mathbb{R}^{d \times d}$，则 $Q,K,V\in \mathbb{R}^{n\times d}$。

然后attention输出：$Attention(Q,K,V)=softmax(\frac {QK^T}{\sqrt{d_k}})V$

## 为什么attention对顺序不敏感？

设排列矩阵$P\in \mathbb{R}^{n\times n}$ ，目的是用来打乱 $X$ 的行顺序，则 $X'=PX$ （左乘行变换）。

计算打乱后的attention：

①计算 $Q',K',V'$ ：

$Q'=X'W_Q=(PX)W_Q=P(XW_Q)=PQ$

$K'=X'W_K=(PX)W_K=P(XW_K)=PK$

$V'=X'W_V=(PX)W_V=P(XW_V)=PV$

②计算 $Q'K'^T$ ：

$Q'K'^T=PQ(PK)^T=PQK^TP^T$

③ $softmax$ 内参数：

原attention分数：$S=\frac{QK^T}{\sqrt{d_k}}$

排列矩阵 $P$ 打乱后的attention分数：$S'=\frac{Q'K'^T}{\sqrt{d_k}}=\frac{PQK^TP^T}{\sqrt{d_k}}=PSP^T$

则我们发现：$S=PSP^T$

④ $softmax$ 做运算：

$softmax(\frac{Q'K'^T}{\sqrt{d_k}})=softmax(PSP^T)$ 

因为 $softmax$ 是只对行做运算，则 $softmax(PSP^T)=P\cdot softmax(S)\cdot P^T$

⑤最终输出：

$Attention(X')=softmax(S')V'=softmax(\frac{Q'K'^T}{\sqrt{d_k}})V'=softmax(PSP^T)PV=P\cdot softmax(S)\cdot V$

$Attention(X)=softmax(S)\cdot V$

则 $Attention(X')=P\cdot Attention(X)$

即最终我们知道 $Attention(PX)=P\cdot Attention(X)$ ，这意味着无论我们以什么方式打乱输入的行顺序，最终的输出也是相同的打乱方式。即，从embedding层输出的id序列句矩阵$X$是`[X_1,X_2,...,X_seqlen]`和其他排列方式是一样的，没有区别。因此self-Attention对顺序不敏感。这也是我们需要进行PE的主要原因！

**所以我们需要利用PE使得**$Attention(P(X+PE))\neq P\cdot Attention(X+PE)$，**把原始输入的**$X$**矩阵转化为**$X+PE$**矩阵，这样我们就可以使得self-attention对顺序敏感。**

## PE（Positional Encoding）矩阵的三种方法

### 1.Sinusoidal PE

这一种PE矩阵是一个固定值矩阵，一旦 $d\_model$ 和 $seq\_len$ 确定了，PE矩阵就固定了。
$$
偶数维度PE_{(pos,2i)}=sin(\frac{pos}{10000^{\frac{2i}{d}}})
$$

$$
奇数维度PE(pos,2i+1)=cos(\frac{pos}{10000^{\frac{2i}{d}}})
$$

$pos$ 表示token在序列中的位置顺序

$i$ 表示维度索引

$d$ 表示$d\_model$

$PE$ 是一个矩阵，$PE$ 矩阵内部的元素位置$行i=pos,列j=2i或2i+1$ 。

如第0个token，维度1对应的PE矩阵元素为：$pos=0$，$2i+1=1\implies i=0$ ，则$PE(0,1)=cos(\frac{0}{10000^{\frac{0}{d}}})$

![image-20260226212923141](/Users/wmyh0416/Library/Application Support/typora-user-images/image-20260226212923141.png)

根据PE公式算完PE矩阵之后我们得到attention最终的输入矩阵为：$X'=X+PE$

这样就可以输入到attention开始我们的transformer运算了！！！



### 2.RoPE（Rotary PE）

RoPE（Rotary PE）方法不是像Sin/Cos方法那样对从Embedding层出来的$X$矩阵直接进行操作，而是对$X$的计算产生的$Q$和$K$矩阵进行操作。

 在想知道RoPE是怎么作用的之前，我们需要知道什么是[旋转矩阵](#旋转矩阵(rotation matrix))。

先假设我们有一个模型，其向量维度 $d\_model=2$，则每一个token的向量都是2维的，给定一个embedding矩阵`[token_1, token_2, ..., token_m, ..., token_n, ..., token_seq_len]`，那么`token_m`对`token_n`的未加入RoPE的raw attention score如下**（数学界默认一维向量为列向量，下面利用列向量计算）**：
$$
raw\,score=Q[m]^T\cdot K[n]
$$
加入RoPE后（即$Q[m]$逆时针旋转$m\theta$，$K[n]$逆时针旋转$n\theta$）：
$$
\begin{aligned}
raw\,score
&=(R(m\theta)Q[m])^T\cdot R(n\theta)K[n]\\
&=Q[m]^TR(m\theta)^T\cdot R(n\theta)K[n]\\
&=Q[m]^TR(-m\theta)\cdot R(n\theta)K[n]\\
&=Q[m]^T\cdot R[(n-m)\theta]\cdot K[n]
\end{aligned}
$$
我们可以看到打分不再只是单纯向量的内积，而是在相对位置（relative position）$(n-m)$的控制下的内积，也就是说缩放注意力得分矩阵（Scaled Attention Score Matrix）的元素$S_{mn}$。**因此模型不仅感知“这个词是什么”，也感知“这个词相对于当前位置有多远”。并且每两个词相对距离是固定的，这样会使得模型学习到词和词之前随着位置变化产生的意义。**

只要`token_m`和`token_n`的相对距离不变，那`token_m`到`token_n`的旋转角度永远是$(n-m)\theta$（注意这里的$\theta$还是变量）。

实际模型操作中，token的维度肯定不止2维，我们是把每两维打包，然后旋转。

#### 实际RoPE操作

​	所以RoPE就是对$Q$ 矩阵和$K$ 矩阵的第$m$ 个token的向量每两个维度为一组旋转$m\theta_i$ 角度，其中$\theta_i=base^{\frac{-2i}{d_k}}$（这里$base$ 一般为10000）。$i$ 表示第$i$ 组，从第0组开始，第0组对应第一、二维度（第0、1列），则$i_{max}=\frac{d_k}{2}-1$。

**写成公式：**$Q[m]=(q0,q1,q2,q3,…)$

**分组：**$(q_{2i},q_{2i+1})$

**旋转：**$\begin{pmatrix}q'_{2i}\\q'_{2i+1}\end{pmatrix}=\begin{pmatrix}\cos(m\theta_i)&-\sin(m\theta_i)\\\sin(m\theta_i)&\cos(m\theta_i)\end{pmatrix}\cdot\begin{pmatrix}q_{2i}\\q_{2i+1}\end{pmatrix}$

$K$ 矩阵同理！

​	RoPE 是对 Q 和 K 矩阵的第 m 个 token 的向量，每两个维度为一组，按照不同频率$\theta_i$ 旋转 $m\theta_i$ 角度。这样在计算 attention score 时，通过旋转矩阵的性质，使得最终打分只依赖于相对位置$(n-m)$，从而实现了相对位置编码。

#### 旋转矩阵(rotation matrix)：

$$
R(\theta)=
\begin{pmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{pmatrix}
$$

假设我们有一个向量$x=\begin{pmatrix}x_1\\x_2\end{pmatrix}$，把它逆时针旋转$\theta$ 就是$x'=R(\theta)\cdot x=\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}\cdot\begin{pmatrix}x_1\\x_2\end{pmatrix}=\begin{pmatrix}x_1\cos\theta-x_2\sin\theta\\x_1\sin\theta+x_2\cos\theta\end{pmatrix}$

**其重要的性质如下：**

1. **保持长度：**$\|R(\theta)\cdot x\|=\|x\|$，即$x$向量逆时针旋转$\theta$角度长度不变。

![image-20260226234949132](/Users/wmyh0416/Library/Application Support/typora-user-images/image-20260226234949132.png)

2. **正交矩阵：**$R(\theta)^T=R(-\theta)$，即旋转矩阵的逆矩阵意味着顺时针旋转$\theta$度。

![image-20260226235302533](/Users/wmyh0416/Library/Application Support/typora-user-images/image-20260226235302533.png)

3. **旋转可叠加：**$R(a)\cdot R(b)=R(a+b)$，这在RoPE里面非常重要。

<img src="/Users/wmyh0416/Library/Application Support/typora-user-images/image-20260226235832581.png" alt="image-20260226235832581" style="zoom:50%;" />

# Add & Norm（Add & Normalization）

<img src="/Users/wmyh0416/Library/Application Support/typora-user-images/image-20260301031031489.png" alt="image-20260301031031489" style="zoom:33%;" />

Add & Norm主要核心是$LayerNrom(x+F(x))$。

1. 在Attention子层：$F(x)=Attention(x)$。（这里的$x$ 来自于进入attention layer之前的矩阵$X_{n\times d_{model}}$）
2. 在FFN子层：$F(x)=FFN(x)$。（这里的$x$ 来自于第一次Add & Norm之后的矩阵$X_{n\times d_{model}}$）
3. $LayerNorm$ 就是对矩阵做一种固定运算，加入可学习参数

## Attention子层

从 Embedding 层输出出来的矩阵经过PE之后的到$X\in\mathbb{R}^{n\times d_{model}}$ ，进入Multi-Head Attention之后的到attention层的输出$O=Concat(O^1,O^2,...,O^H)\cdot W_O=...=Attnetion(X)\in\mathbb{R}^{n\times d_{model}}$。

则Add是：$x+F(x)=X+Attention(X)=X+O$，这里是两个矩阵的相加，维度为$\mathbb{R}^{n\times d_{model}}$。

## FFN子层

FFN（Feed Forward Network）是**对每个token独立进行非线性特征变换**。

核心是一个**两层全连接网络**：
$$
FFN(x)=\sigma(xW_1)W_2
$$
其中：$x\in\mathbb{R}^{n\times d_{model}}$

$W_1\in\mathbb{R}^{d_{model}\times d_{ff}}$

$W_2\in\mathbb{R}^{d_{ff}\times d_{model}}$

$d_{ff}\approx 4d_{model}$

$\sigma$ 是GELU或者ReLU

步骤如下：

1. 扩维：$d_{model}\to d_{ff}$ ，把特征投影到更高维空间。
2. 非线性：$\sigma(\cdot)$ 是激活函数，如果没有激活函数的话，$xW_1W_2$ 多层叠加仍是线性，表达能力有限。
3. 压回到原维度：$ d_{ff}\to d_{model}$ ，恢复到原始维度，方便方差相加。

>$W_1,W_2$ 是模型的可学习参数，各自的参数量为：$d_{model}*d_{ff}\approx d_{model}*4d_{model}=4d_{model}^2$ ，则每一个FFN层产生$8d_{model}^2$ 个可学习参数。

则$FFN(x)=\sigma(xW_1)W_2\implies FFN(X_{(n\times d_{model})})=\sigma(X_{(n\times d_{model})}\cdot W_{1_{(d_{model}\times d_{ff})}})\cdot W_{2_{(d_{ff}\times d_{model}})}$最终得到一个矩阵的维度还是$\mathbb{R}^{n\times d_{model}}$。

## Norm\LayerNorm（Normalization）

Nrom这一步就是对一个矩阵，做固定方法的运算。如下：

给定矩阵：$Z\in\mathbb{R}^{n\times d_{model}}$，$n$ 为token数，$d_{model}$ 是每个token向量的维度。

取其中一个token向量：$z_i=[z_{i1},z_{i2},...,z_{id_{model}}]$

1. 计算均值：$u_i=\frac{1}{d_{model}}\sum_{j=1}^{d_{model}}z_{ij}$
2. 算方差：$\sigma_i^2=\frac{1}{d_{model}}\sum_{j=1}^{d_{model}}(z_{ij}-u_i)^2$
3. 标准化：$\tilde{z}_{ij}=\frac{z_{ij}-u_i}{\sqrt{\sigma_i^2+\epsilon}}\quad (for\ j\ from\ 1\ to \ d_{model})$，这里的$\epsilon$ 是一个很小的正数常量，通常取$10^{-5}$ 或$10^{-6}$，防止除以0和数值爆炸的情况，
4. 可学习缩放和平移：$\hat{z}_{ij}=\gamma_j\tilde{z}_{ij}+\beta_j\quad (for\ j\ from\ 1\ to\ d_{model})$，这里的$\gamma_j$ 和$\beta_j$ 都是可学习参数。

> 可学习参数：$\gamma=(\gamma_1,\gamma_2,...,\gamma_{d_{model}}),\qquad\beta=(\beta_1,\beta_2,...,\beta_{d_{model}})$

注意：一次Nrom操作里所有token共用$\gamma$ 和$\beta$ 参数，则一次Norm产生$2\times d_{model}$ 个可学习参数。

综上，一次Nrom操作就是对$Z$ 矩阵里的$n$ 个token向量全部进行如上4步，输出的矩阵维度还是$\mathbb{R}^{n\times d_{model}}$。

# Transformer参数调节

## Softmax

$Softmax$ 是将一串由logits（模型原始打分）矩阵$Z$ ，转化为由预测概率矩阵$P$ 。
$$
p_i=\frac{e^{z_i}}{\sum_je^{z_j}}
$$

## Cross Entropy（多分类）

真是标签是one-hot编码：$y=(y_1,y_2,...,y_{vocabsize})$

损失函数：$L=-\sum_iy_i\log p_i=-(y_1\log p_1+...+y_p\log p_k+...y_{vocabsize}\log{p_{vocabsize}})=-\log{p_k}$

这里我们计算的loss只看了正确词的loss，因为softmax 是一个“竞争机制”，概率总和必须等于 1。提高正确类别概率，就等于压低其他类别的概率。

> Sigmoid函数是将实数压成概率，这些概率互不影响，则计算Loss时，需要考虑到每一个数据的loss，即考虑Loss总和。
>
> 但Softmax是将一组实数**互相制约地**压成概率，计算Loss时，只需要考虑一个目标数据的loss，因为降低这个loss，其他的loss也会相应改变。即提高一个token的预测概率，其他token的预测概率自然减少

#### Sigmoid + BCE（Binary / Multi-label）

每个类别 **独立判断**

输出：
$$
p_i = \sigma(z_i)
$$
Loss：
$$
L = \sum_i \left[-y_i \log p_i - (1-y_i)\log(1-p_i)\right]
$$
类别之间 **没有竞争关系**

必须对所有类别计算 loss

适用于：二分类，多标签问题（可同时属于多个类别）
#### Softmax + Cross Entropy（Multi-class）

类别之间 **互斥**

输出：
$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$
Loss：
$$
L = -\sum_i y_i \log p_i
$$
若 one-hot：
$$
L = -\log p_k
$$
虽然只写正确类别的 loss，但由于 softmax 归一化，所有类别都会参与竞争。

适用于：多分类问题，语言模型（下一个 token 只能选一个）

## 模型最终输出

模型从最后一层Transformer Encoder Block出来的时候，仍然是一个$n_t\times d_{model}$ 维度的矩阵（$n_t$ 代表目标语言token序列长度，每一次预测之后该长度会加1）。

我们需要取最后一个token的向量，因为我们要用这个词中的信息去计算推导下一个token是什么？

但是这个维度为$1\times d_{model}$ 的向量是怎么得到$1\times vocabsize$ 大小的对每一词的概率预测呢？如下：

1. 我们会让这个token向量和Embedding矩阵的逆$E^T$ 相乘，则维度变化为$(1\times d_{model})(d_{model}\times vocabsize)=(1\times vocabsize)$ 就得到了一个原始打分。
2. 我们需要对这个原始打分向量，经过softmax层就会得到最终的概率预测P矩阵！

## Transformer 参数更新机制总结

### 1. 整体目标

训练的本质是：

$$
\min_{\theta} L(\theta)
$$

其中：

- $\theta$ = 所有模型参数（Attention 矩阵、FFN、Embedding、LayerNorm 等）
- $L$ = Cross Entropy Loss

### 2. 前向传播（Forward）

#### Step 1：Embedding

$$
x_t \rightarrow E[x_t] \in \mathbb{R}^{d_{model}}
$$

#### Step 2：Transformer 计算

经过多层：

- Multi-Head Attention
- Add & Norm
- Feed Forward

得到：

$$
h_t \in \mathbb{R}^{d_{model}}
$$

#### Step 3：投影到词表空间

$$
z = h_t W_{out}
$$

其中：

$$
W_{out} \in \mathbb{R}^{d_{model} \times |V|}
$$

得到：

$$
z \in \mathbb{R}^{|V|}
$$

#### Step 4：Softmax

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

得到概率分布：

$$
p \in \mathbb{R}^{|V|}
$$

#### Step 5：Cross Entropy Loss

真实标签是 one-hot：

$$
L = -\sum_i y_i \log p_i
$$

若正确类别是 $k$：

$$
L = -\log p_k
$$

### 3. 反向传播（Backpropagation）

#### Step 1：对 logits 求导

关键公式：

$$
\frac{\partial L}{\partial z_i} = p_i - y_i
$$

解释：

- 正确类别：$p_k - 1$
- 错误类别：$p_i$

#### Step 2：链式法则传播梯度

梯度向后传播：

$$
\frac{\partial L}{\partial W_{out}}
\rightarrow
\frac{\partial L}{\partial h_t}
\rightarrow
\frac{\partial L}{\partial Attention}
\rightarrow
\frac{\partial L}{\partial Embedding}
$$

所有参数都会获得：

$$
\frac{\partial L}{\partial \theta}
$$

### 4. 参数更新

使用梯度下降：

$$
\theta \leftarrow \theta - \eta \nabla L(\theta)
$$

其中：

- $\eta$ = 学习率
- $\nabla L(\theta)$ = 当前点的梯度

### 5. Loss 在训练中的作用

- Loss 数值用于监控训练
- 梯度决定参数如何更新
- 更新依赖梯度，而不是 Loss 数值本身

### 6. Embedding 的训练机制

若使用 weight tying：

$$
W_{out} = E^T
$$

则：

- 输入时使用 $E$
- 输出时使用 $E^T$

Embedding 在：

- 输入阶段参与计算
- 输出阶段参与内积
- 反向传播中被更新

### 7. 核心理解

Transformer 训练本质是：

- 在高维参数空间中
- 通过梯度下降
- 最小化 Cross Entropy
- 不断调整语义空间结构

### 8. 一句话总结

Forward 负责算预测  
Loss 负责衡量错误  
Backward 负责算梯度  
Optimizer 负责更新参数



# Decoding

Transformer在每一步都会输出：
$$
P(y_i|y_1,y_2,...,y_{i-1})
$$
意思是：在给定已经生成的token下，预测下一个token的概率分布，$P$ 的维度为$vocabsize$。

**所以在已经算得P之后，我们要选择哪一个token，这个选择过程就是Decoding。**

Decoding的理论上最好的生成方式是找到概率最大的整个句子，即$y^*=\arg\max_y P(y)$。但是这种方法不切实际。所以有了如下很多方法。

**但整体流程是：logits -> temperatrue -> softmax -> top-p \ top-k -> sampling。**

## Greedy Decoding

最简单的方法，每一步选择概率最大的token，即$y_i=\arg\max P(y_i|context)$。

优点：快、简单。

缺点：生成文本很死板。



## Beam Search

**Beam Search** was  introduced to reduce search errors in sequence generation by keeping multiple candidate sequences instead of committing to a single greedy choice.

Beam Search





## Sampling（抽样）

Sampling即按照概率抽样，则对P中的token按照其概率进行抽样，如`dog 0.4, cat 0.3, trash 0.001,...`就是有`40% dog, 30% cat, 0.01% trash,...`。

优点：更自然，更具有创造性。

缺点：可能抽到垃圾token，如`0.01%`的`trash`token。

则我们衍生出了[Top-k Sampling](#Top-k Sampling)和[Top-p Sampling（Nucleus Sampling）](#Top-p Sampling（Nucleus Sampling）)两种方法。

### Top-k Sampling

Top-k的思想：只保留概率最大的k个token。这样很多垃圾token就被直接丢掉了。

步骤：1. 排序vocabsize个概率；2. 保留前k个概率的token；3. 重新归一化；4.再进行sampling。

### Top-p Sampling（Nucleus Sampling）

top-p的思想：不是固定token的数量，而是保留累计概率$\geq$ p的token 。

> 为什么Top-p比Top-k好？
>
> Top-p更adaptive。

## Temperature

Temperatrue是对模型输出的logits $z_i$ 进行操作。

正常的softmax为：$P_i=softmax(z_i)$；

加入temperatrue之后的softmax为：$P_i=softmax(\frac{z_i}{T})$。其中 $T$ 为temperatrue。

其作业是改变概率分布，使概率分布更确定或者更随机：

1. $T<1$ ，更确定，效果是最大概率更大，其他概率更小；
2. $T=1$ ，不变，等于没加入Temperatrue，还是原始分布；
3. $T>1$ ，更随机，概率更平均；
4. $T\to 0$ ，把最大概率拉到最大，相当于Greedy Decoding； 
5. $T\to \infin$ ，把概率变成接近均匀分布，这样会让所有token完全随机。













 # 待学习

1. compression ratio
2. 怎么计算针对一个文本的大概编码（Encode）时长是多少？
3. 针对 Vocab 大小为 1 万的词表来说，用什么去储存它的 ID 效率更高：是 int32 还是 int16？默认的是 int16 吗？

4. 针对不同参数的 GPT 模型，Transformer 模型的一些参数方面的计算
5. FLOPs？

# 调用工具

1. 文本框

<div style="border:2px solid #4CAF50; padding:10px; border-radius:8px;">
<strong>❓ 我的疑问</strong><br>
为什么这个算法是 O(n)？
</div>













# 2.27 Lec SFT/RLHF

## SFT（Supervised Fine-Tuning）

**Supervised Fine-Tuning (SFT)** is a training procedure in which a pretrained language model is further trained on high-quality, human-labeled instruction–response pairs using supervised learning.

More formally:

> Supervised Fine-Tuning trains a pretrained language model to generate desired outputs conditioned on given inputs by minimizing the cross-entropy loss between the model’s predictions and human-provided target responses.

In probabilistic terms, SFT optimizes:

$\max \sum_{t \in \text{response}} \log P_\theta(y_t \mid x, y_{<t})$

where:

- x = the input (e.g., instruction or question)
- y = the target response
- The model is trained to predict response tokens conditioned on the input and previous response tokens

In short:

> SFT converts a general pretrained language model into an instruction-following model by training it on supervised input–output examples.



**FLAN、Alpaca、OpenAssistant 都属于“早期大规模 SFT / Instruction Tuning 工作”。**



Instruction tuning 的“风格”，很大程度由数据长度分布决定。

<img src="/Users/wmyh0416/Library/Application Support/typora-user-images/image-20260227133216187.png" alt="image-20260227133216187" style="zoom:25%;" />





![image-20260227134200528](/Users/wmyh0416/Library/Application Support/typora-user-images/image-20260227134200528.png)

MMLU GSM BBH TydiQA Codex-Eval AlpacaEval都是公开的验证数据集。而左边的Vanilla LLaMa 13B是benchmark，我们做的是在benchmark model在某个SFT数据集训练之后再去在公开数据集上验证。





## RLHF（Reinforcement Learning from Human Feedback）

这一步的核心是Reward Model。

流程：

1. 给模型一个问题x
2. 模型生成多个回答（2-4个）
3. 人类标注员手工排序选出最好的再喂给LMs
4. 训练Reward Model使得$R(x,A)>R(x,B)$
5. 训练完成Reward Model后，人类退出
6. RL阶段模型自动生成回答$\implies$Reward Model自动打分$\implies$用PPO（Proximal Policy Optimization）更新模型

> OpenAI的一部分RLHF数据来源于真实用户的使用行为和反馈，如有时候在用gpt的时候，他会让我们选择我们喜欢哪一个回答。



## DPO（Direct Preference Optimization）

DPO是RLHF的更简单替代方案。更准确来说DPO是一种绕过了Reward Model和PPO的更简单的偏好优化方法。

DPO的核心是：提高「被人类偏好答案」的概率，降低「不被人类偏好答案」的概率。也就是使得$\log \pi_\theta(y_w|x) - \log \pi_\theta(y_l|x)$即好答案的log概率-坏答案的log概率差值变大。



## PPO>DPO为什么还需要用DPO？

因为：

> **PPO 更强，但更贵、更复杂、更不稳定。**

DPO 是一个：

> 性价比极高的近似方案。





# 3.6 Lec RLHF vs. RLVR

RLHF（Reinforce Learning from Human Feedback）是训练模型偏好回答的。而RLVR（Reinforce Learning with Verifiable Reward）是训练模型推理能力的（这些推理是可以verify的，也就是只有一个正确答案）

| Aspect               | RLHF (Reinforcement Learning from Human Feedback)          | RLVR (Reinforcement Learning with Verifiable Rewards)        |
| -------------------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
| Main Goal            | Align model outputs with human preferences                 | Improve reasoning ability using automatically verifiable rewards |
| Typical Tasks        | Chat, summarization, dialogue, instruction following       | Math, coding, reasoning, agentic tasks                       |
| Example Prompt       | "Tell me three ways to live a healthy lifestyle"           | "Solve 23 × 47" or "Make this repo compile and pass tests"   |
| Training Data Format | (x, y⁺, y⁻) → prompt + preferred answer + rejected answer  | (x, a) → prompt + correct answer                             |
| Reward Source        | Reward model trained on human preferences (LLM-as-a-judge) | Automatic verification (correct = 1, incorrect = 0)          |
| Reward Quality       | Subjective / noisy                                         | Objective / deterministic                                    |
| Human Annotation     | Required (expensive and slow)                              | Usually not required                                         |
| Algorithms           | PPO, DPO                                                   | PPO, GRPO                                                    |
| Main Use Case        | Conversational alignment                                   | Reasoning model training                                     |
| Example Models       | ChatGPT, Claude chat models                                | DeepSeek-R1, OpenAI o1-style reasoning models                |
| Key Advantage        | Aligns with human style and safety preferences             | Scales well for reasoning tasks with clear answers           |
| Key Limitation       | Reward model bias and high annotation cost                 | Only works for tasks with verifiable answers                 |



## Expert Iteration（专家迭代）

RLVR的一种简单方法就是Expert Iteration。

核心思想：模型尝试自己做题$\to$如果做对了$\to$就把这个推理当成训练数据

### 流程

1. 我们拥有数据`(x_i, a_i)`，即问题`x_i`和正确答案`a_i`。

2. 生成答案`y_pred = LLM(x_i)`

3. 验证如果正确`reward = 1`，如果错误`reward = 0`

4. 训练：如果正确训练`SFT on (x_i, y_pred)`

   















