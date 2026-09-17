---
layout: post
title: VLLM KV Cache 管理
---

本文整理总结了 VLLM 的 KV Cache 管理机制。这里的核心问题在于：对于推理框架，要怎么将一整块连续可用的显存按需分配给不同请求，并满足以下两个核心要求：

- 灵活性：分配方式不受预先指定的 bs, seqlen 等参数限制，在实际负载情况高度不确定的线上推理服务场景中，始终能把所有可用显存榨干，避免显存浪费。
- 通用性：同一个分配思路尽可能兼容不同模型结构，尽可能避免对不同 Attention 结构需要定制重写分配逻辑。

[graph 0]

KV Cache 分配本来是一个很简单的问题，不过随着模型 Attention 结构越来越复杂，对 KV Cache 管理机制也提出了越来越多的要求。本文由易到难，顺序分析 VLLM 现有的几种 KV Cache 管理模式及对应的思路。

另外，近来新模型结构层出不穷，推理框架发展日新月异，本文内容仅以当前 VLLM 现有结构为例，重在分析 KV Cache 结构发展历程及对应的设计思路。若涉及相关代码的分析/功能开发，最好的方式还是直接阅读 [vllm](https://github.com/vllm-project/vllm) & [vllm-ascend](https://github.com/vllm-project/vllm-ascend) 源码、实际部署调试。

# 0: BSND KV Cache - Demo

这是一个题外话，BSND KV Cache 常见于一些小 demo (例如用来展示模型基础结构的 demo 代码仓)，常见的推理引擎中通常不会使用这种格式。

该格式与训练时类似，根据用户手动指定的最大并发 `B(batch_size)` 及最大序列长度 `S(sequence_length)`，则 KV Cache 最多会储存 B * S 个 token。再加上每个 token 所需空间为 `N(head_num) * D(head_dim)`，最后 KV Cache 结构就是 [B, S, N, D].

假设我们指定 `B` = 4, `S` = 4096, 模型 `N` = 8, `D` = 512, 使用 BF16 格式，则此时我们申请的 KV Cache 占用 B * S * N * D * 2 = 128MB 空间，最多能储存 B * S = 16384 个 token。

[graph 0-0]

BSND 格式的 KV Cache 已经能够满足基础的推理需求。但它有一个显而易见的问题：灵活性不足。我们预先指定了最大并发数及最大序列长度，但实际的在线推理场景中负载分布是高度不确定的。当实际的并发数、序列长度与预先指定值相差较大时，就很容易出现显存利用率低的问题。

例如，当推理系统收到 1 条长度为 8192 的序列，虽然总的 KV Cache buffer 容量大于该序列长度，但因为每条请求只能使用自己固定的 `S` = 4096 个 token 空间，无法跨 slot 分配 KV Cache，所以导致该请求无法被调度。

[graph 0-1]

类似的，当推理系统收到 8 条长度为 1024 的序列。虽然总的 KV Cache buffer 容量大于所有这些序列长度之和，但由于一个 slot 只能用于一条请求的推理，我们最多也只能同时调度 4 条请求，另外 4 条请求需要排队等待，同时又有一半以上的 KV Cache 空间是闲置的。

[graph 0-2]

因此，推理引擎中通常不会使用 BSND KV Cache。为解决上述问题，就引出了以下的 PA (Page Attention) 格式 KV Cache。

# 1: UniformSpec KV Cache - 基础的 PA 格式 KV Cache

UniformSpec 是后来增加了其他更多种类的异构 KV Cache 之后为做区分才有的名字，这里先简单理解为基础的 PA 格式 KV Cache 即可。Uniform Spec 这个名字对应的是比较简单的同构 Attention 结构，即模型所有层都有相同的 Attention, KV Cache 结构 (KV Cache Spec)，例如 [Qwen3-235B](https://huggingface.co/Qwen/Qwen3-235B-A22B), [DeepSeek-V3.1](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) 等。

PA KV Cache 的思路类似于操作系统中的内存分页：我们不再预先假定请求并发数、序列长度，而是把可用显存按固定大小均匀分块，每一块可以储存 `block_size` 个 token。对应的 KV Cache 结构会变成 `[num_blocks, block_size, N, D]`，即把原来的 B * S 换成了 num_blocks * block_size。在实际推理服务中，调度器直接根据序列的实际长度按需分配 block，一条请求分配的 KV Cache 在 block 内连续，block 间则可以不连续，因此每条请求还需要一个 `block_table` 来记录该请求的每个逻辑块实际储存在 KV Cache tensor 中的哪一个物理块上 (类似操作系统中的页表)。在这种情况下，不论序列并发数/长度如何分布，我们总能把 KV Cache 空间基本用满，浪费最多存在于每条请求的最后一个 block 内。

[graph 1-0]

这种基础的 PA KV Cache 分配方式 —— UniformSpec 也是比较简单的。首先，推理引擎会在初始化 KV Cache 前先进行一次 profile run, 用 GPU 所有可用显存大小减去模型权重、激活值所占的部分，剩余的就是 KV Cache 可用的显存大小 `available_kv_cache_memory`。

[graph 1-1]

然后，总的 `available_kv_cache_memory` 空间会均匀分给模型的所有层，每层 KV Cache 可用空间 `available_kv_cache_memory_layer` = `available_kv_cache_memory` / `num_layers`。

[graph 1-2]

为计算每层可用显存可以分给多少个 block，推理引擎会先计算每个 block 所占空间大小: `page_size` = `block_size` * `head_num` * `head_dim`。这里注意区分这两个类似的概念：
- `block_size`: 每个 block 存多少个 token
- `page_size`: 每个 block 占用空间为多少字节

当我们已经知道了每层 KV Cache 可用空间大小，以及每个 block 所需空间大小之后，就可以计算出可以分配的 block 数量 `num_blocks` = `available_kv_cache_memory_layer` / `page_size`，所有层可用的的 `available_kv_cache_memory_layer` 都会被均匀分为 `num_blocks` 块。

[graph 1-3]

目前 vllm 的实现中，模型每一层会申请一个独立的 KV Cache tensor，最终的 KV Cache 结构如下：

```
kv_caches = {
    'layers.0.attn.self_attn': torch.Tensor, # [num_blocks, head_num, head_dim]
    'layers.1.attn.self_attn': torch.Tensor, # [num_blocks, head_num, head_dim]
    ...,
    'layers.num_layers-1.attn.self_attn': torch.Tensor
}
```

最后需要说明的是，虽然模型一共有 `num_layers` 个独立的 KV Cache tensor，但因为每层的 KV Cache 空间大小相同，在服务同一条请求时每层 KV Cache 需要分配的空间大小也相同，所以它们可以共享同一份调度逻辑: UniformSpec KV Cache 只需要 1 个 `kv_cache_manager`，该 `kv_cache_manger` 不感知模型总共有多少层，它只负责根据**单层内**的 KV Cache block 余量以及当前请求所需 block 数量计算该请求的调度方案，这 1 份调度结果 (`new_block_ids`, `block_table` 等) 会被复制到模型每一层以相同的方式使用。

[graph 1-4]

UniformSpec - 这一基础的 PA 格式 KV Cache 已经很好地解决了 KV Cache 的灵活分配问题。对不同规格的 GQA/MLA 模型，只需要在初始化时调整 KV Cache 的 `head_num`, `head_dim` 两个参数至模型实际规格即可。不过混合 Attention 模型的出现对 KV Cache 的组织方式提出了新的要求。

# 2: UniformType KV Cache - 多种不同规格 KV Cache 的组合

UniformType KV Cache 主要是为了混合 Attention 模型而设计的，例如 [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2), [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) 等。相比前面提到的同构 Attention 模型每层有且仅有一个 KV Cache、每层 KV Cache 大小都相同这样比较简单的结构，混合 Attention 模型在 KV Cache 上出现的变化有：
- 不同层间可能含有不同的 KV Cache 类型
- 同一层内也可能含有多个不同的 KV Cache 类型

这里我们以 [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) 为例，首先，该模型每层都有一个普通的 MLA KV Cache (以下缩写为 KV Cache)，形状为 `[seqlen, 1(kv_head_num), 576(kv_head_dim)]` (简单起见，这里就不再区分 k & v 或者说 nope & rope ，把 kv 当成一个 cache)。其次，每 4 层中会有 1 层还有一个 Lightning Indexer K Cache (以下缩写为 LI Cache)，形状为 `[seqlen, 1(indexer_head_num), 128(indexer_head_dim)]` (模型开始还有两层连续的 Indexer Source 层，非严格 1:3，不过简单起见先忽略，不影响此处核心内容)。所以该模型整体 KV Cache 结构为：
- Layer 0, 4, 8, 12, ... : Indexer Source 层，每层含 2 个 Cache: KV Cache `[seqlen, 1, 576]` + LI Cache `[seqlen, 1, 128]`
- Layer 1, 2, 3, 5, 6, 7, ... : Indexer Share 层，每层只含 1 个 Cache: KV Cache `[seqlen, 1, 576]`

这种情况下在第 1 节中提到的 UniformSpec KV Cache 分配方案会出现两个问题：首先，所有可用显存空间不能再均匀按 `num_layers` 分给每一层，因为对于相同长度的序列，Indexer Source 层和 Indexer Share 层的 KV Cache 占用并不相同；其次，每层可用的显存空间也不能再均匀按一个固定的 `page_size` 切分，因为 Indexer Source 层需要的两个 KV Cache `head_dim` 并不相同，因此对应的 `page_size` 也不同。

UniformType KV Cache 就是针对这种场景的设计——模型的多个 KV Cache 非相同规格 (Spec)，但整体而言仍然是相同的类型 (Type)，区别只是 `head_num`, `head_dim` 参数不同 (至于什么样的 KV Cache 连相同 Type 都不算，参见下一节)。

UniformType KV Cache 的分配思路简单来说就是：抛弃层的概念，直接根据不同 Cache 的 `page_size` 按比例分配显存空间。UniformType KV Cache 会首先会使用一个相同的 `block_size` 计算出所有层所有 KV Cache 的 `page_size`，计算公式还是跟之前一样的 `page_size` = `block_size` * `head_num` * `head_dim`。由于 `head_num`, `head_dim` 不同，不同 KV Cache 可能有不同的 `page_size`。然后所有这些 KV Cache 会被组合为一个大的 UniformType KV Cache，其 `page_size` 即为所有 KV Cache `page_size` 之和。这个 `UniformType KV Cache page_size` 的含义是：要存储 1 个 block 的 `block_size` 个 token，对于整个模型所有的 KV Cache 来说，需要占用的显存空间大小。

[graph 2-0]

然后在计算可分配 block 数量时，把所有可用显存空间大小按上述 Uniform Type KV Cache 所占大小均匀切分：`num_blocks` = `available_kv_cache_memory` / `UniformType KV Cache page_size`。注意这里使用的是原始的未除以模型层数的全部 `available_kv_cache_memory`，因为层的概念已经被融入了 `UniformType KV Cache page_size` 中。

[graph 2-1]

在实际申请 KV Cache Tensor 时，不同的 KV Cache 仍然分别各自申请独立的 Tensor。所有 KV Cache Tensor 最终都拥有相同的 `num_blocks` 和 `block_size`，这保证了所有 KV Cache 能存储的总上下文长度是相同的，不会出现部分 KV Cache 耗尽已无法调度新请求而另一部分 KV Cache 仍有可用显存空间被浪费的问题。不同 KV Cache 由于存储每个 token 需要占用的空间大小可能不同，所以每个 KV Cache Tensor 占用的显存大小可能是不同的，不同 KV Cache 占用显存大小之比就是它们的 `page_size` 之比，或者说是 `head_num * head_dim` 之比。

[graph 2-2]

最终的 KV Cache 结构如下：

```
kv_caches = {
    'layers.0.attn.self_attn': torch.Tensor, # [num_blocks, kv_head_num, kv_head_dim] (MLA KV Cache)
    'layers.0.attn.indexer': torch.Tensor, # [num_blocks, li_head_num, li_head_dim] (Indexer K Cache)
    'layers.1.attn.self_attn': torch.Tensor, # [num_blocks, kv_head_num, kv_head_dim] (MLA KV Cache)
    'layers.2.attn.self_attn': torch.Tensor,
    'layers.3.attn.self_attn': torch.Tensor,
    'layers.4.attn.self_attn': torch.Tensor,
    'layers.4.attn.indexer': torch.Tensor,
    ...,
}
```

最后在调度方面，跟 UniformSpec KV Cache 类似的，虽然模型一共有 `n` 层 `m` 个独立的 KV Cache tensor，但每个 KV Cache `num_blocks`, `block_size` 都相同，在分配时也是等比例分配，所以它们仍然可以共享同一份调度逻辑: UniformType KV Cache 也只需要 1 个 `kv_cache_manager`，其调度结果 (`new_block_ids`, `block_table` 等) 会被复制到模型每一层以相同的方式使用。

[graph 2-3]

UniformType KV Cache - 针对不同规格 KV Cache 做了改进之后，当前的分配方案看起来已经足够完善，不论模型每层有多少个 KV Cache，每个 KV Cache 的 `head_num`, `head_dim` 有多大区别，我们总能保证所有 KV Cache 都能被按需分配等比例的显存，可以容纳相同数量的 token，从而避免显存浪费。不过非线性增长 KV Cache 的出现又给我们带来的新的问题。

# 3: Hybrid KV Cache / HMA - 非线性增长 KV Cache

# 4: V4 - UniformType & Hybrid 的结合

# 5: Packed KV Cache - 为优化通信而设计的 Block 内连续组织方式

# 6: Prefill (Layerwise) KV Cache Offload - 层间复用 KV Cache

# 7: Decode (Sparse) KV Cache Offload (Ascend) - 基于 Uniform Type 的扩展

# 8: Decode (Sparse) KV Cache Offload (Hisparse) - 基于 HMA 的扩展
