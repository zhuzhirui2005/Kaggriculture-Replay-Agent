![Kaggriculture 回放学习智能体：等距农场、协作工人与神经网络连接](assets/readme-banner.png)

<p align="center">
  <a href="README.md">English</a> | 简体中文
</p>
<p align="center">
  <a href="https://www.kaggle.com/competitions/kaggriculture">比赛概览</a> | <a href="https://www.kaggle.com/competitions/kaggriculture/submissions?submissionId=56720708">对战记录</a> | <a href="https://www.kaggle.com/competitions/kaggriculture/writeups/51st-place-solution-replay-bc">方案详解</a>
</p>

这是一个基于 PyTorch 的 Kaggriculture 智能体，通过行为克隆从对局回放中学习工人指令和市场决策。仓库提供约 2240 万参数的预训练模型、CPU 推理、回放预处理，以及单设备和分布式训练代码。

模型结合了棋盘卷积编码器、日内循环记忆、工人间的注意力机制和自回归市场解码器。它直接预测完整动作，而不是从预设的经营策略库中选择策略。

## 安装

需要 Python 3.11 或更高版本。以下命令均在仓库根目录执行，建议使用虚拟环境：

```sh
python -m pip install -e .
```

推理在 CPU 上运行。CUDA 训练需要支持 CUDA 的 PyTorch 版本，以及支持 bfloat16 的 GPU。多 GPU 分布式训练面向 Linux，使用 NCCL。

如需在本地运行对局，请同时安装官方环境：

```sh
python -m pip install -e ".[environment]"
```

本项目适配 `kaggle-environments==1.32.7` 的规则：棋盘大小为 10 × 10，每局包含 720 个状态，每个游戏日有 24 个回合。策略最多支持 32 个行动实体（农夫及工人），每次决策最多输出 10 笔市场订单。

## 运行预训练模型

模型权重位于 `weights/model.pt`。以下示例让两个独立的智能体实例完成一场对局：

```python
from kaggle_environments import make
from farm_policy.agent import Agent

env = make("kaggriculture", debug=True)
env.run([Agent("weights/model.pt"), Agent("weights/model.pt")])
print([player.reward for player in env.state])
```

接入其他对局运行器时，调用 `agent(observation, configuration)` 即可获得动作字典。每场对局中的每个玩家都应使用独立的 `Agent` 实例，并按时间顺序传入观测。推理封装会维护动作历史，在每日开始时重置循环记忆，并处理同一回合的重复调用，避免重复推进记忆状态。

默认策略使用固定的风格标签 `default`。风格是模型对自身行为模式的条件表示，不是对手标识。默认启用最后一个可行动回合的库存清仓处理，同时保留网络输出的工人指令。

## 构建提交包

```sh
python -m tools.package --weights weights/model.pt --output submission.tar.gz
```

提交包包含入口文件、推理模块和模型权重，不包含训练工具及文档。

打包自行训练的模型时，将对应的 `step-XXXXXX.pt` 文件传给 `--weights`，并选择一个尚不存在的输出路径。

## 模型结构

| 组件 | 配置 |
| --- | --- |
| 棋盘编码器 | 52 个输入通道、192 个隐藏通道、3 个残差块 |
| 共享状态与工人状态的表示 | 768 维 |
| 日内记忆 | 共享 GRU 与逐工人 GRU，各为 256 维 |
| 工人协同 | 2 层注意力，每层 4 个头 |
| 工人动作 | 44 类指令，并以指令为条件预测数量 |
| 市场动作 | 从含 23 个 token 的词表中依次生成，最多输出 10 笔订单 |

输入特征包括公共棋盘与市场状态、智能体的私有库存、历史动作，以及已观测的市场变化。回放中的未来状态仅用于生成训练标签。局部动作掩码和按顺序扣减的共享种子额度共同约束工人动作的预测。

训练在动作模仿之外，还使用工人近期任务意图和后续分差变化作为辅助目标。随机擦除历史动作输入，降低模型对完美示范历史的依赖。

## 准备回放数据

请提供已结束对局的官方回放，格式为 `.json` 或 `.json.gz`。本仓库不附带回放数据。

创建 `data/replays.json`，每场对局对应一个条目。文件路径相对于这份清单文件解析。以下文件名仅为示例，请替换为自己的回放：

```json
{
  "episodes": [
    {
      "id": "game_001",
      "file": "replays/game_001.json.gz",
      "players": {"0": "default"},
      "group": "new"
    },
    {
      "id": "game_002",
      "file": "replays/game_002.json",
      "players": {"1": "default"},
      "group": "rehearsal"
    }
  ]
}
```

`players` 指定要模仿的玩家席位及其训练风格标签。用于微调默认策略的示范可以标记为 `default`；也可以使用自己定义且前后一致的标签来训练不同风格。同一个条目中可以同时包含双方席位。可选字段 `group` 默认为 `new`；将旧示范标记为 `rehearsal`，即可在后续训练中混入这些数据。

```sh
python -m tools.data prepare --manifest data/replays.json --output data/encoded --workers 4
python -m tools.data cache --source data/encoded --output data/cache
```

预处理会将每条观测与其后记录的动作对齐，并按玩家席位写出压缩数组。程序根据对局 ID，按约 80/10/10 的比例划分训练集、验证集和测试集，同一对局的双方始终位于同一分区。请保持 ID 稳定，不要用不同 ID 重复列出同一场对局，并提供足够多的对局以填充训练集和验证集。

缓存包含训练集和验证集的只读内存映射数组，不包含测试集对局。这些数组未经压缩，需要预留额外磁盘空间；上述两条数据准备命令都要求输出目录为空。

## 微调与训练

使用准备好的数据微调预训练策略：

```sh
python -m tools.train --cache data/cache --output runs/finetune --warmstart weights/model.pt --style default --batch-days 8 --steps 12000
```

清单中的风格标签决定哪些条件参与训练。`--style` 只选择保存到模型中的默认部署条件，不会重新标记或筛选数据。已有风格会复用对应的嵌入；新增标签会获得随机初始化的嵌入，同时保留共享的预训练网络。训练会更新网络参数，已有风格不会被冻结。

从零开始训练时，省略 `--warmstart`：

```sh
python -m tools.train --cache data/cache --output runs/scratch --style default --batch-days 8 --steps 12000
```

在 Linux 上启动 4 个 CUDA 训练进程：

```sh
torchrun --standalone --nproc_per_node=4 -m tools.train --cache data/cache --output runs/distributed --warmstart weights/model.pt --style default --batch-days 32 --steps 12000
```

`--batch-days` 表示所有训练进程合计的全局批量，而不是每张 GPU 的批量。该值必须能被训练进程数整除。每个游戏日对应一段 24 帧的序列；显存不足时，请减小批量。

默认训练配置使用 AdamW，权重衰减为 `1e-4`，梯度范数裁剪阈值为 1，历史动作擦除概率为 15%。从零训练时，默认学习率为 `1e-4`；从已有权重继续训练时为 `1e-5`。新旧两组数据都存在时，70% 的样本取自新示范，30% 取自旧示范。对新示范组，先均匀采样风格，再选择轨迹和游戏日。这些设置是起点，可以通过命令行参数调整。

训练会定期生成可直接部署的 `step-XXXXXX.pt` 文件，并将最新的优化器状态和随机状态保存在 `training.pt` 中。断点续训时，应保持数据、批量大小、训练进程数和采样设置一致：

```sh
python -m tools.train --cache data/cache --output runs/finetune --resume runs/finetune/training.pt --style default --batch-days 8 --steps 16000
```

`--steps` 表示总更新次数，包含已经完成的更新。恢复分布式训练时，请沿用相同的 `torchrun` 启动方式。在运行目录中创建名为 `STOP` 的文件，会使程序在下一次更新后保存检查点并退出；恢复训练前需删除该文件。可使用 `--device cpu` 进行小规模调试。

验证损失和动作准确率衡量的是回放预测能力，不是对战强度。请固定模型检查点，在未参与筛选的世界中进行完整对局评测，并覆盖双方席位。本仓库提供可复用的训练方法；若要精确复现随仓库提供的权重，还需要原始示范数据和完整训练过程。

## 仓库结构

| 路径 | 用途 |
| --- | --- |
| `main.py` | Kaggle 入口文件 |
| `farm_policy/` | 模型、特征编码、有状态推理及最终回合清仓处理 |
| `tools/data.py` | 回放预处理、数据划分与序列批量采样 |
| `tools/train.py` | 行为克隆与分布式训练 |
| `tools/package.py` | 提交包构建工具 |
| `weights/model.pt` | 预训练模型权重 |
