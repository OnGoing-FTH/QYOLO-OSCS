# QYOLO-OSCS 开发约定

本文档记录当前 YOLO Open-set Cold-start Segmentation（YOLO-OSCS）项目的架构边界和实现注意事项。后续新增模块、损失或训练逻辑时应遵守这些规则。

## 当前范围

当前阶段只实现模型结构、前向特征流、损失计算接口和训练/推理包装。

暂不实现以下内容：

- 图像文件读取、数据集和 DataLoader。
- 数据增强。
- 完整训练循环、优化器、学习率调度和检查点。
- 正式目标分配策略。
- 完整后处理，例如 NMS、实例掩码裁剪和评估指标。

不要为了补齐上述未定义能力而将其混入模型、损失或 `main.py`。

## 目录职责

```text
configs/                  超参数配置，不承载模型逻辑
oscs/models/              网络组件、backbone、neck、heads 和组装模型
oscs/losses/              检测、分割、开放集损失及其组合
oscs/training/            训练单步包装，不负责读取数据
oscs/inference/           预测解码包装，不复制模型前向
oscs/utils/types.py       跨模块数据契约
main.py                   组合根：加载配置、创建模型/损失/包装、最小演示
tests/                    pytest 契约与回归测试
```

`main.py` 只能负责装配，不应放入网络层定义、损失公式或数据处理细节。

## Ultralytics 依赖边界

- 仅在 `oscs/models/ultralytics_blocks.py` 中导入 Ultralytics YOLOv11 模块。
- 当前集中导出的上游组件包括：`Conv`、`C3k2`、`SPPF`、`C2PSA`。
- 其他本地模型文件不得直接依赖 Ultralytics 的深层导入路径。
- 升级 Ultralytics 时，优先在适配层处理 API 和行为兼容问题。

## 训练与推理共享模型

- `YOLOOpenSetColdStartSegmentation` 是训练和推理唯一共享的网络实例。
- 模型 `forward(images)` 始终返回 `ModelOutput` 原始预测，不根据训练/推理模式改变输出数据结构。
- `TrainingTask` 负责调用模型和 `CompositeLoss`。
- `Predictor` 负责对同一 `ModelOutput` 做分数、类别、top-k 和掩码解码。
- 不得在训练或推理包装中复制 backbone、neck 或 heads 的前向实现。

## 数据契约

跨模块传递的数据结构定义在 `oscs/utils/types.py`。

- `ModelOutput`：网络原始输出，包含类别 logits、框、mask coefficients、mask prototypes、实例 mask、knownness logits 和 embeddings。
- `BatchTargets`：当前损失基线使用的稠密占位目标。`labels == -1` 表示 padding/无效实例。
- `LossOutput`：总损失和按名称区分的损失分项。
- `PredictionBatch`：推理解析后的 scores、labels、boxes、masks、knownness。

修改任一契约前，必须同步评估所有 consumers：模型、损失、训练包装、推理包装和测试。

## PixelUnshuffleStem

`oscs/models/components.py` 中的 `PixelUnshuffleStem` 是独立、可持续优化的输入组件。当前固定特征流为：

```text
PixelUnshuffle(r) -> 1x1 Conv(C) -> Norm(C) -> 3x3 Conv(C/2)
```

张量形状规则：

```text
输入:                 [B, Cin, H, W]
PixelUnshuffle(r):    [B, Cin * r^2, H/r, W/r]
1x1 Conv(C):          [B, C, H/r, W/r]
Norm(C):              [B, C, H/r, W/r]
3x3 Conv(C/2):        [B, C/2, H/r, W/r]
```

默认案例：`Cin=3`、`H=W=128`、`r=4`、`C=64`。

```text
[B, 3, 128, 128]
-> [B, 48, 32, 32]
-> [B, 64, 32, 32]
-> [B, 64, 32, 32]
-> [B, 32, 32, 32]
```

实现约束：

- 输入高、宽必须能被 `downscale_factor` 整除。
- `hidden_channels` 即 `C`，必须是大于等于 2 的偶数，因为最终输出通道为 `C/2`。
- 不要将该组件的内部细节分散到 backbone 中；后续优化应优先修改 `components.py`。
- `forward_with_features()` 必须保留，并返回 `input`、`pixel_unshuffle`、`project`、`norm`、`output`，供结构调试和特征流检查使用。
- 组件通过 `use_pixel_unshuffle_stem` 与 `pixel_unshuffle_factor` 接入 backbone，默认启用；关闭时应保持原始 stem 路径可用。

## 多分辨率邻近扩展

`oscs/models/multi_resolution.py` 负责将固定 RGB 输入尺度对齐到 `[B, 3, 640, 640]`，再进行自适应邻近融合。

- 支持的原始输入分辨率仅为：`5120`、`2560`、`1280`、`640`，且均为 `[B, 3, H, W]`。
- 禁止在该模块及其子组件中调用 `resize`、`interpolate`、池化 resize 或任何隐式尺度变换；尺寸准备属于数据加载/预处理层。
- 5120 对齐路径固定为：`PixelUnshuffleStem(r=2) -> PixelUnshuffleStem(r=4) -> ChannelHalvingBlock(12->6) -> ChannelHalvingBlock(6->3)`。
- 2560 对齐路径固定为：`PixelUnshuffleStem(r=2) -> PixelUnshuffleStem(r=2)`。
- 1280 对齐路径固定为：`PixelUnshuffleStem(r=2)`。
- 640 输入使用 `Identity` 路径。
- 四条路径的输出都必须是 `[B, 3, 640, 640]`，才允许输入 `AdaptiveNeighborFusion`。
- `AdaptiveNeighborFusion` 通过全局描述预测四个尺度的 softmax 权重，对齐特征加权求和后再做卷积细化。
- 真实尺寸测试只要求覆盖 5120 分支，使用 `(1, 3, 5120, 5120)` 作为输入并开启 `torch.inference_mode()`。

## 特征流记录

新增或修改会改变特征尺寸、通道数、分支合并方式的组件时：

1. 在该组件的 docstring 中写明输入、各关键中间特征和输出的形状公式。
2. 前向过程使用具名中间变量，不要用多层嵌套的一行表达式隐藏特征流。
3. 对关键研究组件提供类似 `forward_with_features()` 的可选 trace 接口，返回命名后的中间特征。
4. 当改变 backbone/neck 的尺度时，检查检测、分割、开放集 heads 的 query 数量和 mask 分辨率是否仍一致。

## 损失与目标分配

- `CompositeLoss` 只负责编排 DetectionLoss、SegmentationLoss 和 OpenSetLoss。
- 当前 `_matched_targets()` 是用于接口验证的一对一槽位匹配基线：目标槽位 `i` 对应预测槽位 `i`。
- 这不是正式检测目标分配算法。接入 TaskAlignedAssigner、Hungarian matching 或 cold-start support-set 匹配时，应优先替换匹配逻辑，不改动模型输出契约。
- 新增损失分项时，必须返回命名明确的标量 Tensor，并由 `CompositeLoss` 汇总。
- 空正样本场景必须返回与模型输出相连的零损失，保证反向传播图有效。

## 模块独立自测

每个实际执行单元都必须拥有独立入口：

```python
if __name__ == "__main__":
    # 构造最小随机输入
    # 执行该模块
    # 打印关键形状或标量结果
```

要求：

- 使用 `python -m 包路径` 执行，例如 `python -m oscs.models.components`。
- 网络组件打印关键输入/输出或 trace 形状。
- 损失模块打印命名损失标量。
- 训练、推理包装打印输出/预测的关键形状与损失信息。
- 纯导出文件、纯数据类型文件和 Ultralytics 适配层不需要独立入口。
- 包级 `__init__.py` 采用惰性导出，避免模块通过 `python -m` 运行时产生预加载警告。

当前可独立执行的模块包括：

```text
oscs.models.components
oscs.models.backbone
oscs.models.neck
oscs.models.heads.detection_head
oscs.models.heads.segmentation_head
oscs.models.heads.open_set_head
oscs.models.model
oscs.losses.detection_loss
oscs.losses.segmentation_loss
oscs.losses.open_set_loss
oscs.losses.composite_loss
oscs.training.task
oscs.inference.predictor
```

## 验证要求

每次修改后至少执行与改动范围相对应的命令：

```bash
python -m oscs.models.components
pytest -q
python main.py
```

如果修改了公共输出、损失、训练或推理契约，必须运行全量测试。提交前执行：

```bash
git diff --check
```
