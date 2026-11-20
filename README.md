# QYOLO-OSCS

YOLO Open-set Cold-start Segmentation (YOLO-OSCS) 的 PyTorch 研究骨架。

当前实现刻意不包含图像加载、数据增强、真实数据集、正式目标分配和完整训练循环。它提供可运行的 YOLOv11 风格网络编排、分割与开放集输出、模块化损失，以及共享模型的训练/推理接口。

## 结构

```text
configs/                  模型、损失和推理超参数
oscs/models/              Ultralytics 模块适配、backbone、neck、heads、统一模型
oscs/losses/              检测、分割、开放集和组合损失
oscs/training/            单步训练任务包装
oscs/inference/           预测解码包装
oscs/utils/               跨模块数据契约
main.py                   仅负责加载配置并组装组件
tests/                    随机张量契约测试
```

## Ultralytics 依赖边界

所有上游 YOLOv11 模块都集中在 `oscs/models/ultralytics_blocks.py`：`Conv`、`C3k2`、`SPPF` 与 `C2PSA`。其他本地模块不直接依赖 Ultralytics 内部导入路径，方便后续升级版本时集中处理兼容性。

`oscs/models/components.py` 提供可独立优化的 `PixelUnshuffleStem`：`PixelUnshuffle(4) -> 1x1 Conv(C) -> BatchNorm2d -> 3x3 Conv(C/2)`。它通过 `use_pixel_unshuffle_stem` 和 `pixel_unshuffle_factor` 注入 backbone，默认启用。

## 运行

在已安装 PyTorch、Ultralytics 和 PyYAML 的环境中执行：

```bash
python main.py
pytest -q
```

## 当前损失基线

`CompositeLoss` 汇总分类、边界框、掩码 BCE、Dice 和 open-set knownness 损失。当前为验证接口提供了一个隔离的一对一槽位匹配基线；后续接入 TaskAlignedAssigner、Hungarian matching 或面向冷启动支持集的匹配策略时，只需替换 `oscs/losses/detection_loss.py` 的匹配逻辑，不需要改动网络、训练或推理接口。
