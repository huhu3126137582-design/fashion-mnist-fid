# FashionMNIST-FID 指标规范

完整规范见仓库根目录的 `FashionMNIST-FID 评估方案.md`。

核心约束：

- 使用正式 FashionMNIST 分类器 logits 前的 128 维特征。
- 输入为 `[B, 1, 32, 32]`、`float32`、值域 `[0, 1]`。
- 不使用 logits、softmax 或 L2 normalization 作为特征。
- 均值、无偏协方差和 Frechet 距离使用 `float64`。
- 真实分布固定为完整 FashionMNIST test10k。
- classifier checkpoint 的 test10k 准确率必须不低于 93%。
- real stats 必须严格绑定 classifier SHA256 和数据指纹。
- 该指标名称为 `FashionMNIST-FID`，不是标准 Inception-FID。
