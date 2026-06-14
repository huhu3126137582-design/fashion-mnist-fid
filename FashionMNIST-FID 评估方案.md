# FashionMNIST-FID 评估方案

## 1. 指标定义

- 指标名称统一为 `FashionMNIST-FID`。
- 使用 FashionMNIST 分类器最终 logits 层之前的 128 维倒数第二层输出作为特征。
- 特征张量形状必须为 `[B, 128]`。
- 128 维特征必须是最终分类层的直接输入。
- 不使用分类器 logits。
- 不使用 softmax 输出。
- 不对特征执行 L2 normalize。
- FID 的均值、协方差和 Frechet 距离使用 `float64` 计算。

## 2. 分类器规范

### 2.1 输入

- 输入形状统一为 `[B, 1, 32, 32]`。
- 输入数据类型统一为 `float32`。
- 输入值域统一为 `[0, 1]`。
- 输入形状、数据类型或值域不符合要求时必须终止并报告错误。
- 分类器内部执行固定的 FashionMNIST mean/std 标准化。
- mean/std 必须记录在 classifier checkpoint 中。

### 2.2 模型接口

- 分类器提供 `forward(x)`，返回形状为 `[B, 10]` 的 logits。
- 分类器提供 `extract_features(x)`，返回形状为 `[B, 128]` 的特征。
- `extract_features(x)` 的输出必须直接送入最终的 `Linear(128, 10)` 分类层。
- 特征提取和 FID 计算期间分类器必须处于 eval 模式。
- 分类器参数和特征推理使用 `float32`。

### 2.3 训练与 checkpoint

- 使用固定随机种子。
- 使用固定的分层训练集/验证集划分。
- 保存验证准确率最佳的模型。
- 最佳模型必须在完整 FashionMNIST test10k 上评估。
- test10k 分类准确率必须大于或等于 `93%`。
- 准确率低于 `93%` 的 checkpoint 不得发布为正式 classifier checkpoint。
- 准确率低于 `93%` 的 checkpoint 不得用于创建 real stats 或执行 FID。
- 正式 checkpoint 使用原子写入。
- checkpoint 至少包含：
  - 模型 state dict
  - 架构版本
  - 特征版本
  - 特征维度 `128`
  - 输入形状 `[1, 32, 32]`
  - 输入值域 `[0, 1]`
  - 标准化 mean/std
  - 数据划分信息
  - 随机种子
  - 最佳验证准确率
  - test10k 准确率

## 3. Real Stats 规范

### 3.1 数据和预处理

- 真实分布固定使用完整 FashionMNIST test10k，共 10,000 张图像。
- 原始 `28x28` 图像使用固定的双线性插值 resize 到 `32x32`。
- resize 后输入保持为 `[0, 1]` 的 `float32` 单通道张量。
- 分类器内部执行与训练一致的标准化。
- test10k 的所有样本必须参与 real stats 计算。

### 3.2 统计量

- 提取形状为 `[10000, 128]` 的特征。
- 使用 `float64` 计算特征均值。
- 使用 `float64` 计算无偏协方差。
- 缓存至少包含：
  - `mu`
  - `sigma`
  - 样本数 `10000`
  - 特征维度 `128`
  - classifier checkpoint SHA256
  - 架构版本
  - 特征版本
  - classifier test10k 准确率
  - 输入形状和值域
  - resize 参数
  - 标准化 mean/std
  - FashionMNIST test 数据文件指纹

### 3.3 缓存校验和重建

- real stats 缓存默认使用严格校验。
- 缓存缺失时默认终止并报告错误。
- 缓存损坏时默认终止并报告错误。
- classifier checkpoint SHA256 不匹配时默认终止并报告错误。
- 架构版本、特征版本或特征维度不匹配时默认终止并报告错误。
- 数据文件指纹或预处理元数据不匹配时默认终止并报告错误。
- 不允许默认静默创建、覆盖或修复 real stats。
- 只有显式传入 `--rebuild_real_stats` 时才允许重新计算 real stats。
- 重建必须使用当前指定的 classifier checkpoint 和完整 test10k。
- 重建结果必须原子写入。
- 重建完成后必须重新加载并执行完整一致性校验。
- real stats 文件名或目录必须包含 classifier checkpoint hash 标识。

## 4. MF/iMF 评估规范

### 4.1 共享资源

- 所有 MF 和 iMF checkpoint 必须使用同一个正式 classifier checkpoint。
- 所有 1-step、2-step 和 4-step 评估必须使用同一份 test10k real stats。
- 每次评估必须记录 classifier checkpoint SHA256。
- 每次评估必须记录 real stats SHA256。
- classifier checkpoint 或 real stats 身份不一致的结果不得汇总比较。

### 4.2 采样

- 评估入口支持一次执行 `--sample_steps 1 2 4`。
- classifier checkpoint 和 real stats 在一次评估中只加载一次。
- 默认生成 10,000 张样本。
- 生成样本数必须能被 10 整除，否则终止并报告错误。
- 10 个类别的生成数量必须完全相等。
- 默认 10,000 张评估中每类生成 1,000 张。
- 1-step、2-step 和 4-step 使用相同随机种子。
- 1-step、2-step 和 4-step 使用相同类别顺序。
- 1-step、2-step 和 4-step 使用相同初始噪声。
- 每个 step 必须严格生成指定数量，不得因 batch 大小向上取整。

### 4.3 生成图像输入

- 生成模型的原始输出必须保持 `[B, 1, 32, 32]`。
- 在 clamp 前先累计越界像素统计。
- 完成越界统计后将图像 clamp 到 `[0, 1]`。
- clamp 后以 `float32` 的 `[B, 1, 32, 32]` 张量输入分类器。
- 不将灰度图复制为 RGB。

### 4.4 Pre-clamp 越界像素统计

- 每个 sample step 分别记录：
  - `pre_clamp_below_zero_pixel_ratio`
  - `pre_clamp_above_one_pixel_ratio`
  - `pre_clamp_out_of_range_pixel_ratio`
- `pre_clamp_below_zero_pixel_ratio` 为值小于 `0` 的像素数除以全部生成像素数。
- `pre_clamp_above_one_pixel_ratio` 为值大于 `1` 的像素数除以全部生成像素数。
- `pre_clamp_out_of_range_pixel_ratio` 为前两项比例之和。
- 边界值 `0` 和 `1` 不计为越界。
- 比例必须在该 step 的全部生成样本上按像素总数累计。
- 不得先计算 batch ratio 再对 batch ratio 求平均。
- 越界统计必须发生在任何 clamp、量化或类型转换之前。

### 4.5 Fake Stats 和 FID

- 使用 classifier 的 `extract_features()` 提取 fake features。
- fake features 不执行 softmax、logits 替换或 L2 normalize。
- 使用 `float64` 计算 fake 特征均值和无偏协方差。
- 使用与 `pytorch-fid` 一致的 Frechet 距离统计定义计算分数。
- 结果必须是有限的非负数。
- 输出指标名称必须为 `FashionMNIST-FID`。
- 不得将结果标记为标准 Inception-FID。

## 5. 结果记录

每个 MF/iMF checkpoint 和 sample step 的报告至少包含：

- 生成模型 checkpoint 路径
- 生成模型 checkpoint SHA256
- 模式：MF 或 iMF
- sample step
- CFG 配置
- 随机种子
- 生成样本总数
- 每类生成样本数
- classifier checkpoint 路径
- classifier checkpoint SHA256
- classifier test10k 准确率
- real stats 路径
- real stats SHA256
- real stats 样本数
- 特征版本
- 特征维度 `128`
- `pre_clamp_below_zero_pixel_ratio`
- `pre_clamp_above_one_pixel_ratio`
- `pre_clamp_out_of_range_pixel_ratio`
- `FashionMNIST-FID`

## 6. 测试规范

### 6.1 分类器测试

- 验证 `forward(x)` 输出形状为 `[B, 10]`。
- 验证 `extract_features(x)` 输出形状为 `[B, 128]`。
- 验证 128 维特征是最终 logits 层的直接输入。
- 验证 feature 路径不包含 softmax。
- 验证 feature 路径不返回 logits。
- 验证 feature 路径不执行 L2 normalize。
- 验证输入形状、数据类型和值域检查。
- 验证分类器内部标准化使用 checkpoint 中的 mean/std。
- 验证 test10k 准确率低于 `93%` 时拒绝发布和评估。

### 6.2 Real Stats 测试

- 验证 real stats 使用完整 test10k。
- 验证特征维度、均值和协方差形状。
- 验证统计量使用 `float64`。
- 验证缓存缺失时默认失败。
- 验证缓存损坏时默认失败。
- 验证 classifier hash 不匹配时默认失败。
- 验证数据或预处理元数据不匹配时默认失败。
- 验证未传入 `--rebuild_real_stats` 时不会创建或覆盖缓存。
- 验证传入 `--rebuild_real_stats` 后可以重建并通过严格校验。

### 6.3 越界统计测试

- 使用人工构造的越界张量验证三个 ratio。
- 验证值为 `0` 和 `1` 的像素不计为越界。
- 验证统计使用所有 batch 的像素总数。
- 验证统计发生在 clamp 之前。
- 验证分类器接收到的张量已 clamp 到 `[0, 1]`。

### 6.4 FID 测试

- 验证相同特征分布的 FID 接近 `0`。
- 验证不同特征分布得到有限非负值。
- 验证 fake stats 使用 `float64`。
- 验证生成样本数严格等于请求值。
- 验证 10 个类别的样本数量完全相等。
- 验证 1/2/4-step 共享相同 seed、类别顺序和初始噪声。
- 验证 MF/iMF 结果共享相同 classifier checkpoint 和 real stats 身份。

## 7. 验收标准

- 正式 classifier checkpoint 的 test10k 准确率大于或等于 `93%`。
- 128 维特征符合 logits 前直接输入、不使用 softmax/logits、不做 L2 normalize 的要求。
- 所有分类器输入均为 `[0,1]` 的 `[B,1,32,32]` `float32` 张量。
- real stats 默认严格校验，所有不匹配情况均失败。
- 只有 `--rebuild_real_stats` 可以触发 real stats 重建。
- MF/iMF 的 1/2/4-step 评估共用同一 classifier checkpoint 和 real stats。
- 每个 step 的 pre-clamp 越界像素比例均被完整记录。
- 评估报告包含全部模型、缓存、采样和指标身份信息。
