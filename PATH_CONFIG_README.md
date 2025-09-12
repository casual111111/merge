# 实验和日志路径配置说明

## 概述

现在您可以通过配置文件自定义实验和日志的保存路径，而不必依赖固定的文件夹名称。`experiments` 和 `tb_logger` 文件夹将使用相同的名称。

## 新增配置参数

在配置文件的 `general settings` 部分添加以下参数：

```yaml
# general settings
name: diff-retinex_plus_lolv1_train
model_type: DRNetModel
num_gpu: 1
manual_seed: 1234

# 实验和日志保存路径配置
folder_name: my_custom_folder  # experiments和tb_logger共用文件夹名称，默认为name
```

## 参数说明

- `folder_name`: 指定 `experiments/` 和 `tb_logger/` 下的文件夹名称
  - 如果不设置，默认使用 `name` 字段的值
  - 同时影响实验相关文件和TensorBoard日志的保存位置

## 使用示例

### 示例1：使用默认路径（不设置folder_name）
```yaml
name: diff-retinex_plus_lolv1_train
# 结果：
# experiments/diff-retinex_plus_lolv1_train/
# tb_logger/diff-retinex_plus_lolv1_train/
```

### 示例2：自定义文件夹名称
```yaml
name: diff-retinex_plus_lolv1_train
folder_name: my_experiment_v1
# 结果：
# experiments/my_experiment_v1/
# tb_logger/my_experiment_v1/
```

### 示例3：使用时间戳或版本号
```yaml
name: diff-retinex_plus_lolv1_train
folder_name: diff_retinex_v2_20250912
# 结果：
# experiments/diff_retinex_v2_20250912/
# tb_logger/diff_retinex_v2_20250912/
```

## 文件夹结构

无论使用什么名称，文件夹结构保持不变：

```
experiments/{folder_name}/
├── models/                    # 模型权重文件
├── training_states/           # 训练状态文件
├── visualization/             # 可视化结果
├── train_{name}_{timestamp}.log  # 训练日志
└── {name}.yml                # 配置文件副本

tb_logger/{folder_name}/
└── tensorboard_logs/         # TensorBoard日志文件
```

## 注意事项

1. 如果指定的文件夹已存在，系统会自动创建归档版本（添加时间戳后缀）
2. 建议使用有意义的文件夹名称，便于实验管理
3. 可以结合时间戳、版本号等来区分不同的实验
4. 修改配置后重新运行训练即可使用新的路径结构

## 配置文件示例

参考 `options/train_diff_retinex_plus_lolv1_custom_path.yml` 文件查看完整的使用示例。