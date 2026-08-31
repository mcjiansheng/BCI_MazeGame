# 文档索引

本目录记录 BCI Maze 的数据处理、模型实验、算法说明和 Unity 联调信息。项目入口说明请先看根目录 [README.md](../README.md)。

## 当前实现与实验文档

- [BCIC2a 预处理与三模型实现说明](BCIC2a预处理与三模型实现说明.md)：BCIC2a 数据格式、事件窗口、滤波器组和基础模型。
- [BCIC2b 预处理与适配说明](BCIC2b预处理与适配说明.md)：BNCI 004-2014 / BCIC2b 的 MAT 读取和三通道适配。
- [EEG-Conformer、FBCNet、SE-MHAF 模型解析](EEG_Conformer_FBCNet_SE-MHAF_模型解析.md)：模型结构和输入输出关系。
- [BCIC2a 三模型实验结果](BCIC2a三模型实验结果.md)：BCIC2a 基线结果。
- [BCIC2a 与 2b 三模型综合实验报告](BCIC2a与2b三模型综合实验报告.md)：两套数据集的统一比较。
- [SE-MHAF-Conformer V2 优化说明](SE-MHAF-Conformer_V2优化说明.md)：V2 的结构和训练策略。
- [SE-MHAF-Conformer 最终优化与验证报告](SE-MHAF-Conformer_最终优化与验证报告.md)：Final 模型、验证结果和复现实验入口。
- [迷宫生成与寻路算法原理](迷宫生成与寻路算法原理.md)：Unity 迷宫生成、BFS 寻路和可视化逻辑。
- [Unity BCI Maze 接口说明](../Maze-game/Assets/README_BCI_Maze.md)：键盘操作、UDP 协议、ACK 语义和 Python 联调。
- [LK-Mini-EEG16 在线 MI 系统调研与实施工作流](在线MI系统调研与实施工作流.md)：设备资料依据、外部项目与论文调研、当前问题、目标架构、任务顺序和验收条件。

## 外部 Box 资料

- Box 资料阅读与项目参考分析.md：对南京大学 Box 文件夹的下载清单、目录结构、可复用设计、不可直接复用部分和后续建议的详细分析；该报告仅保留在本地，不同步到远端。
- 原始下载资料位于项目本地 tmp/box-nju-1644522/，该目录由 .gitignore 忽略，仅用于本地阅读，不应上传 Git。
- Box 资料包含访谈文本、受试者安排和面部行为数据等敏感内容。仓库文档只保留结构性结论，不复制个人信息、原始文本或外部项目的未经验证指标。

## 背景材料

仓库中还保留了一些项目历史材料、论文笔记、汇报稿和阶段性方案。它们用于追溯设计来源，不一定与当前代码完全同步；其中 PDF、Word、PPT 和若干中文阶段性文档由 .gitignore 排除，不作为当前实现的唯一依据。

阅读或复现实验时，以以下内容为准：

1. 根目录 README.md；
2. src/bci_maze/ 和 scripts/ 的当前代码；
3. outputs/ 中的已保存结果；
4. 本目录中对应的实验报告；
5. 背景材料和外部 Box 资料分析。

## 结果文件与复核

实验结果保存在 outputs/ 下，默认不提交：

- benchmark：基准模型汇总；
- se_mhaf_v2：V2 结果；
- se_mhaf_final：Final 结果；
- checkpoints/：模型权重；
- bci_maze_integration_test.json：Unity 联调冒烟测试记录。

如果结果、文档和代码不一致，应先检查脚本默认参数、数据预处理缓存和 checkpoint 配置，再运行 scripts/verify_final_results.py 复核。
