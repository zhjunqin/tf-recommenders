# 代码介绍

## 目录文件列表 

主要的目录文件列表的含义
- recommenders: TF 模型代码
  - config: 模型的特征等配置，使用 ymal 格式
  - estimator: 各个模型的 estimator
  - models: 模型的定义
  - layers: 各个子 layer 的定义
  - serving: TF serving grpc client 示例
  - util: 一些 util 代码
