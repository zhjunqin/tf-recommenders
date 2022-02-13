# TF Recommenders

## 目录文件列表

主要的目录文件列表的含义
- docker: Dockerfile
- docs： 文档
- recommenders: TF 模型代码
  - config: 模型的特征等配置，使用 ymal 格式
  - estimator: 各个模型的 estimator
  - models: 模型的定义
  - layers: 各个子 layer 的定义
  - serving: TF serving grpc client 示例
  - util: 一些 util 代码
- local_run.sh: 本地执行训练任务的执行脚本

## Models

目前包含的模型
- Wide&Deep: https://arxiv.org/abs/1606.07792
- MMOE: https://dl.acm.org/doi/10.1145/3219819.3220007
- DeepFM: https://arxiv.org/abs/1703.04247
- DCN: https://arxiv.org/abs/1708.05123
- DCNv2: https://arxiv.org/abs/2008.13535
