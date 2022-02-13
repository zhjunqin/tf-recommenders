# 快速开始

本地执行一个 Recommenders 的模型训练。

## 构建执行环境

按照下面的命令，逐个执行，构建所需环境

```
$ curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
$ bash /tmp/miniconda.sh  -b -p /data/conda-py38  # 请根据自己需要修改安装路径

$ vi ~/.bashrc # 修改 ~/.bashrc 增加如下行
export PATH=/data/conda-py38/bin:$PATH   # 增加该行
$ source ~/.bashrc
$ which conda
/data/conda-py38/bin/conda
$ source /data/conda-py38/bin/activate

# clone 该项目
$ pip install -r requirement.txt
```
