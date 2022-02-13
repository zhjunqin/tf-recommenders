# GRPC client python example

A python GRPC client for TF serving example 

## Usage help

```
# python grpc_client.py  --h
usage: grpc_client.py [-h] --server SERVER 
                           --config_dir CONFIG_DIR
                           [--batch_size BATCH_SIZE] 
                           --model_name MODEL_NAME 
                           [--signature_name SIGNATURE_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --server SERVER       PredictionService host:port
  --config_dir CONFIG_DIR
                        Dir for config yaml
  --batch_size BATCH_SIZE
                        batch size
  --model_name MODEL_NAME
                        model name
  --signature_name SIGNATURE_NAME
                        signature name
```

## Usage example

```
# export PYTHONPATH=$PYTHONPATH:/path/to/recommenders
# python grpc_client.py \
    --server  host:port
    --config_dir /path/to/recommenders/config/mmoe
    --model_name mmoe
```
