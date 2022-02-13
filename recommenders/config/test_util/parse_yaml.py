import json
import sys
import yaml

# Use example
# python parse_yaml.py xx.yaml

file_name = sys.argv[1]
print("yaml file name: {}".format(file_name))
with open(file_name, 'r') as f:
    y = yaml.safe_load(f)
    print(json.dumps(y, indent=2))

