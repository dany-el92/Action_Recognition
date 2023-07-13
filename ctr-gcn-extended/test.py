from ctrgcn import Model
import yaml
import torch

config = yaml.safe_load(open('config.yaml'))
model = Model(**config['model_args'])
weights = torch.load("runs-50-3300.pt")
model.load_state_dict(weights)

print("ok")