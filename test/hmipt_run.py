import logging
import torch
from hmipt.utils.config import get_config_from_json
from hmipt.src.models.hmipt import HmipT
import numpy as np
import torch.nn as nn
# from hmipt import HmipT
# from hmipt import process_config

config_path = "./hmipt.json"
config, _ = get_config_from_json(config_path)

logger = logging.getLogger()

model = HmipT(config=config, logger=logger) 

checkpoint_path = "./checkpoint_7.pth.tar"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])
model = model.cuda()


# _, proto = yolo(imgs_input)[:2]
# proto: np.ndarray = proto[-1]
proto = torch.randn(5, 32, 60, 48).cuda()
hand = torch.randn(1, 5, 24, 14).cuda()
head = torch.randn(1, 5, 7).cuda()


pooling_layer = nn.AvgPool2d(kernel_size=2).cuda()
pooled_proto: np.ndarray = pooling_layer(proto)
pooled_proto = pooled_proto.unsqueeze(dim=0)

output = model(pooled_proto, hand, head)
output = output.squeeze()

print(output)
print(len(output))