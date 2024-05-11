import logging
import torch
from hmipt.utils.config import get_config_from_json
from hmipt.src.models.hmipt import HmipT
# from hmipt import HmipT
# from hmipt import process_config

config_path = "./hmipt.json"
config, _ = get_config_from_json(config_path)

logger = logging.getLogger()

model = HmipT(config=config, logger=logger) 

checkpoint_path = "./checkpoint_fianl.pth.tar"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])
