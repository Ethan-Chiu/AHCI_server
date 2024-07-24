import sys
import numpy as np
from PIL import Image
import os
import platform
import sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WORLD_SIZE"] = "1"
import torch
import io
from functools import reduce
import logging
import torch.nn as nn
from hmipt.utils.config import get_config_from_json
from hmipt.src.models.hmipt import HmipT
from torchvision import transforms
from collections import deque

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

sys.path.append('./yolov9')

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox
from PIL import Image
import time
import pickle


def process_image(image_data, is_torch_mode):
    outputfile = open("output.txt", 'a')
    im0 = np.array(Image.open(io.BytesIO(image_data)))
    im = letterbox(im0, imgsz)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred, proto = model(im, augment=augment, visualize=False)[:2]
    det = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

    yolo_history.append(proto[-1][0])

    if len(history) < 5:
        outputfile.write(f"starting...{len(history)}\n")
        return None
    
    handhead = list(history)
    hand = [a[0] for a in handhead]
    head = [a[1] for a in handhead]

    # proto = torch.randn(5, 32, 60, 48).cuda()
    proto_five = torch.stack(list(yolo_history)).cuda()
    # hand = torch.randn(1, 5, 24, 14).cuda()
    hand = torch.tensor(hand).unsqueeze(0).cuda()
    # head = torch.randn(1, 5, 7).cuda()
    head = torch.tensor(head).unsqueeze(0).cuda()

    if is_torch_mode:
        saves = proto, pred, history[-1]
        cur_time = int(time.time()*100000)
        with open(f'saves/{cur_time}.pkl', 'wb') as f:
            pickle.dump(saves, f)

    pooling_layer = nn.AvgPool2d(kernel_size=2).cuda()
    pooled_proto: np.ndarray = pooling_layer(proto_five)
    pooled_proto = pooled_proto.unsqueeze(dim=0)
    prediction = model2(pooled_proto, hand, head)
    prediction = prediction.squeeze().detach()

    left_distance = torch.dist(hand[0,3,0,0:3], hand[0,4,0,0:3])
    right_distance = torch.dist(hand[0,3,0,7:10], hand[0,4,0,7:10])

    threshold = 0
    if left_distance < threshold and right_distance < threshold:
        output = np.full_like(np.transpose(im.cpu()[0]*255, (1, 2, 0)), (0, 255, 0), dtype=np.uint8)
    elif len(det[0]):
        obj_labels = torch.tensor([39, 41, 64, 66, 67]).to(model.device)
        obj_mask = torch.isin(det[0][:, 5], obj_labels)
        hand_labels = torch.tensor([0,]).to(model.device)
        hand_mask = torch.isin(det[0][:, 5], hand_labels)
        filtered = det[0][obj_mask]
        handed = det[0][hand_mask]
        blank = torch.zeros(1, 38).to(model.device)
        filtered = torch.cat((filtered, blank), dim=0)
        product = torch.matmul(filtered[:, 6:], prediction)
        closest_index = torch.argmax(product)
        with_hand = torch.cat((filtered[closest_index:closest_index+1], handed), dim=0)
        with_hand[0, 6:] = prediction
        masks = process_mask(proto[-1][0], with_hand[:, 6:], with_hand[:, :4], im.shape[2:], upsample=True)
        # masks = process_mask(proto[-1][0], det[0][:, 6:], det[0][:, :4], im.shape[2:], upsample=True)
        # masks = process_mask(proto[-1][0], prediction.unsqueeze(0), det[0][:, :4], im.shape[2:], upsample=True)
        ormask = np.array(np.logical_or.reduce(masks.cpu(), axis=0)[..., np.newaxis])
        outputfile.write(str(ormask.shape))
        ormask = ormask * 255.0
        np_array = ormask.astype(np.uint8)
        outputfile.write(str(np_array.shape))
        np_array = np.squeeze(np_array, axis=2)
        output = Image.fromarray(np_array, mode='L')
        output = np.array(output)
        image_bytes = cv2.imencode('.jpg', output)[1].tobytes()
        return image_bytes

if __name__ == "__main__":
    weights='models/gelan-c-seg.pt'
    source='yolov9/data/images/horses.jpg'
    data=ROOT / 'data/coco.yaml'
    imgsz=(480,480)
    conf_thres=0.25
    iou_thres=0.45
    max_det=1000
    device='0'
    view_img=False
    save_txt=False
    save_conf=False
    save_crop=False
    nosave=False
    classes=None
    agnostic_nms=False
    augment=False
    visualize=False
    update=False
    project='runs'
    name='test'
    exist_ok=True
    line_thickness=3
    hide_labels=False
    hide_conf=False
    half=False
    dnn=False
    vid_stride=1
    retina_masks=False
    queue=None
    stop=None
    
    device = int(sys.argv[1])
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    config_path = "./hmipt.json"
    checkpoint_path = "./models/checkpoint_5.pth.tar"
    config, _ = get_config_from_json(config_path)
    logger = logging.getLogger()
    model2 = HmipT(config=config, logger=logger) 
    checkpoint = torch.load(checkpoint_path)
    model2.load_state_dict(checkpoint['state_dict'])
    model2 = model2.cuda()

    history = deque(maxlen=5)
    yolo_history = deque(maxlen=5)

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    bs = 1
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

    # Read the image data from stdin until the delimiter is encountered
    is_torch_mode = False
    while True:
        byte_data = b''
        while True:
            chunk = sys.stdin.buffer.read(1024)
            if not chunk:
                break
            byte_data += chunk
            if byte_data.endswith(b"X") and b'IMAGE_COMPLETE' in byte_data :  # Check for the delimiter
                byte_data = byte_data.split(b'IMAGE_COMPLETE')[0]
                break
        byte_data = byte_data.split(b'handhead')
        image_data = byte_data[0]
        byte_data = byte_data[1].split(b'torch')
        handhead = byte_data[0].decode("utf-8").split('\n')
        hand = np.array([reduce(lambda x, y: x+y, [[float(n) for n in handhead[m].split('(')[k].split(')')[0].split(', ')] for k in range(1,5)]) for m in range(24)], dtype=np.float32)
        head = np.array(reduce(lambda x, y: x+y, [[float(n) for n in handhead[24].split('(')[k].split(')')[0].split(', ')] for k in range(1,3)]), dtype=np.float32)
        history.append((hand, head))
        if len(byte_data) > 1:
            torch_xy = byte_data[1].decode("utf-8").split(',')
            torch_x = float(torch_xy[0])
            torch_y = float(torch_xy[1])
            is_torch_mode = True
        else:
            is_torch_mode = False
        # Process the image
        result = process_image(image_data, is_torch_mode)

        if result:
            # Convert the result to bytes and write it to stdout
            result_bytes = result + b'ARRAY_COMPLETE'
        else:
            result_bytes = b'ARRAY_COMPLETE'
        result_bytes += b'X' * (1024 - (len(result_bytes) % 1024))
        sys.stdout.buffer.write(result_bytes)
        sys.stdout.flush()
