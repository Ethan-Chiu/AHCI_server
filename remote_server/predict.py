import sys
import numpy as np
from PIL import Image
import os
import platform
import sys
from pathlib import Path
import torch
import io

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

def process_image(image_data):
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
    if len(det[0]):
        masks = process_mask(proto[-1][0], det[0][:, 6:], det[0][:, :4], im.shape[2:], upsample=True)
        ormask = np.array(np.logical_or.reduce(masks.cpu(), axis=0)[..., np.newaxis])
        output = np.transpose(im.cpu()[0]*255, (1, 2, 0)).numpy()[:,:,::-1] * ormask
        normask = (1-ormask) * np.array([0,255,0])
        output = output + normask
        output = np.concatenate((output, ormask), axis=-1).astype(np.uint8)
    else:
        output = np.full_like(np.transpose(im.cpu()[0]*255, (1, 2, 0)), (0, 255, 0), dtype=np.uint8)

    image_bytes = cv2.imencode('.jpg', output)[1].tobytes()
    return image_bytes


if __name__ == "__main__":
    weights='models/gelan-c-seg.pt'
    source='yolov9/data/images/horses.jpg'
    data=ROOT / 'data/coco.yaml'
    imgsz=(640,640)
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

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    bs = 1
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

    # Read the image data from stdin until the delimiter is encountered
    while True:
        image_data = b''
        while True:
            chunk = sys.stdin.buffer.read(1024)
            if not chunk:
                break
            image_data += chunk
            if image_data.endswith(b"X") and b'IMAGE_COMPLETE' in image_data :  # Check for the delimiter
                image_data = image_data.split(b'IMAGE_COMPLETE')[0]
                break

        # Process the image
        result = process_image(image_data)

        # Convert the result to bytes and write it to stdout
        result_bytes = result + b'ARRAY_COMPLETE'
        result_bytes += b'X' * (1024 - (len(result_bytes) % 1024))
        sys.stdout.buffer.write(result_bytes)
        sys.stdout.flush()
