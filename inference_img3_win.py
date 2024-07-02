import os
import sys
import logging
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
import time
import socket
from model.pytorch_msssim import ssim_matlab
from io import BytesIO
import numpy as np
import struct
from queue import Queue, Empty
import _thread
from logging.handlers import TimedRotatingFileHandler 
import pdb

write_buffer = Queue()
read_buffer = Queue()

width = 1920
height = 1080
y_size = width * height
uv_size = y_size // 4 
frame_size = y_size + 2 * uv_size


def init_log(log_name='inference', filemode='a', FileHandler=TimedRotatingFileHandler, maxBytes=0):
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(formatter)
    fh = FileHandler(os.path.join(log_path, log_name+'.log'), when='D', interval=1, backupCount=60)
    fh.setFormatter(formatter)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def clear_write_buffer(write_buffer):
    count = 0
    wcount = 0
    while True:
        item = write_buffer.get()

        start_time = time.time()
        item_output = item[0]
        start_number = item[1]
        end_number = item[2]
        item_number = item[3]
        item_frames = item[4]
        
        if item_number == start_number:
            f = open(item_output, 'wb') 

        logger.info(f'item_number:{item_number}')
        for index, frame in enumerate(item_frames[:-1]):
            yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV_I420)
            f.write(yuv.tobytes())
            wcount += 1
            #cv2.imwrite(f'./{item_number}_{index}.png', frame[:,:,::-1])
        logger.info(f'_write {wcount} frames!')
        if item_number == end_number:
            yuv = cv2.cvtColor(item_frames[-1], cv2.COLOR_RGB2YUV_I420)
            f.write(yuv.tobytes()) 
            wcount += 1
            f.close()
        end_time = time.time()        

        logger.info(f'send time:{end_time-start_time}')

        count += 1
        logger.info(f'processed {item_number} frame!')
        logger.info(f'write {wcount} frames!')

_thread.start_new_thread(clear_write_buffer,(write_buffer,))
warnings.filterwarnings("ignore")

# set inference
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--gpu_n', dest='gpu_n', type=int, default=0)
parser.add_argument('--host', dest='host', type=str, default=None)
parser.add_argument('--port', dest='port', type=int, default=60001) 
parser.add_argument('--start_number', dest='start_number', type=int, default=0)
parser.add_argument('--gpu_count', dest='gpu_count', type=int, default=1)
parser.add_argument('--exp', dest='exp', type=int, default=1) 
args = parser.parse_args()
# initialize device 
device = torch.device(f"cuda:{args.gpu_n}" if torch.cuda.is_available() else "cpu")
gpu_count = args.gpu_count
multi = (2 ** args.exp)

logger = init_log(log_name=f'inference_{args.gpu_n}')
try:
    try:
        from model.RIFE_HDv2 import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v2.x HD model.")
    except:
        from train_log.RIFE_HDv3 import Model
        model = Model(gpu_n=args.gpu_n)
        model.load_model(args.modelDir, -1)
        print("Loaded v3.x HD model.")
except:
    from model.RIFE_HD import Model
    model = Model()
    model.load_model(args.modelDir, -1)
    print("Loaded v1.x HD model")
if not hasattr(model, 'version'):
    model.version = 0
model.eval()
model.device(gpu_n=args.gpu_n)

def receive_server():

    filename1 = 'input.yuv'   
    filename2 = 'input.yuv'

    outputname = 'output_.yuv'

    start_number = 20 
    end_number = 620 

    count = start_number
    f = open(filename1, 'rb')    
    f.seek(start_number * frame_size)
    frame1 = f.read(width*height*3//2)
    if len(frame1) < frame_size:
        f.close() 
        f = open(filename2, 'rb')
        frame1 = f.read(width*height*3//2)
    frame1 = np.frombuffer(frame1, dtype=np.uint8).reshape((height*3//2, width))
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_YUV2BGR_I420)
    while True:
        start_time = time.time()
        frame2 = f.read(width*height*3//2) 
        if len(frame2) < frame_size:
            f.close()
            f = open(filename2, 'rb')
            frame2 = f.read(width*height*3//2) 
        count += 1
        frame2 = np.frombuffer(frame2, dtype=np.uint8).reshape((height*3//2, width))
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_YUV2BGR_I420)

        end_time = time.time()
        logger.info(f'receive frames time:{end_time-start_time}')

        frames = np.stack([frame1[:,:,::-1], frame2[:,:,::-1]], axis=0) 
        read_buffer.put([outputname, start_number, end_number, count-1, frames])
        frame1 = frame2.copy()
        if count == end_number: 
            f.close()
            break
        
_thread.start_new_thread(receive_server,())
def pad_image(frame, padding):
    return F.pad(frame, padding)

def process_frames(frame1, frame2):
    h, w, c = frame1.shape  
    scale = 1
    tmp = max(128, int(128 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    I0 = torch.from_numpy(np.transpose(frame1, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.0
    I1 = torch.from_numpy(np.transpose(frame2, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.0
    I0 = pad_image(I0, padding)
    I1 = pad_image(I1, padding)
    I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
    I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
    ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
    output = [frame1]
    if ssim > 0.996:
        for i in range(multi-1):
            output.append(frame2)    
    elif ssim < 0.2: 
        for i in range(multi-1):
            output.append(frame1)
    else:
        for i in range(multi-1):
            output.append(np.ascontiguousarray((model.inference(I0, I1, (i+1) * 1./multi, 1)[0]*255.).byte().cpu().numpy().transpose(1,2,0)[:h,:w]))
    output.append(frame2)
    #return (model.inference(I0, I1, 1./2, 1)[0]*255.).byte().cpu().numpy().transpose(1,2,0)[:h,:w] 
    #return np.ascontiguousarray((model.inference(I0, I1, 1./2, 1)[0]*255.).byte().cpu().numpy().transpose(1,2,0)[:h,:w])
    return output

while True:
    item = read_buffer.get()
    if item is None:
        continue

    start_time = time.time()

    output_name = item[0]
    start_number = item[1]
    end_number = item[2]
    item_number = item[3]
    mid = process_frames(item[4][0], item[4][1])

    #yuv = cv2.cvtColor(mid[:,:,::-1], cv2.COLOR_RGB2YUV_I420)
    #yuv = yuv.tobytes()
    #mid = item[1][0]
    #cv2.imwrite(f"./tmp/{item_number}_mid.png", mid[:,:,::-1])

    end_time = time.time()
    logger.info(f'process time:{end_time-start_time}')
    write_buffer.put([output_name, start_number, end_number, item_number, mid])


