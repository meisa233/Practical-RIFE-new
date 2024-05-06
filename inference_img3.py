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

import ctypes
from ctypes import *
logger = None
frame1 = np.empty([1080, 1920, 3], dtype=np.uint8)
frame2 = np.empty([1080, 1920, 3], dtype=np.uint8)

rows, cols, chanel = frame1.shape

dataptr1 = frame1.ctypes.data_as(c_char_p)
dataptr2 = frame2.ctypes.data_as(c_char_p)

ll = cdll.LoadLibrary
lib = ctypes.cdll.LoadLibrary('./librgbqueue.so')

lib.TestReadFromFile.restype = c_int
lib.TestReadFromFile.argtypes = [ctypes.c_char_p]

lib.GetQueueSize.restype = c_int

lib.GetFrameMatrix.restype = c_int
lib.GetFrameMatrix.argtypes = [ctypes.c_char_p, ctypes.c_int]

lib.FinishFrameGenerate.restype = c_int
lib.FinishFrameGenerate.argtypes = [ctypes.c_char_p, ctypes.c_int]


write_buffer = Queue()
read_buffer = Queue()

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
res = lib.TestReadFromFile('20240418_150354_001.mxf.rgb'.encode('utf-8'))
res = lib.TestReadFromFile('20240418_150354_001.mxf.rgb'.encode('utf-8'))

#res = lib.TestReadFromFile('split_red_1920x1080_rgb.rgb'.encode('utf-8'))
#res = lib.TestReadFromFile('split_red_1920x1080_rgb.rgb'.encode('utf-8'))
def clear_write_buffer(write_buffer):
    count = 0
    while True:
        item = write_buffer.get()

        start_time = time.time()
        item_number = item[0]
        item_frame = item[1]
        pdb.set_trace()
        dataptr3 = item_frame.ctypes.data_as(c_char_p) 
        lib.FinishFrameGenerate(dataptr3, item_number)
        end_time = time.time()

        logger.info(f'send time:{end_time-start_time}')
        del dataptr3
        del item_frame
        cv2.imwrite(f'processed__{count}.png', item[1][:,:,::-1])
        count += 1
        logger.info(f'processed {item_number} frame!')

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
parser.add_argument('--gpu_count', dest='gpu_count', type=int, default=4)
args = parser.parse_args()
# initialize device 
device = torch.device(f"cuda:{args.gpu_n}" if torch.cuda.is_available() else "cpu")
gpu_count = args.gpu_count

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
    i = args.start_number 
    while True:
        start_time = time.time() 
        status1 = lib.GetFrameMatrix(dataptr1, i) 

        while status1 != 0:
            status1 = lib.GetFrameMatrix(dataptr1, i)
        pdb.set_trace()
        status2 = lib.GetFrameMatrix(dataptr2, i+1)
        while status2 != 0: 
            status2 = lib.GetFrameMatrix(dataptr2, i + 1)

#        for h in range(1080):  # 高度
    # 在一行内生成所有相同高度的像素值列表
#            row_pixels = ' '.join(str(frame1[h, w].tolist()) for w in range(1920))
#            with open('res3.txt', 'a') as f:
#                 f.writelines([f'Height {h}: {row_pixels}\n'])
        end_time = time.time()
        logger.info(f'receive frames time:{end_time-start_time}')
        frames = np.stack([frame1, frame2], axis=0) 
        read_buffer.put([i,frames])
        i = i+gpu_count 
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
    if ssim > 0.996: print(ssim);return frame2
    if ssim < 0.2: print(ssim);return frame1
    return (model.inference(I0, I1, 1./2, 1)[0]*255.).byte().cpu().numpy().transpose(1,2,0)[:h,:w] 

while True:
    item = read_buffer.get()
    if item is None:
        continue
    start_time = time.time()
    item_number = item[0]
    mid = process_frames(item[1][0], item[1][1])
    end_time = time.time()
    logger.info(f'process time:{end_time-start_time}')
    write_buffer.put([item_number,mid])

