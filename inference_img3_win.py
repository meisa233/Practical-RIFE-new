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

#import ctypes
#from ctypes import *
#logger = None
#frame1 = np.empty([1080, 1920, 3], dtype=np.uint8)
#frame2 = np.empty([1080, 1920, 3], dtype=np.uint8)
#
#rows, cols, chanel = frame1.shape
#
#dataptr1 = frame1.ctypes.data_as(c_char_p)
#dataptr2 = frame2.ctypes.data_as(c_char_p)
#
#ll = cdll.LoadLibrary
##lib = ctypes.cdll.LoadLibrary('./librgbqueue.so')
#lib = ctypes.cdll.LoadLibrary('./librgbqueueyun.so')
#
#lib.TestReadFromFile.restype = c_int
#lib.TestReadFromFile.argtypes = [ctypes.c_char_p]
#
#lib.GetQueueSize.restype = c_int
#
#lib.GetFrameMatrix.restype = c_int
#lib.GetFrameMatrix.argtypes = [ctypes.c_char_p, ctypes.c_int]
#
#lib.FinishFrameGenerate.restype = c_int
#lib.FinishFrameGenerate.argtypes = [ctypes.c_char_p, ctypes.c_int]


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
#res = lib.TestReadFromFile('20240418_150354_001.mxf.rgb'.encode('utf-8'))
#res = lib.TestReadFromFile('20240418_150354_001.mxf.rgb'.encode('utf-8'))

#res = lib.TestReadFromFile('split_red_1920x1080_rgb.rgb'.encode('utf-8'))
#res = lib.TestReadFromFile('split_red_1920x1080_rgb.rgb'.encode('utf-8'))
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

        #print(time.time(),"Converted array order before:", item_frame.flags['C_CONTIGUOUS'])  # 应该输出True
        #dataptr3 = item_frame.ctypes.data_as(c_char_p) 
        #lib.FinishFrameGenerate(dataptr3, item_number)
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
        #print(time.time(),"Converted array order end:", item_frame.flags['C_CONTIGUOUS'])  # 应该输出True

        logger.info(f'send time:{end_time-start_time}')
        #del dataptr3
        #del item_frame

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

## start tcpserver
#lib.startTcpServerThread()
#
#size_ = lib.GetQueueSize()
#num = 0
#while size_ <= 0:
#    size_ = lib.GetQueueSize()
#    num += 1
#    logger.info(f'current size:{size_} sleeped {num} seconds')
#    time.sleep(1)

def receive_server():
    i = args.start_number 
    start_number = 0
    end_number = 598 
    while True:
        start_time = time.time() 
#        status1 = lib.GetFrameMatrix(dataptr1, i) 
#        while status1 != 0:
#            status1 = lib.GetFrameMatrix(dataptr1, i)
#            #logger.info(f'can\'t get dataptr1! index{i}')
#            time.sleep(0.005)
#
#        #pdb.set_trace()
#        status2 = lib.GetFrameMatrix(dataptr2, i+1)
#        while status2 != 0: 
#            status2 = lib.GetFrameMatrix(dataptr2, i + 1)
#            #logger.info(f'can\'t get dataptr2! index{i + 1}')
#            time.sleep(0.005)

        #dataptr3 = frame1.ctypes.data_as(c_char_p)
        #lib.FinishFrameGenerate(dataptr3, i)
        filename = 'output.yuv'
        frame1 = cv2.imread(f'./tmp_images2/{i}.png')[:,:,::-1]
        frame2 = cv2.imread(f'./tmp_images2/{i+1}.png')[:,:,::-1]
        end_time = time.time()
        logger.info(f'receive frames time:{end_time-start_time}')

        frames = np.stack([frame1, frame2], axis=0) 
        read_buffer.put([filename, start_number, end_number, i, frames])

        i = i+gpu_count 
        if i == end_number+1:
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


