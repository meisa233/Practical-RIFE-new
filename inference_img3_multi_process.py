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
import threading
from logging.handlers import TimedRotatingFileHandler 
from torch.multiprocessing import Pool, Manager, Process
import pdb

import ctypes
from ctypes import *
logger = None
frame1 = np.empty([1080, 1920, 3], dtype=np.uint8)
frame2 = np.empty([1080, 1920, 3], dtype=np.uint8)
frame3 = np.empty([1080, 1920, 3], dtype=np.uint8)
frame4 = np.empty([1080, 1920, 3], dtype=np.uint8)
frame5 = np.empty([1080, 1920, 3], dtype=np.uint8)
frame6 = np.empty([1080, 1920, 3], dtype=np.uint8)
frame7 = np.empty([1080, 1920, 3], dtype=np.uint8)
frame8 = np.empty([1080, 1920, 3], dtype=np.uint8)

rows, cols, chanel = frame1.shape

dataptr1 = frame1.ctypes.data_as(c_char_p)
dataptr2 = frame2.ctypes.data_as(c_char_p)
dataptr3 = frame3.ctypes.data_as(c_char_p)
dataptr4 = frame4.ctypes.data_as(c_char_p)
dataptr5 = frame5.ctypes.data_as(c_char_p)
dataptr6 = frame6.ctypes.data_as(c_char_p)
dataptr7 = frame7.ctypes.data_as(c_char_p)
dataptr8 = frame8.ctypes.data_as(c_char_p)

def init_model(gpu_n):
    try:
        try:
            from model.RIFE_HDv2 import Model
            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v2.x HD model.")
        except:
            from train_log.RIFE_HDv3 import Model
            model = Model(gpu_n=gpu_n)
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
    model.device(gpu_n=gpu_n)
    logger.info(f'init model, gpu:{gpu_n}')
    return model

ll = cdll.LoadLibrary
#lib = ctypes.cdll.LoadLibrary('./librgbqueue.so')
lib = ctypes.cdll.LoadLibrary('./librgbqueueyun.so')

lib.TestReadFromFile.restype = c_int
lib.TestReadFromFile.argtypes = [ctypes.c_char_p]

lib.GetQueueSize.restype = c_int

lib.GetFrameMatrix.restype = c_int
lib.GetFrameMatrix.argtypes = [ctypes.c_char_p, ctypes.c_int]

lib.FinishFrameGenerate.restype = c_int
lib.FinishFrameGenerate.argtypes = [ctypes.c_char_p, ctypes.c_int]

def add_args_wrap(data):
    process_task(*data)

#write_buffer = Queue()
#read_buffer = Queue()
frame_list = [[frame1, frame2], [frame3, frame4], [frame5, frame6], [frame7, frame8]]
dataptr_list = [[dataptr1, dataptr2], [dataptr3, dataptr4], [dataptr5, dataptr6], [dataptr7, dataptr8]]
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
    while True:
        item = write_buffer.get()

        start_time = time.time()
        item_number = item[0]
        item_frame = item[1]
        #cv2.imwrite(f"./tmp/{item_number}_mid.png", item_frame[:,:,::-1])
        #pdb.set_trace()
        print("Converted array order:", item_frame.flags['C_CONTIGUOUS'])  # 应该输出True
        dataptr3 = item_frame.ctypes.data_as(c_char_p) 
        lib.FinishFrameGenerate(dataptr3, item_number)
        end_time = time.time()

        logger.info(f'send time:{end_time-start_time}')
        del dataptr3
        del item_frame
        #cv2.imwrite(f'processed__{count}.png', item[1][:,:,::-1])
        count += 1
        logger.info(f'processed {item_number} frame!')

#_thread.start_new_thread(clear_write_buffer,(write_buffer,))
warnings.filterwarnings("ignore")

# set inference
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--host', dest='host', type=str, default=None)
parser.add_argument('--port', dest='port', type=int, default=60001) 
parser.add_argument('--gpu_count', dest='gpu_count', type=int, default=4)
args = parser.parse_args()
# initialize device 
gpu_count = args.gpu_count
model_list = []
device_list = []
#model_list = []
#device_list = []
#for i in [0, 1, 2, 3]:
    #model_list.append(init_model(i))
    #device_list.append(torch.device(f"cuda:{i}" if torch.cuda.is_available() else "cpu"))
logger = init_log(log_name=f'inference')

#try:
    #try:
        #from model.RIFE_HDv2 import Model
        #model = Model()
        #model.load_model(args.modelDir, -1)
        #print("Loaded v2.x HD model.")
    #except:
        #from train_log.RIFE_HDv3 import Model
        #model = Model(gpu_n=args.gpu_n)
        #model.load_model(args.modelDir, -1)
        #print("Loaded v3.x HD model.")
#except:
    #from model.RIFE_HD import Model
    #model = Model()
    #model.load_model(args.modelDir, -1)
    #print("Loaded v1.x HD model")
#if not hasattr(model, 'version'):
    #model.version = 0
#model.eval()
#model.device(gpu_n=args.gpu_n)

# start tcpserver
lib.startTcpServerThread()

size_ = lib.GetQueueSize()
num = 0
while size_ <= 0:
    size_ = lib.GetQueueSize()
    num += 1
    logger.info(f'current size:{size_} sleeped {num} seconds')
    time.sleep(1)
def process_task(gpu_n, gpu_count):
    logger.info(f"thread number:{gpu_n}")
    global model_list, device_list, frame_list, dataptr_list
    #model = model_list[gpu_n]
    model = init_model(gpu_n)
    device = torch.device(f"cuda:{gpu_n}" if torch.cuda.is_available() else "cpu") 
    #device = device_list[gpu_n]
    write_buffer = Queue()
    read_buffer = Queue()
    frame1 = frame_list[gpu_n][0]
    frame2 = frame_list[gpu_n][1]
    dataptr1 = dataptr_list[gpu_n][0]
    dataptr2 = dataptr_list[gpu_n][1]
    start_number = gpu_n
    _thread.start_new_thread(receive_server,(read_buffer, frame1, frame2, dataptr1, dataptr2, start_number, gpu_count))
    _thread.start_new_thread(clear_write_buffer, (write_buffer,))
    while True:
        item = read_buffer.get()
        if item is None:
            continue
        start_time = time.time()
        item_number = item[0]
        mid = process_frames(model, device, item[1][0], item[1][1])
        #yuv = cv2.cvtColor(mid[:,:,::-1], cv2.COLOR_RGB2YUV_I420)
        #yuv = yuv.tobytes()
        #mid = item[1][0]
        #cv2.imwrite(f"./tmp/{item_number}_mid.png", mid[:,:,::-1])
        end_time = time.time()
        logger.info(f'gpu number:{gpu_n}, process time:{end_time-start_time}')
        write_buffer.put([item_number,mid])
    #read_t = threading.Thread(target=receiver_server,args=(read_buffer, frame1, frame2, dataptr1, dataptr2, gpu_count))
    #write_t = threading.Thread(target=write_thread, args=(read_buffer,write_buffer))
    #clear_write_buffer_t = threading.Thread(target=clear_write_buffer, args=(writer_buffer,))

    #clear_write_buffer_t.start() 
    #read_t.start()
    #write_t.start() 
    
    #read_t.join()
    #write_t.join()
    #clear_write_buffer_t.join()



def receive_server(read_buffer, frame1, frame2, dataptr1, dataptr2, start_number, gpu_count):
    i = start_number 
    while True:
        start_time = time.time() 
        status1 = lib.GetFrameMatrix(dataptr1, i) 
        while status1 != 0:
            status1 = lib.GetFrameMatrix(dataptr1, i)
            logger.info('can\'t get dataptr1!')
        status2 = lib.GetFrameMatrix(dataptr2, i + 1)
        while status2 != 0: 
            status2 = lib.GetFrameMatrix(dataptr2, i + 1)
            logger.info('can\'t get dataptr2!')

        
        end_time = time.time()
        logger.info(f'receive frames time:{end_time-start_time}')
        #frame1 = cv2.imread(f'./tmp/1_1.png') 
        #frame2 = cv2.imread(f'./tmp/1_2.png')
        #cv2.imwrite(f'./tmp/{i}_2.png', frame2[:,:,::-1])
        frames = np.stack([frame1, frame2], axis=0) 
        read_buffer.put([i,frames])
        i = i+gpu_count 
        
#_thread.start_new_thread(receive_server,())
def pad_image(frame, padding):
    return F.pad(frame, padding)

def process_frames(model, device, frame1, frame2):
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
    #return (model.inference(I0, I1, 1./2, 1)[0]*255.).byte().cpu().numpy().transpose(1,2,0)[:h,:w] 
    return np.ascontiguousarray((model.inference(I0, I1, 1./2, 1)[0]*255.).byte().cpu().numpy().transpose(1,2,0)[:h,:w])
def write_thread(read_buffer, write_buffer):
    while True:
        item = read_buffer.get()
        if item is None:
            continue
        start_time = time.time()
        item_number = item[0]
        mid = process_frames(item[1][0], item[1][1])
        #yuv = cv2.cvtColor(mid[:,:,::-1], cv2.COLOR_RGB2YUV_I420)
        #yuv = yuv.tobytes()
        #mid = item[1][0]
        #cv2.imwrite(f"./tmp/{item_number}_mid.png", mid[:,:,::-1])
        end_time = time.time()
        logger.info(f'process time:{end_time-start_time}')
        write_buffer.put([item_number,mid])
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    model_list = []
    device_list = []
    #for i in [0, 1, 2, 3]:
    #    model_list.append(init_model(i))
    #    device_list.append(torch.device(f"cuda:{i}" if torch.cuda.is_available() else "cpu"))
    with Pool(processes = args.gpu_count, maxtasksperchild=1) as pool:
        inputs = []
        for i in range(args.gpu_count):
            inputs.append((i, args.gpu_count))
        pool.map(add_args_wrap,inputs,chunksize=1) 
