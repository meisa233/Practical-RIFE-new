import os
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

write_buffer = Queue()
read_buffer = Queue()

def clear_write_buffer(write_buffer):
    count = 0
    while True:
        item = write_buffer.get()
        cv2.imwrite(f'processed_{count}.png', item[:,:,::-1])
        count += 1
        print('processed 1 frame!')

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
args = parser.parse_args()
# initialize device 
device = torch.device(f"cuda:{args.gpu_n}" if torch.cuda.is_available() else "cpu")

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

def receive_file_from_connection(conn):
    # 接收包头（8字节文件大小 + 1字节文件类型 + 4字节 frame1 + 4字节 frame2）
    header = conn.recv(17)  # 8 + 1 + 4 + 4 = 17 bytes
    if not header:
        return False  # 没有接收到包头，可能连接已经关闭

    file_size, file_type, frame1, frame2 = struct.unpack('>Qbii', header)
    print(f"Receiving file of size {file_size} bytes with type {file_type}")
    print(f"Frame1 value: {frame1}, Frame2 value: {frame2}")

    # 根据文件大小接收文件内容
    received_bytes = 0
    buffer = BytesIO()
    while received_bytes < file_size:
        chunk_size = min(file_size - received_bytes, 4096)
        chunk = conn.recv(chunk_size)
        if not chunk:
            break  # 连接可能已经关闭
        buffer.write(chunk) 
        received_bytes += len(chunk)

    if received_bytes < file_size:
        print('File did not receive completely.')
        return False
    buffer.seek(0) 
    npy_data = np.load(buffer, allow_pickle=False)
    read_buffer.put(npy_data)
def receive_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print('begin receiving data...') 
        s.bind((args.host, args.port))
        s.listen()
        print(f"Server is listening on {args.host}:{args.port}")
    
        while True:  # 无限循环，接受新的连接
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
    
                while True:  # 对于每个连接，持续接收数据直到连接关闭
                    success = receive_file_from_connection(conn)
                    if not success:
                        break  # 如果接收过程中出现问题，跳出循环处理下一个连接

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
    mid = process_frames(item[0], item[1])
    end_time = time.time()
    print(f'end_time-start_time:{end_time-start_time}')
    write_buffer.put(mid)

