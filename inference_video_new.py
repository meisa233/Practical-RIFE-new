import os.path
import traceback

import cv2
import torch
import sys
import numpy as np
from model.pytorch_msssim import ssim_matlab
import pdb
from torch.nn import functional as F

def pad_image(img):
    return F.pad(img, padding)

gpu_n = 0
multi = 2
device = torch.device(f"cuda:{gpu_n}" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
try:
    from train_log.RIFE_HDv3 import Model
except:
    traceback.print_exc()
    print("Please download our model from model list")

video_path = "argvsfrabbc108050en_cut.mkv"
Capture = cv2.VideoCapture(video_path)
if not Capture.isOpened():
    print('Error: Could not open video.')
    sys.exit(-1)
frame_count = int(Capture.get(cv2.CAP_PROP_FRAME_COUNT))
count = 0
redundancy = 0
model = Model(gpu_n=gpu_n)
modelDir = 'train_log'
if not hasattr(model, 'version'):
    model.version = 0
model.load_model('train_log', -1)
print("Loaded 3.x/4.x HD model.")
model.eval()
model.device(gpu_n=gpu_n)
scale = 1.0
def make_inference(I0, I1, n):
    global model
    if model.version >= 3.9:
        res = []
        for i in range(n):
            res.append(model.inference(I0, I1, (i+1) * 1. / (n+1), scale))
        return res
    else:
        middle = model.inference(I0, I1, scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n//2)
        second_half = make_inference(middle, I1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]
fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
video_path_wo_ext, ext = os.path.splitext(video_path)
fps = Capture.get(cv2.CAP_PROP_FPS)
vid_out_name = f'{video_path_wo_ext}_{multi}X_{int(np.round(fps))}.{ext}'

while True:
    ret, frame = Capture.read()
    if ret:
        break
    count += 1
h,w,_ = frame.shape
tmp = 128
ph = ((h-1)//tmp+1) * tmp
pw = ((w-1)//tmp+1) * tmp
padding = (0, pw-w, 0, ph-h)
vid_out = cv2.VideoWriter(vid_out_name, fourcc, fps, (w,h))
write_count = 0
temp = None
while True:
    if temp is not None:
        newframe = temp
        temp = None
    else:
        ret, newframe = Capture.read()
    if newframe is None:
        break
    if ret:
        I0 = np.copy(frame[:,:,::-1]) 
        I0 = torch.from_numpy(np.transpose(I0, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float()/255.
        I0 = pad_image(I0)
        I1 = np.copy(newframe[:,:,::-1])
        I1 = torch.from_numpy(np.transpose(I1, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float()/255.
        I1 = pad_image(I1)
        I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
        break_flag = False
        if ssim > 0.996:
            ret, newframe = Capture.read() # read a new frame
            if newframe is None:
                break_flag = True
                newframe = frame
            else:
                temp = newframe
            tmp_frame = np.copy(newframe[:,:,::-1])
            I1 = torch.from_numpy(np.transpose(tmp_frame, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(
                0).float() / 255.
            I1 = pad_image(I1)
            I1 = model.inference(I0, I1, 1)
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            newframe = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        if ssim < 0.2:
            output = []
            for i in range(2 - 1):
                output.append(I0)
            '''
            output = []
            step = 1 / args.multi
            alpha = 0
            for i in range(args.multi - 1):
                alpha += step
                beta = 1-alpha
                output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
            '''
        else:
            output = make_inference(I0, I1, 2 - 1)
    count +=1
    print(count)
    vid_out.write(frame)
    for mid in output:
        count += 1
        vid_out.write((((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))[:h,:w,::-1])
        print(count)
    frame = newframe
    if break_flag:
        break

