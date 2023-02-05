import cv2
import torch
from itertools import count
import numpy as np

def generate_frames(mp4_path):
    '''generate stream of frames'''
    video = cv2.VideoCapture(mp4_path, cv2.CAP_FFMPEG)
    _, prev_frame = video.read()
    for t in count():
        ret, curr_frame = video.read()
        if ret == False:
            break
        yield prev_frame, curr_frame
        prev_frame = curr_frame
    video.release()
    cv2.destroyAllWindows()

def to_tt(image):
    return torch.Tensor(image).permute(2,0,1).unsqueeze(0)

def transform_image(image, flip = False):
    if flip:
        image = cv2.flip(image,1)
    return cv2.resize(image[200:-250,100:-100], (360, 128), interpolation = cv2.INTER_AREA)

def process(image, flip=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return to_tt(transform_image(image, flip)).to(device)

def m_a(a, n=3) :
  ret = np.cumsum(a, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return ret[n - 1:] / n


def get_mse(gt, test):
  test = np.nan_to_num(test)
  return np.mean(np.nanmean((gt - test)**2, axis=0))

def get_error(preds, gt, window):
  zero_mses = []
  mses = []

  window = 10
  zero_mses.append(get_mse(gt[window:], np.zeros_like(gt[window:])))

  test = preds

  pitch = m_a(preds[:,0],window)
  yaw = m_a(preds[:,1],window)
  test = list(zip(pitch,yaw))

  mses.append(get_mse(gt[window-1:], test))
  percent_err_vs_all_zeros = 100*np.mean(mses)/np.mean(zero_mses)
  print(f'error vs constant zero pred is {percent_err_vs_all_zeros:.2f}% ')