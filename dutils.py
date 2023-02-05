import random
import torch
import tqdm
from utils import generate_frames, process
import numpy as np
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
import torch.utils.data as data_utils

def train_data(videos_train=[0,1,2,4], flips=False):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model_large = raft_large(weights=Raft_Large_Weights.C_T_V2, progress=False).to(device)
  model_large.eval()
  with torch.no_grad():
    for v in videos_train:
        video_path, txt_path = f'labeled/{v}.hevc', f'labeled/{v}.txt'
        angles = np.loadtxt(txt_path)[1:]
        labels = angles[~np.isnan(angles).any(axis=1)]
        true_t = 0
        for t, (prev,curr) in enumerate(tqdm(generate_frames(video_path))):
            if np.isnan(angles[t]).any():
                continue
            if flips:
              if flip := random.choice([True,False]):
                  labels[true_t][1]*=-1
            prev, curr = process(prev, flip).to(device), process(curr,flip).to(device)
            flow = model_large(prev,curr)[-1]
            if t==0:
                data = flow
            else:
                data = torch.cat([data,flow])
            true_t+=1
        dataset = torch.utils.data.TensorDataset(data, torch.from_numpy(labels).float().to(device))
        torch.save(dataset, f'pts/{v}_train.pt')

def eval_data(videos_eval=[3]):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model_large = raft_large(weights=Raft_Large_Weights.C_T_V2, progress=False).to(device)
  model_large.eval()
  with torch.no_grad():
    for v in videos_eval:
        video_path, txt_path = f'labeled/{v}.hevc', f'labeled/{v}.txt'
        angles = np.loadtxt(txt_path)[1:]
        labels = angles[~np.isnan(angles).any(axis=1)]
        for t, (prev,curr) in enumerate(tqdm(generate_frames(video_path))):
            if np.isnan(angles[t]).any():
                continue
            prev, curr = process(prev).to(device), process(curr).to(device)
            flow = model_large(prev,curr)[-1]
            if t==0:
                data = flow
            else:
                data = torch.cat([data,flow])
        dataset = torch.utils.data.TensorDataset(data, torch.from_numpy(labels).float().to(device))
        torch.save(dataset, f'pts/{v}_eval.pt')

def train_eval_datasets(train_ids=[0,1,2,4], eval_ids=[3]):
  for v in train_ids:
      datasets.append(torch.load(f'pts/{v}_train.pt'))
  train_d = torch.utils.data.ConcatDataset(datasets)

  datasets = []
  for v in eval_ids:
      datasets.append(torch.load(f'pts/{v}_eval.pt'))
  eval_d = torch.utils.data.ConcatDataset(datasets)
  return train_d, eval_d