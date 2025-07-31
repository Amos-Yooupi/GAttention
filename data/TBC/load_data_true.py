import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import torch
import time

Span = 5
Train = 7
DataDir = rf"E:\Dataset\0723\Car{Train}_Bri{Span}\data"
BeamNameList = ["AccXBriBeam", "AccYBriBeam", "AccZBriBeam", "DisXBriBeam", "DisYBriBeam", "DisZBriBeam"]
CarNameList = ['AccYcar', 'AccZcar', 'DisYcar', 'DisZcar']
save_path = fr"Span_{Span}_Car_{Train}"
PosList = ["Ori", "Mid", "End"]


def create_dir(file_path):
    if os.path.exists(file_path):
        pass
    else:
        os.makedirs(file_path)

def load_train(file_path):
    print("Load Train Data")
    train_save_path = os.path.join(save_path, "Train.pt")
    train_data = loadmat(file_path)
    AccYcar = torch.from_numpy(train_data['AccYcar'])
    AccZcar = torch.from_numpy(train_data['AccZcar'])
    DisYcar = torch.from_numpy(train_data['DisYcar'])
    DisZcar = torch.from_numpy(train_data['DisZcar'])
    train_data = torch.stack([AccZcar, AccYcar, DisZcar, DisYcar], dim=-1)
    print(train_data.shape)
    torch.save(train_data, train_save_path)


def load_bridge(file_path):
    print("Load Bridge Data")
    bridge_save_path = os.path.join(save_path, "Span.pt")
    bridge_data = loadmat(file_path)
    AccXBriBeamMid = torch.from_numpy(bridge_data['AccXBriBeamMid'])
    AccYBriBeamMid = torch.from_numpy(bridge_data['AccYBriBeamMid'])
    AccZBriBeamMid = torch.from_numpy(bridge_data['AccZBriBeamMid'])
    DisXBriBeamMid = torch.from_numpy(bridge_data['DisXBriBeamMid'])
    DisYBriBeamMid = torch.from_numpy(bridge_data['DisYBriBeamMid'])
    DisZBriBeamMid = torch.from_numpy(bridge_data['DisZBriBeamMid'])
    bridge_data = torch.stack([AccXBriBeamMid, AccYBriBeamMid, AccZBriBeamMid, DisXBriBeamMid, DisYBriBeamMid, DisZBriBeamMid], dim=-1)
    print(bridge_data.shape)
    torch.save(bridge_data, bridge_save_path)


def load_earth(file_path):
    print("Load Earth Data")
    earth_save_path = os.path.join(save_path, "Earthquake.pt")
    earth_data = loadmat(file_path)
    earth_data = torch.from_numpy(earth_data['EqAccOfWavTraTRec'])
    length = earth_data.shape[1]
    earth_data = earth_data
    print(earth_data.shape)
    torch.save(earth_data, earth_save_path)


def load_file():
    create_dir(save_path)
    load_train(os.path.join(DataDir, f"ResOfTrain.mat"))
    load_bridge(os.path.join(DataDir, f"ResOfBriBeamMid.mat"))
    load_earth(os.path.join(DataDir, f"EqAcc.mat"))
    print("Load successfully")


if __name__ == '__main__':
    """
       ResOfTrain.mat
        ResOfBriBeamMid.mat
        EqAcc.mat
    """
    load_file()



