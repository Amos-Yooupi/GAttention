import os.path
import torch


def load_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, model_path):
    if os.path.exists(model_path):
        cheek_point = torch.load(model_path)
        try:
            model.load_state_dict(cheek_point['model_state_dict'])
            optimizer.load_state_dict(cheek_point['optimizer_state_dict'])
            print(f"模型加载成功！--- {cheek_point['epoch']} 保存的模型")
        except:
            print("模型加载失败！, 请检查模型结构是否一致！")
    else:
        print("模型加载失败！, 路径不存在！")
