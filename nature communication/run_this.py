import torch
from utils import *
from model import BasicModel
from model.BackBone import BackBone
from model_train import *

# 加载配置文件
config_path = "config.json"
config = Config(config_path)

# 加载数据集`
data_iter = init_data_iter(config)

# 初始化模型0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = BackBone(config, "GAttention")
model = BasicModel(config, backbone).to(device)
# 初始化优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
criterion = torch.nn.MSELoss()

# 是否加载模型
if config.is_load:
    load_model(model, optimizer, config.model_path)
else:
    print("开始重新训练模型！")

# 训练模型
is_train = False
if is_train:
    train_list, val_list = train(model, data_iter, config, optimizer, criterion,
                                 device, config.epoch, config.model_path, val_freq=1, is_continue=False)
    show_loss(train_list, val_list)
    save_loss(train_list, fr"loss\train_loss.txt")
    save_loss(val_list, fr"loss\val_loss.txt")

# show_loss_compare(r"C:\Users\Administrator\Desktop\model\no_region-4",
#                   r"C:\Users\Administrator\Desktop\model\region-4")
# 测试模型
for i in range(len(data_iter)):
    choose_dataiter = data_iter.choose_dataiter()
    true, predict = test(model, choose_dataiter, config, criterion, device)

    show_result(true, predict)


