from utils import *
from model import BasicModel
from model.BackBone import BackBone
from model_train import *


# 加载配置文件
config_path = "config.json"
config = Config(config_path)
model_choose = "RGNN"
config.model_choose = model_choose


# 加载数据集
data_iter = init_data_iter(config)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = BackBone(config, model_choose)
model = BasicModel(config, backbone).to(device)
# 初始化优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.lr)

final_model_path = os.path.join(get_model_save_path(model, config), model_choose + "_final.pth")
best_model_path = os.path.join(get_model_save_path(model, config), model_choose + "_best.pth")


criterion = torch.nn.MSELoss()


is_load = False
# 是否加载模型
if is_load:
    load_model(model, optimizer, best_model_path)
else:
    print("开始重新训练模型!")

# 训练模型
is_train = True
if is_train:
    train(model, data_iter, config, optimizer, criterion,
                                 device, config.epoch, val_freq=1, is_continue=False)


# 测试模型
for i in range(len(data_iter)):
    print("best model on val")
    choose_dataiter = data_iter.choose_dataiter()
    # print(choose_dataiter)
    load_model(model, optimizer, best_model_path)
    true, predict = test(model, choose_dataiter, config, criterion, device)
    show_result(true, predict)

    if config.embedding_choose == "Weather":
        true = data_iter.re_norm_data(true, "test")
        predict = data_iter.re_norm_data(predict, "test")
        print(torch.abs(true-predict).mean() / data_iter.weather_mask.sum() * data_iter.weather_mask.numel())
        show_weather_result(true, predict)




