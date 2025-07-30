import os.path
import time
import torch
from save_model import save_model


def train(model, data_iter, config, optimizer,
          criterion, device, epoch, model_save_path, val_freq,
          is_continue=True):

    final_model_path = os.path.join(config.model_path, config.model_choose + "_final.pth")
    val_model_path = os.path.join(config.model_path, config.model_choose + "_best.pth")

    # 记录训练过程
    val_save_file_path = os.path.join(os.path.dirname(model_save_path), "val.txt")
    if is_continue:
        if os.path.exists(val_save_file_path):
            with open(val_save_file_path, "r") as f:
                val_list = [float(f.read())]
        else:
            val_list = []
    else:
        val_list = []
    train_list = []
    start_time = time.time()
    previous_time = time.time()

    for i in range(epoch):
        # 选择对应的dataiter
        choose_data_iter = data_iter.choose_dataiter()
        train_dataset = choose_data_iter.generate_dataset("train", config.batch_size,
                                                   config.in_len, config.out_len, shuffle=True, load_mode="complex")
        val_dataset = choose_data_iter.generate_dataset("val", config.batch_size, config.in_len,
                                                 config.out_len, shuffle=True, load_mode="simple")
        # 初始化邻接矩阵
        adj = choose_data_iter.adj.to(device)

        torch.cuda.empty_cache()
        epoch_loss_train = []
        for iter_i, (x, label) in enumerate(train_dataset):
            x = [(item + 0e-2*torch.randn_like(item)).to(device, dtype=torch.float) for item in x]
            label = label.to(device, dtype=torch.float) # (B, N, D, L) -> (B, N, L, D)
            input_adj = adj.repeat(x[0].shape[0], 1, 1)
            model.train()
            predict = model(*x, adj=input_adj)  # (B, N, L, D)
            if choose_data_iter.name == "TBC":
                # 取出对应节点
                predict = predict[:, :choose_data_iter.num_task]
            loss = criterion(predict, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            per_tier_time = time.time() - previous_time
            previous_time = time.time()
            cost_time = time.time() - start_time
            if iter_i % 1 == 0:
                print(f"Epoch:{i}---Iter:{iter_i}---Train Loss:{round(loss.item(), 6)}"
                      f"---Cost Time: {round(cost_time, 3)}s---"
                      f"Per iter Time: {round(per_tier_time, 3)}s")
            epoch_loss_train.append(loss.detach().cpu().item())
        train_list.append(sum(epoch_loss_train) / len(epoch_loss_train))
        # 验证集

        count_limit = 2
        if (i+1) % val_freq == 0:
            model.eval()
            with torch.no_grad():
                epoch_loss_val = []
                count = 0
                for (x, label) in val_dataset:
                    x = [item.to(device, dtype=torch.float) for item in x]
                    label = label.to(device, dtype=torch.float)
                    input_adj = adj.repeat(x[0].shape[0], 1, 1)
                    model.train()
                    predict = model(*x, adj=input_adj)
                    if choose_data_iter.name == "TBC":
                        # 取出对应节点
                        predict = predict[:, :choose_data_iter.num_task]
                    loss = criterion(predict, label)
                    epoch_loss_val.append(loss.detach().cpu().item())
                val_list.append(sum(epoch_loss_val) / len(epoch_loss_val))
                print(f"Epoch:{i} Train Loss:{round(train_list[-1], 6)} Val Loss:{round(val_list[-1], 6)}")
                # 保存最优模型
                if val_list[-1] <= min(val_list):
                    save_model(i, model, optimizer, val_model_path)
                    with open(val_save_file_path, "w") as f:
                        f.write(str(val_list[-1]))
                    print("Model saved!")
                count += 1
                if count == count_limit:
                    break
        if i % 200 == 0:
            save_model(epoch, model, optimizer, final_model_path)
    save_model(epoch, model, optimizer, final_model_path)
    return train_list, val_list
