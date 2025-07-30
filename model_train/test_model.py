import torch
import matplotlib.pyplot as plt
from calculate_loss import fourier_loss


def test(model, data_iter, config, criterion, device):
    model.eval()
    torch.cuda.empty_cache()  # 在训练/推理的循环间隙或结束后调用
    test_dataset = data_iter.generate_dataset("test", config.batch_size,
                                              config.in_len, config.out_len, shuffle=False, load_mode="simple")

    adj = data_iter.adj.to(device, dtype=torch.float)

    predict_list = []
    true_list = []
    with torch.no_grad():
        for iter_i, (x, label) in enumerate(test_dataset):
            x = [item.to(device, dtype=torch.float) for item in x]
            label = label.to(device, dtype=torch.float)
            input_adj = adj.repeat(x[0].shape[0], 1, 1)
            model.eval()
            # 取出列车数据,需要预测这个
            iter_predict = model(*x, adj=input_adj)  # (B, N, L, D)
            if data_iter.name == "TBC":
                iter_predict = iter_predict[:, :data_iter.num_task]
            true_list.append(label.detach().cpu())
            predict_list.append(iter_predict.detach().cpu())

    try:
        true = torch.cat(true_list, dim=0).transpose(1, 2)  # (B, N, L, D) -> (B, L, N, D)
        predict = torch.cat(predict_list, dim=0).transpose(1, 2)  # (B, N, L, D) -> (B, L, N, D)
        _, _, N, D = true.size()
        true = true.contiguous().view(-1, N, D)
        predict = predict.contiguous().view(-1, N, D)
    except:
        true = torch.cat(true_list, dim=0)
        predict = torch.cat(predict_list, dim=0)
        B, L, D, H, W = true.size()
        true = true.contiguous().view(-1, D, H, W)
        predict = predict.contiguous().view(-1, D, H, W)

    # true = data_iter.re_norm_data(true, "test")
    # predict = data_iter.re_norm_data(predict, "test")
    print("Freq Loss:", fourier_loss(true, predict))
    loss = criterion(predict, true)
    mae_loss = torch.abs(predict - true).mean()
    print(f"Test Loss: {loss.item():.6f},   MAE Loss: {mae_loss.item():.6f}")
    return true, predict

