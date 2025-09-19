def save_loss(loss_list, path):
    with open(path, 'w') as f:
        for loss in loss_list:
            f.write(str(loss) + '\n')
    print('Loss saved to', path)