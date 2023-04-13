import torch
import numpy as np
import os


def pass_through_network(model, loader, X_path=None, y_path=None, device='cuda'):
    if X_path is not None and os.path.isfile(X_path):
        features = np.load(X_path)
        targets = np.load(y_path)
    else:
        features = []
        targets = []
        for data, target in loader:
            features.append(pass_batch_through_network(model, data, device))
            targets.append(target)

        features = np.concatenate(features)
        targets = np.concatenate(targets)
        if X_path is None:
            np.save('tmp/test_x.npy', features)
            np.save('tmp/test_x.npy', targets)
        else:
            np.save(X_path, features)
            np.save(y_path, targets)
    return features, targets


def pass_batch_through_network(model, batch, device='cuda'):
    with torch.no_grad():
        ans = []
        for data in batch:
            data_in = data.to(device)
            output = model(data_in)
            ans.append(output.cpu().numpy().reshape(-1))
        return np.concatenate(ans)
