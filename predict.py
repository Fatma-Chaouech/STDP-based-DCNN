import torch
import numpy as np
import os


def pass_through_network(model, loader, device='cuda'):
    X_path = 'tmp/test_x.npy'
    y_path = 'tmp/test_y.npy'
    if os.path.isfile(X_path):
        features = np.load(X_path)
        targets = np.load(y_path)
    else:
        os.makedirs('tmp', exist_ok=True)
        features = []
        targets = []
        for data, target in loader:
            features.append(pass_batch_through_network(model, data, device))
            targets.append(target)

        features = np.concatenate(features)
        targets = np.concatenate(targets)
        np.save(X_path, features)
        np.save(y_path, targets)
    return features, targets


def pass_batch_through_network(model, batch, device='cuda'):
    with torch.no_grad():
        ans = []
        for data in batch:
            data_in = data.to(device)
            output = model(data_in)
            ans.append(output.reshape(-1).cpu().numpy())
        return np.array(ans)
    

def eval(X, y, predictions):
    non_silence_mask = np.count_nonzero(X, axis=1) > 0
    correct_mask = predictions == y
    correct_non_silence = np.logical_and(correct_mask, non_silence_mask)
    correct = np.count_nonzero(correct_non_silence)
    silence = np.count_nonzero(~non_silence_mask)
    return (correct / len(X), (len(X) - (correct + silence)) / len(X), silence / len(X))
