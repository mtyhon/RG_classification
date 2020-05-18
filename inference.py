from model import *
from utils.image_processing import *
from utils.plot import *
import torch.nn.functional as F
import os, torch, argparse
import pandas as pd

package_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()

parser.add_argument("--classifier", type=str, default=package_dir +'/saved_models/4yr_Bayes.torchmodel')
parser.add_argument("--psd_file", type=str, required=True)
parser.add_argument("--numax", type=float, required=True)
parser.add_argument("--dnu", type=float, required=True)
parser.add_argument("--mc_samples", type=int, default=10)
parser.add_argument("--plot_spectrum", type=bool, default=True)

config = parser.parse_args()

def infer(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    saved_model_dict = config.classifier

    trained_model = Bayes_Classifier()
    trained_model.load_state_dict(torch.load(saved_model_dict))
    trained_model.to(device)
    trained_model.eval()

    if config.psd_file.endswith('csv'):
        df = pd.read_csv(config.psd_file, header=None)
    else:
        df = pd.read_table(config.psd_file, delim_whitespace=True, header=None)
    freq = df.iloc[:,0].values
    power = df.iloc[:,1].values

    X = create_folded_spectrum(freq, power, config.numax, config.dnu)
    X = torch.Tensor([X]).to(device)
    dnu = torch.Tensor([config.dnu]).to(device)

    pred_grid = np.empty((config.mc_samples, len(dnu), 2))
    with torch.no_grad():
        for i in range(config.mc_samples):
            outputs, kl = trained_model.probforward(X, dnu)
            pred_grid[i, :] = F.softmax(outputs, dim=1).data.cpu().numpy()
    pred_mean = np.mean(pred_grid, axis=0)
    epistemic = np.mean(pred_grid ** 2, axis=0) - np.mean(pred_grid, axis=0) ** 2
    aleatoric = np.mean(pred_grid * (1 - pred_grid), axis=0)
    pred = np.argmax(pred_mean, axis=1)
    prob = pred_mean[:,1]
    epistemic_sigma = epistemic[np.arange(len(pred)), np.argmax(pred_mean, axis=1)]
    aleatoric_sigma = aleatoric[np.arange(len(pred)), np.argmax(pred_mean, axis=1)]
    full_sigma = np.sqrt(epistemic_sigma + aleatoric_sigma)

    print('File: %s' %config.psd_file)
    print('Ev. State (0=RGB,1=HeB) : %d' %pred)
    print('Score (0=Very RGB-like, 1=Very HeB-like) : %s' %prob[0])
    print('Epistemic Uncertainty: %s' %epistemic_sigma[0])
    print('Aleatoric Uncertainty: %s' %aleatoric_sigma[0])
    print('Full Sigma: %s' %full_sigma[0])

    if config.plot_spectrum:
        plot_folded_spectrum(X.data.cpu().numpy(), config.psd_file, config.dnu, config.numax, pred, package_dir, freq, power)

if __name__ == '__main__':
    infer(config)
