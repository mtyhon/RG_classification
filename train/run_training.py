import os, torch, argparse, sys
sys.path.append("..")
import torch.optim as optim
import torch.utils.data as utils
import pandas as pd
from model import *
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from generator import *
from sklearn.metrics import precision_score, recall_score

parser = argparse.ArgumentParser()

parser.add_argument("--root", type=str,required=True)
parser.add_argument("--k_folds", type=int, default=1)
parser.add_argument("--weight_llrgb", type=bool, default=False)
parser.add_argument("--llrgb_factor", type=float, default=0.5)
parser.add_argument("--holdout_frac", type=float, default=0.15)
parser.add_argument("--obs_len", type=float, default=82)
parser.add_argument("--init_lr", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=250)

config = parser.parse_args()


def train(model, model_optimizer, input_image, input_label, input_dnu, batch_idx, train_dataloader, config): 

    model_optimizer.zero_grad()

    # Combined forward pass
    outputs, kl = model.probforward(input_image, input_dnu)  # Bayesian

 
    # Calculate loss and backpropagate
    loss = elbo(outputs, input_label, kl, get_beta(math.ceil(len(train_dataloader) / 32), beta_type="Blundell", batch_idx=batch_idx))  # 32 is batch size
   
    if config.weight_llrgb: # upweight for Kepler (5), but downweight for K2 (0.25)?
        if loss[input_dnu.squeeze(1) > 9].size()[0] < 1:
            loss = torch.mean(loss[input_dnu.squeeze(1) <= 9])
        elif loss[input_dnu.squeeze(1) <= 9].size()[0] < 1:
            loss = config.llrgb_factor*torch.mean(loss[input_dnu.squeeze(1) > 9])
        else:
            loss = torch.mean(loss[input_dnu.squeeze(1) <= 9]) + config.llrgb_factor*torch.mean(loss[input_dnu.squeeze(1) > 9])
    else:
        loss = torch.mean(loss)
    loss.backward()

    # Update parameters
    model_optimizer.step()
    pred = torch.max(outputs, dim=1)[1]
    correct = torch.sum(pred.eq(input_label)).item()
    total = input_label.numel()
    acc = 100. * correct / total
    return loss.item(), acc, pred


def validate(model, val_dataloader, config):
    model.eval()  # set to evaluate mode
    val_loss = 0
    val_batch_acc = 0
    val_batches = 0
    val_pred = []
    val_truth = []
    val_prob = []
    val_kic_array = []
    val_dnuz = []
    val_logits = []
    for batch_idy, val_data in enumerate(val_dataloader, 0):  # indices,scaled_indices, numax, teff, fe_h, age, tams_age

        val_image = val_data[0].cuda().float()
        val_label = val_data[1].cuda().long().squeeze(1)
        val_dnu_var = val_data[2].cuda().float()
        val_flag = val_data[3].cuda().float()
        val_kic = val_data[4].cuda().float()

        val_image = val_image[val_flag != 1]
        val_label = val_label[val_flag != 1]
        val_dnu_var = val_dnu_var[val_flag != 1]
        val_kic = val_kic[val_flag != 1]

        if len(val_label) < 1:
            continue


        with torch.no_grad():

            outputs, kl = model.probforward(val_image, val_dnu_var) # Bayesian
            val_batch_loss = elbo(outputs, val_label, kl,
                                  get_beta(math.ceil(len(val_dataloader)) / 32, beta_type="Blundell", batch_idx=batch_idy))
          

            if config.weight_llrgb: # upweight for Kepler (5), but downweight for K2 (0.25)?
                if val_batch_loss[val_dnu_var.squeeze(1) > 9].size()[0] < 1:
                    val_batch_loss = torch.mean(loss[val_dnu_var.squeeze(1) <= 9])
                elif val_batch_loss[val_dnu_var.squeeze(1) <= 9].size()[0] < 1:
                    val_batch_loss = config.llrgb_factor*torch.mean(loss[val_dnu_var.squeeze(1) > 9])
                else:
                    val_batch_loss = torch.mean(val_batch_loss[val_dnu_var.squeeze(1) <= 9]) + config.llrgb_factor*torch.mean(loss[val_dnu_var.squeeze(1) > 9])
            else:
                val_batch_loss = torch.mean(val_batch_loss)

            val_logits.append(outputs.data.cpu().numpy())

        pred = torch.max(outputs, dim=1)[1]
        correct = torch.sum(pred.eq(val_label)).item()
        total = val_label.numel()
        val_loss += val_batch_loss.mean().item()
        val_batch_acc += 100. * correct / total
        val_batches += 1
        val_pred.append(pred.data.cpu().numpy())
        val_dnuz.append(val_dnu_var.data.cpu().numpy())
        val_truth.append(val_label.data.cpu().numpy())
        val_prob.append(F.softmax(outputs, dim=1).data.cpu().numpy())
        val_kic_array.append(val_kic.data.cpu().numpy())
    val_kic_array = np.concatenate(val_kic_array, 0)
    val_dnuz = np.concatenate(val_dnuz, 0)
    val_logits = np.concatenate(val_logits, 0)

    return (val_loss / val_batches), (val_batch_acc / val_batches), np.concatenate(val_pred, axis=0), np.concatenate(
        val_truth, axis=0), np.concatenate(val_prob, axis=0), val_kic_array, val_logits


def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = Bayes_Classifier()
   

    model.to(device)
    
    folder_kic = []
    root_folder = config.root
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filex in filenames:
            folder_kic.append(int(filex.split('-')[0]))

    folder_kic = np.unique(folder_kic)

    train_kics, val_kics = train_test_split(folder_kic, test_size=config.holdout_frac, random_state=137)
    print('Number of Train Stars: ', len(train_kics))
    print('Number of Validation Stars: ', len(val_kics))
    train_gen = npz_generator(root=root_folder, select_kic=train_kics, obs_len=356*4)
    train_dataloader = utils.DataLoader(train_gen, shuffle=True, batch_size=32)

    val_gen = npz_generator(root=root_folder, select_kic=val_kics, obs_len=356*4)
    val_dataloader = utils.DataLoader(val_gen, shuffle=False, batch_size=32)

    jie_data = pd.read_csv('JieData_Full2018.txt', delimiter='|', header=0)
    jie_kic = jie_data['KICID'].values
    jie_dnu = jie_data['dnu'].values

    label_data = pd.read_csv('Elsworth_Jie_2019_ID_Label.dat', header=0, delim_whitespace=True)
    label_kic = label_data['KIC'].values
    label_truth = label_data['Label'].values
    train_label = np.array([label_truth[np.where(label_kic == kicz)][0] for kicz in train_kics])
    val_label = np.array([label_truth[np.where(label_kic == kicz)][0] for kicz in val_kics])
    train_dnu = np.array([jie_dnu[np.where(jie_kic == kicz)][0] for kicz in train_kics])
    val_dnu = np.array([jie_dnu[np.where(jie_kic == kicz)][0] for kicz in val_kics])

    learning_rate = config.init_lr
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(model_optimizer, mode='min', factor=0.5, patience=20, verbose=True, min_lr=1E-6)

    n_epochs = config.num_epochs
    best_loss = 1.e9
    model_checkpoint = True
    print(model_checkpoint)
    for epoch in range(1, n_epochs + 1):
        print('---------------------')
        print('Epoch: ', epoch)
        train_loss = 0
        train_batches = 0
        acc_cum = 0
        train_kic_array = []
        dnu_array = []
        pred_array = []
        model.train()  # set to training mode

        for batch_idx, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), unit='batches'):

            image = data[0].cuda().float()
            label = data[1].cuda().long().squeeze(1)
            dnu = data[2].cuda().float()
            flag = data[3].cuda().float()
            train_kic = data[4].cuda().float()

            image = image[flag != 1] # flag 1 is bad stuff
            label = label[flag != 1]
            dnu = dnu[flag != 1]
            train_kic = train_kic[flag != 1]

            if len(label) < 2:
                print('Insufficient Batch!')
                continue

            loss, acc, predz = train(model, model_optimizer, image, label, dnu, batch_idx, train_dataloader, config)
            train_loss += loss  # Summing losses across all batches, so if you want the mean for EACH sample, divide by number of batches
            train_batches += 1
            acc_cum += acc
            train_kic_array.append(train_kic.data.cpu().numpy())
            pred_array.append(predz.data.cpu().numpy())
            dnu_array.append(dnu.data.cpu().numpy())
        train_loss = train_loss / train_batches
        train_acc = acc_cum / train_batches
        train_kic_array = np.concatenate(train_kic_array, 0)
        pred_array = np.concatenate(pred_array, axis = 0)
        dnu_array = np.concatenate(dnu_array, axis = 0)
      
        val_loss, val_acc, val_pred, val_truth, val_prob, val_kic, val_logits = validate(model, val_dataloader, config)
        scheduler.step(train_loss)  # reduce LR on loss plateau

        print('\n\nTrain Loss: ', train_loss)
        print('Train Acc: ', train_acc)

        val_loss = val_loss
        val_acc = val_acc
        print('Val Loss: ', val_loss)
        print('Val Acc: ', val_acc)

        for param_group in model_optimizer.param_groups:
            print('Current Learning Rate: ', param_group['lr'])

        if model_checkpoint:
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)

            if is_best:
                filepath = 'checkpoint/Epoch%d-ACC:%.2f-Loss:%.3f' % (
                    epoch,val_acc, val_loss)
                print('Model saved to %s' % filepath)
                torch.save(model.state_dict(), filepath)
            else:
                print('No improvement over the best of %.4f' % best_loss)

def train_model_k_fold(config):

    folder_kic = []
    root_folder = config.root
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filex in filenames:
            folder_kic.append(int(filex.split('-')[0]))

    train_kics = np.copy(folder_kic)
    folder_kic = np.unique(folder_kic)

    label_data = pd.read_csv('Elsworth_Jie_2019_ID_Label.dat', header=0, delim_whitespace=True)
    label_kic = label_data['KIC'].values
    label_truth = label_data['Label'].values
    train_label = np.array([label_truth[np.where(label_kic == kicz)][0] for kicz in train_kics])

    cv = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=137)
    prob_over_folds = []
    truth_over_folds = []
    acc_over_folds = []
    rgb_precision_over_folds = []
    heb_precision_over_folds = []
    rgb_recall_over_folds = []
    heb_recall_over_folds = []
    fold_count = 0

    for train_idx, valid_idx in cv.split(train_kics, train_label):
        fold_count += 1
        print('Validation Fold: ', fold_count)
      
        model = Bayes_Classifier()
        model.cuda()
        torch.backends.cudnn.benchmark = True

        train_gen = npz_generator(root=root_folder, select_kic=train_kics[train_idx], obs_len=config.obs_len)
        train_dataloader = utils.DataLoader(train_gen, shuffle=True, batch_size=32)

        val_gen = npz_generator(root=root_folder, select_kic=train_kics[valid_idx], obs_len=config.obs_len)
        val_dataloader = utils.DataLoader(val_gen, shuffle=False, batch_size=32)

        learning_rate = config.init_lr
        model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(model_optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1E-6)

        n_epochs = config.num_epochs
        for epoch in range(1, n_epochs + 1):
            print('---------------------')
            print('Epoch: ', epoch)
            print('Validation Fold: ', fold_count)
            train_loss = 0
            train_batches = 0
            acc_cum = 0

            model.train()  # set to training mode

            for batch_idx, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), unit='batches'):

                image = data[0].cuda().float()
                label = data[1].cuda().long().squeeze(1)
                dnu = data[2].cuda().float()
                flag = data[3].cuda().float()

                image = image[flag != 1]
                label = label[flag != 1]
                dnu = dnu[flag != 1]

                if len(label) < 2:
                    print('Insufficient Batch!')
                    continue

                loss, acc, predz = train(model, model_optimizer, image, label, dnu, batch_idx, train_dataloader, config)
                train_loss += loss  # Summing losses across all batches, so if you want the mean for EACH sample, divide by number of batches
                train_batches += 1
                acc_cum += acc

            train_loss = train_loss / train_batches
            train_acc = acc_cum / train_batches

            val_loss, val_acc, val_pred, val_truth, val_prob, val_kic, val_logits = validate(model, val_dataloader, config)
            scheduler.step(train_loss)  # reduce LR on loss plateau

            print('\n\nTrain Loss: ', train_loss)
            print('Train Acc: ', train_acc)

            print('Val Loss: ', val_loss)
            print('Val Acc: ', val_acc)
               
            for param_group in model_optimizer.param_groups:
                print('Current Learning Rate: ', param_group['lr'])

        # After training, one last validation pass after training
        _, final_acc, final_pred, final_truth, final_prob, final_kic, final_logits = validate(model, val_dataloader, config)
        fold_precision_heb = precision_score(y_true=final_truth, y_pred=final_pred, pos_label=1)
        fold_precision_rgb = precision_score(y_true=final_truth, y_pred=final_pred, pos_label=0)
        fold_recall_heb = recall_score(y_true=final_truth, y_pred=final_pred, pos_label=1)
        fold_recall_rgb = recall_score(y_true=final_truth, y_pred=final_pred, pos_label=0)
       
        print('Accuracy for this fold: ', final_acc)
        print('HeB Precision for this fold: ', fold_precision_heb)
        print('HeB Recall for this fold: ', fold_recall_heb)
        print('RGB Precision for this fold: ', fold_precision_rgb)
        print('RGB Recall for this fold: ', fold_recall_rgb)

        prob_over_folds.append(final_prob)
        truth_over_folds.append(final_truth)
        acc_over_folds.append(final_acc)
        print('Accumulated Accuracy Thus Far: ', acc_over_folds)
        rgb_precision_over_folds.append(fold_precision_rgb)
        rgb_recall_over_folds.append(fold_recall_rgb)
        heb_precision_over_folds.append(fold_precision_heb)
        heb_recall_over_folds.append(fold_recall_heb)

    prob_over_folds = np.concatenate(prob_over_folds, axis=0)
    truth_over_folds = np.concatenate(truth_over_folds, axis=0)

    print(prob_over_folds.shape)
    print(truth_over_folds.shape)

    print('Validation Accuracy: %.3f +/- %.3f' % (np.mean(acc_over_folds), np.std(acc_over_folds)))
    print('Validation HeB Precision: %.3f +/- %.3f' % (
    np.mean(heb_precision_over_folds), np.std(heb_precision_over_folds)))
    print('Validation HeB Recall: %.3f +/- %.3f' % (np.mean(heb_recall_over_folds), np.std(heb_recall_over_folds)))
    print('Validation RGB Precision: %.3f +/- %.3f' % (
    np.mean(rgb_precision_over_folds), np.std(rgb_precision_over_folds)))
    print('Validation RGB Recall: %.3f +/- %.3f' % (np.mean(rgb_recall_over_folds), np.std(rgb_recall_over_folds)))



if __name__ == '__main__':
    if config.k_folds == 1:
        train_model(config)
    else:
        train_model_k_fold(config)




