import sys
from multiprocessing import freeze_support
import torch
from dataset import CancerDataset
from TMO_Netplus_model import TMO_Netplus, DownStream_predictor, dfs_freeze, un_dfs_freeze
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from loss_function import cox_loss
from lifelines.utils import concordance_index
import pickle
import torch.multiprocessing as mp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, pairwise_distances
from LogME import LogME
import os


logme = LogME(regression=False)

###66, 100, 166, 188, and 200
def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(66)


base_path = 'D:\TMO-Netplus'

omics_files = [
    '../data/TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_Expression_230109_modified.csv',
    '../data/TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_Methylation_230109_modified.csv',
    '../data/TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_Mutation_230109_modified.csv',
    '../data/TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_CNA_230109_modified.csv'
]

clinical_file = '../data/TCGA_PanCancer_Data_cleaned/cleaned_clinical_info.csv'

train_index_path = '../data/TCGA_PanCancer_Data_cleaned/train_data.csv'
test_index_path = '../data/TCGA_PanCancer_Data_cleaned/test_data.csv'


# omics_data_type = ['gaussian', 'gaussian']
omics_data_type = ['gaussian', 'gaussian', 'gaussian', 'gaussian']

omics = {'gex': 0, 'methy': 1, 'mut': 2, 'cna': 3}
incomplete_omics = {'gex': 0, 'methy': 1}

def train_pretrain(train_dataloader, model, epoch, cancer, optimizer, dsc_optimizer, fold, pretrain_omics):
    model.train()
    print(f'-----start epoch {epoch} base model straining-----')
    total_loss = 0
    total_self_elbo = 0
    total_cross_elbo = 0
    total_cross_infer_loss = 0
    total_dsc_loss = 0
    total_ad_loss = 0
    Loss = []
    pancancer_embedding = torch.Tensor([]).cuda()
    all_label = torch.Tensor([]).cuda()

    with tqdm(train_dataloader, unit='batch') as tepoch:
        for batch, data in enumerate(tepoch):
            tepoch.set_description(f" Epoch {epoch}: ")
            os_event, os_time, omics_data, cancer_label = data
            cancer_label = cancer_label.cuda()
            cancer_label = cancer_label.squeeze()
            all_label = torch.concat((all_label, cancer_label), dim=0)

            input_x = [omics_data[key].cuda() for key in omics_data.keys()]
            for key in omics_data.keys():
                omic = omics_data[key]
                omic = omic.cuda()
                # print(omic)
                input_x.append(omic)
            un_dfs_freeze(model.discriminator)
            un_dfs_freeze(model.infer_discriminator)
            cross_infer_dsc_loss, dsc_loss = model.compute_dsc_loss(input_x, os_event.size(0), pretrain_omics)
            # recon_omics = model.cross_modal_generation(input_x[:2], incomplete_omics)
            ad_loss = cross_infer_dsc_loss + dsc_loss
            total_ad_loss += dsc_loss.item()

            dsc_optimizer.zero_grad()
            ad_loss.backward(retain_graph=True)
            dsc_optimizer.step()

            un_dfs_freeze(model.discriminator)
            un_dfs_freeze(model.infer_discriminator)
#             dfs_freeze(model.discriminator)
#             dfs_freeze(model.infer_discriminator)
            loss, self_elbo, cross_elbo, cross_infer_loss, dsc_loss = model.compute_generate_loss(input_x,
                                                                                                  os_event.size(0),
                                                                                                  pretrain_omics)

            total_self_elbo += self_elbo.item()
            total_cross_elbo += cross_elbo.item()
            total_cross_infer_loss += cross_infer_loss.item()
            multi_embedding = model.get_embedding(input_x, os_event.size(0), pretrain_omics)
            pancancer_embedding = torch.concat((pancancer_embedding, multi_embedding), dim=0)

            contrastive_loss = model.contrastive_loss(multi_embedding, cancer_label)
            loss += contrastive_loss
            classifier_loss =model.classifier_loss(multi_embedding, cancer_label)
            loss += classifier_loss.sum(0)

            # loss = ce_loss
            total_dsc_loss += dsc_loss.item()
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item(), self_elbo_loss=self_elbo.item(), cross_elbo_loss=cross_elbo.item(),
                               cross_infer_loss=cross_infer_loss.item(), dsc_loss=dsc_loss.item())

        print('total loss: ', total_loss / len(train_dataloader))
        Loss.append(total_loss / len(train_dataloader))
        print('self elbo loss: ', total_self_elbo / len(train_dataloader))
        Loss.append(total_self_elbo / len(train_dataloader))
        print('cross elbo loss: ', total_cross_elbo / len(train_dataloader))
        Loss.append(total_cross_elbo / len(train_dataloader))
        print('cross infer loss: ', total_cross_infer_loss / len(train_dataloader))
        Loss.append(total_cross_infer_loss / len(train_dataloader))
        print('ad loss', total_ad_loss / len(train_dataloader))
        print('dsc loss', total_dsc_loss / len(train_dataloader))
        Loss.append(total_dsc_loss / len(train_dataloader))
        # torch.save(pancancer_embedding,
        #            os.path.join(base_path,f'model/model_dict/TCGA_pancancer_multi_train_embedding_fold{fold}_epoch{epoch}.pt'))
        # torch.save(all_label,
        #            os.path.join(base_path,f'model/model_dict/TCGA_pancancer_train_fold{fold}_epoch{epoch}_all_label.pt'))
        pretrain_score = logme.fit(pancancer_embedding.detach().cpu().numpy(), all_label.cpu().numpy())
        print('pretrain score:', pretrain_score)

        return Loss, pretrain_score

#   第4个
def val_pretrain(test_dataloader, model, epoch, cancer, fold, pretrain_omics):
    model.eval()
    print(f'-----start epoch {epoch} base model val-----')
    total_loss = 0
    total_self_elbo = 0
    total_cross_elbo = 0
    total_cross_infer_loss = 0
    total_dsc_loss = 0
    total_cross_infer_dsc_loss = 0
    Loss = []
    pancancer_embedding = torch.Tensor([]).cuda()
    all_label = torch.Tensor([]).cuda()
    with torch.no_grad():
        with tqdm(test_dataloader, unit='batch') as tepoch:
            for batch, data in enumerate(tepoch):
                tepoch.set_description(f" Epoch {epoch}: ")
                os_event, os_time, omics_data, cancer_label  = data
                cancer_label = cancer_label.cuda()
                cancer_label = cancer_label.squeeze()
                all_label = torch.concat((all_label, cancer_label), dim=0)
                input_x = []
                for key in omics_data.keys():
                    omic = omics_data[key]
                    omic = omic.cuda()
                    input_x.append(omic)

                cross_infer_dsc_loss, dsc_loss = model.compute_dsc_loss(input_x, os_event.size(0), pretrain_omics)

                total_cross_infer_dsc_loss += cross_infer_dsc_loss.item()

                loss, self_elbo, cross_elbo, cross_infer_loss, dsc_loss = model.compute_generate_loss(input_x,
                                                                                                      os_event.size(0),
                                                                                                      pretrain_omics)
                multi_embedding = model.get_embedding(input_x, os_event.size(0), pretrain_omics)

                pancancer_embedding = torch.concat((pancancer_embedding, multi_embedding), dim=0)
                total_self_elbo += self_elbo.item()
                total_cross_elbo += cross_elbo.item()
                total_cross_infer_loss += cross_infer_loss.item()

                total_dsc_loss += dsc_loss.item()
                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item(), self_elbo_loss=self_elbo.item(), cross_elbo_loss=cross_elbo.item(),
                                   cross_infer_loss=cross_infer_loss.item(), dsc_loss=dsc_loss.item())

            print('test total loss: ', total_loss / len(test_dataloader))
            Loss.append(total_loss / len(test_dataloader))
            print('test self elbo loss: ', total_self_elbo / len(test_dataloader))
            Loss.append(total_self_elbo / len(test_dataloader))
            print('test cross elbo loss: ', total_cross_elbo / len(test_dataloader))
            Loss.append(total_cross_elbo / len(test_dataloader))
            print('test cross infer loss: ', total_cross_infer_loss / len(test_dataloader))
            Loss.append(total_cross_infer_loss / len(test_dataloader))
            print('test ad loss', total_cross_infer_dsc_loss / len(test_dataloader))
            print('test dsc loss', total_dsc_loss / len(test_dataloader))
            Loss.append(total_dsc_loss / len(test_dataloader))
            # torch.save(pancancer_embedding,
            #            os.path.join(base_path,f'model/model_dict/TCGA_pancancer_multi_test_embedding_fold{fold}_epoch{epoch}.pt'))
            # torch.save(all_label,
            #            os.path.join(base_path,f'model/model_dict/TCGA_pancancer_test_fold{fold}_epoch{epoch}_all_label.pt'))

            pretrain_score = logme.fit(pancancer_embedding.detach().cpu().numpy(), all_label.cpu().numpy())
            print('pretrain score:', pretrain_score)
    return Loss, pretrain_score

def TCGA_Dataset_pretrain(fold, epochs, device_id, cancer_types=None):
    train_dataset = CancerDataset(omics_files, ['gex', 'methy', 'mut', 'cna'], clinical_file, train_index_path,
                                  fold + 1)
    test_dataset = CancerDataset(omics_files, ['gex', 'methy', 'mut', 'cna'], clinical_file, test_index_path, fold + 1)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True
    )

    test_dataloader = DataLoader(test_dataset, batch_size=64)

    model = TMO_Netplus(modal_num=4,
                    modal_dim=[6016, 6617, 4539, 7460],
                    latent_dim=64,
                    encoder_hidden_dims=[2048, 512],
                    decoder_hidden_dims=[512, 2048],
                    omics_data_type=['gaussian', 'gaussian', 'gaussian', 'gaussian'],
                    kl_loss_weight=0.01
                    )

    torch.cuda.set_device(device_id)
    model.cuda()
    print('Number of samples in the training dataset：' + str(len(train_dataset)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    dsc_parameters = list(model.discriminator.parameters()) + list(model.infer_discriminator.parameters())
    dsc_optimizer = torch.optim.Adam(dsc_parameters, lr=0.0001)

    Loss_list_train = []
    Loss_list_test = []
    pretrain_score_list_train = []
    pretrain_score_list_test = []


    patience = 10
    best_pretrain_score = -float('inf')
    min_delta = 0.001
    counter = 0
    best_model_state = None
    best_epoch = 0
    best_optimizer_state = None

    for epoch in range(epochs):
        start_time = time.time()
        loss_train, pretrain_score_train = train_pretrain(train_dataloader, model, epoch, 'PanCancer', optimizer,
                                                          dsc_optimizer, fold, omics)
        current_score = pretrain_score_train
        scheduler.step(current_score)

        Loss_list_train.append(loss_train)
        pretrain_score_list_train.append(pretrain_score_train)
        loss_val, pretrain_score_val = val_pretrain(test_dataloader, model, epoch, 'PanCancer', fold, omics)
        Loss_list_test.append(loss_val)
        pretrain_score_list_test.append(pretrain_score_val)
        current_val_score = pretrain_score_val

        if current_val_score > best_pretrain_score + min_delta:
            best_pretrain_score = current_val_score
            counter = 0
            best_model_state = model.state_dict().copy()
            best_optimizer_state = {
                'optimizer': optimizer.state_dict().copy(),
                'dsc_optimizer': dsc_optimizer.state_dict().copy()
            }
            best_epoch = epoch
            print(f'New best model found at epoch {epoch} with score: {best_pretrain_score:.4f}')
        else:
            counter += 1
            print(
                f' No improvement for {counter} epochs. Best score: {best_pretrain_score:.4f}, Current: {current_val_score:.4f}')
            if counter >= patience:
                print(f' Early stopping triggered at epoch {epoch}')
                print(f'Best model was at epoch {best_epoch} with score {best_pretrain_score:.4f}')
                break

        print(f'fold{fold} time used: ', time.time() - start_time)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if best_optimizer_state is not None:
            optimizer.load_state_dict(best_optimizer_state['optimizer'])
            dsc_optimizer.load_state_dict(best_optimizer_state['dsc_optimizer'])
        print(f'Restored best model from epoch {best_epoch} with score: {best_pretrain_score:.4f}')
        final_loss, final_score = val_pretrain(test_dataloader, model, best_epoch, 'PanCancer', fold, omics)
        final_total_loss = final_loss[0] if isinstance(final_loss, list) and len(final_loss) > 0 else final_loss
        print(f'Final model performance - Loss: {final_total_loss:.4f}, Score: {final_score:.4f}')
    else:
        print('No best model found, using final epoch model')
    model_dict = model.state_dict()
    Loss_list_train = torch.Tensor(Loss_list_train)
    Loss_list_test = torch.Tensor(Loss_list_test)
    pretrain_score_list_test = pd.DataFrame(pretrain_score_list_test, columns=['pretrain_score'])
    pretrain_score_list_test.to_csv(os.path.join(base_path, f'model/pretrain_score_list_fold{fold}.csv'))

    # torch.save(Loss_list_test, os.path.join(base_path,f'model/model_dict/TCGA_pancancer_pretrain_test_loss_fold{fold}.pt'))
    # torch.save(Loss_list_train, os.path.join(base_path,f'model/model_dict/TCGA_pancancer_pretrain_train_loss_fold{fold}.pt'))
    torch.save(model_dict,
               os.path.join(base_path, f'model/model_dict/TCGA_pancancer_pretrain_model_fold{fold}_dim64.pt'))

def train_survival(train_dataloader, model, epoch, cancer, fold, optimizer, omics):
    model.train()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f'-----start {cancer} epoch {epoch} training-----')
    total_loss = 0
    train_risk_score = torch.Tensor([]).cuda()
    train_censors = torch.Tensor([]).cuda()
    train_event_times = torch.Tensor([]).cuda()
    with tqdm(train_dataloader, unit='batch') as tepoch:
        for batch, data in enumerate(tepoch):
            tepoch.set_description(f" Epoch {epoch}: ")
            os_event, os_time, omics_data, _ = data

            os_event = os_event.cuda()
            os_time = os_time.cuda()
            train_censors = torch.concat((train_censors, os_event))
            train_event_times = torch.concat((train_event_times, os_time))

            input_x = [omics_data[key].cuda() for key in omics_data.keys()]

            risk_score = model(input_x, os_event.size(0), omics)
            pretrain_loss, _, _, _, _ = model.cross_encoders.compute_generate_loss(input_x, os_event.size(0), omics)
            train_risk_score = torch.concat((train_risk_score, risk_score))
            CoxLoss = cox_loss(os_time, os_event, risk_score)
            # loss = CoxLoss + 0.05 * pretrain_loss
            alpha = min(1.0, epoch / 10)
            loss = CoxLoss + alpha * pretrain_loss
            ##loss = CoxLoss
            total_loss += CoxLoss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=CoxLoss.item())

        print('cox loss: ', total_loss / len(train_dataloader))
        train_c_index = concordance_index(train_event_times.detach().cpu().numpy(),
                                          -train_risk_score.detach().cpu().numpy(),
                                          train_censors.detach().cpu().numpy())

        print(f'{cancer} train survival c-index: ', train_c_index)


def cal_c_index(dataloader, model, best_c_index, best_c_indices, omics):
    model.eval()
    best_c_index_list = best_c_indices
    device = next(model.parameters()).device

    with torch.no_grad():
        risk_scores = {'all': []}
        censors = []
        event_times = []

        for i, data in enumerate(dataloader):
            os_event, os_time, omics_data, cancer_label= data
            input_x = [omics_data[key].to(device) for key in omics_data.keys()]
            os_event = os_event.to(device)
            os_time = os_time.to(device)

            survival_risk = model(input_x, os_event.size(0), omics)

            if not torch.isnan(survival_risk).any():
                risk_scores['all'].append(survival_risk.cpu())
                censors.append(os_event.cpu())
                event_times.append(os_time.cpu())
        risk_scores_np = -torch.cat(risk_scores['all']).numpy() if risk_scores['all'] else np.array([])
        event_times_np = torch.cat(event_times).numpy() if event_times else np.array([])
        censors_np = torch.cat(censors).numpy() if censors else np.array([])
        valid_mask = ~np.isnan(risk_scores_np) & ~np.isnan(event_times_np) & ~np.isnan(censors_np)
        if valid_mask.sum() == 0:
            c_indices = {'all': 0.5}
        else:
            c_indices = {
                'all': concordance_index(
                    event_times_np[valid_mask],
                    risk_scores_np[valid_mask],
                    censors_np[valid_mask]
                )
            }
        current_c_index = c_indices['all']
        if current_c_index > best_c_index:
            best_c_index = current_c_index
            best_c_index_list = [current_c_index]
        print(f'test survival c-index: {c_indices}')

    return best_c_index_list, best_c_index

def TCGA_Dataset_survival_prediction(fold, epochs, cancer_types, pretrain_model_path, fixed, device_id):
    pancancer_c_index = []
    for cancer in cancer_types:
        print(cancer)
        train_dataset = CancerDataset(omics_files, ['gex', 'methy', 'mut', 'cna'], clinical_file, train_index_path,
                                      fold + 1, [cancer])
        test_dataset = CancerDataset(omics_files, ['gex', 'methy', 'mut', 'cna'], clinical_file, test_index_path,
                                     fold + 1, [cancer])
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=True
        )

        test_dataloader = DataLoader(test_dataset, batch_size=1)

        task = {'output_dim': 1}
        model = DownStream_predictor(4,
                                     [6016, 6617, 4539, 7460],
                                     64,
                                     [2048, 512],
                                     [512, 2048],
                                     pretrain_model_path,
                                     task, omics_data_type,
                                     fixed, omics,
                                     0.01
                                     )
        torch.cuda.set_device(device_id)
        model.cuda()
        param_groups = [
            {'params': model.cross_encoders.parameters(), 'lr': 0.0001},
            {'params': model.downstream_predictor.parameters(), 'lr': 0.0001},
        ]

        optimizer = torch.optim.Adam(param_groups, weight_decay=1e-4)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        best_c_index = 0
        best_c_indices = []
        patience = 40
        counter = 0
        best_model_state = None

        for epoch in range(epochs):
            start_time = time.time()
            train_survival(train_dataloader, model, epoch, cancer, fold, optimizer, omics)
            current_c_index = cal_c_index(test_dataloader, model, best_c_index, best_c_indices, omics)[1]  # 获取当前C-index

            if current_c_index > best_c_index:
                best_c_index = current_c_index
                best_c_indices = [current_c_index]
                counter = 0
                best_model_state = model.state_dict().copy()
                print(f'New best C-index for {cancer}: {best_c_index:.4f} at epoch {epoch}')
            else:
                counter += 1
                print(f'No improvement for {counter} epochs. Best score: {best_c_index:.4f}')
                if counter >= patience:
                    print(f'Early stopping at epoch {epoch}，Best score: {best_c_index:.4f}')
                    break

            print(f'{fold} time used: ', time.time() - start_time)

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        best_c_indices.insert(0, cancer)
        pancancer_c_index.append(best_c_indices)

        #   clean memory of gpu cuda
        del model
        del optimizer
        del train_dataloader
        del test_dataloader
        torch.cuda.empty_cache()

    pancancer_c_index = pd.DataFrame(pancancer_c_index,
                                     columns=['cancer', 'multiomics'])
    pancancer_c_index.to_csv(
        os.path.join(base_path, f'model/all_pancancer_pretrain_cross_encoders_c_index_fold{fold}_all_omics.csv'),
        encoding='utf-8')
    # pancancer_c_index.to_csv(
    #     os.path.join(base_path, f'model/{cancer_types}_pancancer_pretrain_cross_encoders_c_index_fold{fold}_all_omics.csv'),
    #     encoding='utf-8')

#   cancer classification
def train_classification(dataloader, model, epoch, cancer, fold, optimizer, omics, criterion):
    total_loss = 0
    model.train()
    pancancer_embedding = torch.Tensor([]).cuda()
    accumulation_steps = 1

    with tqdm(dataloader, unit='batch') as tepoch:
        total_samples = 0
        all_labels = []
        all_predictions = []

        for batch, data in enumerate(tepoch):
            tepoch.set_description(f" Epoch {epoch}: ")
            os_event, os_time, omics_data, cancer_label = data
            cancer_label = cancer_label.cuda()
            cancer_label = cancer_label.squeeze()
            input_x = [omics_data[key].cuda() for key in omics_data.keys()]
            classification_pred = model(input_x, os_event.size(0), omics)
            _, labels_pred = torch.max(classification_pred, 1)

            pred_loss = criterion(classification_pred, cancer_label)
            optimizer.zero_grad()
            pred_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += pred_loss.item()
            if epoch % 5 == 0:
                with torch.no_grad():
                    embedding_tensor = model.cross_encoders.get_embedding(input_x, os_event.size(0), omics)
                    pancancer_embedding = torch.concat((pancancer_embedding, embedding_tensor), dim=0)

            total_samples += cancer_label.size(0)
            all_labels.extend(cancer_label.tolist())
            all_predictions.extend(labels_pred.tolist())
            tepoch.set_postfix(loss=pred_loss.item())

            del classification_pred, labels_pred
            if batch % 50 == 0:
                torch.cuda.empty_cache()
        # if epoch % 5 == 0:
        #     torch.save(pancancer_embedding, os.path.join(base_path,
        #                                                  f'model/model_dict/train_pancancer_fold{fold}_epoch{epoch}_embedding.pt'))
        del pancancer_embedding
        torch.cuda.empty_cache()

        # Calculate accuracy, precision, recall and F1 score
        acc = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')

        print('fold {} train:, Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'
              .format(fold, total_loss / len(dataloader), acc, precision, recall, f1))


def test_classification(dataloader, model, epoch, cancer, fold, optimizer, omics, criterion):
    total_loss = 0
    model.eval()
    pancancer_embedding = torch.Tensor([]).cuda()

    with torch.no_grad():
        with tqdm(dataloader, unit='batch') as tepoch:
            total_samples = 0
            all_labels = []
            all_predictions = []

            for batch, data in enumerate(tepoch):
                tepoch.set_description(f" Epoch {epoch}: ")
                os_event, os_time, omics_data, cancer_label = data
                cancer_label = cancer_label.cuda()
                cancer_label = cancer_label.squeeze()

                input_x = [omics_data[key].cuda() for key in omics_data.keys()]
                classification_pred = model(input_x, os_event.size(0), omics)
                _, labels_pred = torch.max(classification_pred, 1)

                pred_loss = criterion(classification_pred, cancer_label)
                total_loss += pred_loss.item()

                if epoch % 5 == 0:
                    embedding_tensor = model.cross_encoders.get_embedding(input_x, os_event.size(0), omics)
                    pancancer_embedding = torch.concat((pancancer_embedding, embedding_tensor), dim=0)

                total_samples += cancer_label.size(0)
                all_labels.extend(cancer_label.tolist())
                all_predictions.extend(labels_pred.tolist())

                tepoch.set_postfix(loss=pred_loss.item())
            # if epoch % 5 == 0:
            #     torch.save(pancancer_embedding, os.path.join(base_path,
            #                                                  f'model/model_dict/test_pancancer_fold{fold}_epoch{epoch}_embedding.pt'))

            # Calculate accuracy, precision, recall and F1 score
            acc = accuracy_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions, average='macro')
            recall = recall_score(all_labels, all_predictions, average='macro')
            f1 = f1_score(all_labels, all_predictions, average='macro')

            print('fold {} test:, Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.
                  format(fold, total_loss / len(dataloader), acc, precision, recall, f1))

            del pancancer_embedding
            torch.cuda.empty_cache()
            return acc, precision, recall, f1

def TCGA_Dataset_classification(fold, epochs, pretrain_model_path, fixed, device_id):
    criterion = torch.nn.CrossEntropyLoss()

    train_dataset = CancerDataset(omics_files, ['gex', 'methy', 'mut', 'cna'], clinical_file, train_index_path,
                                  fold + 1)
    test_dataset = CancerDataset(omics_files, ['gex', 'methy', 'mut', 'cna'], clinical_file, test_index_path,
                                 fold + 1)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

    task = {'output_dim': 32}

    model = DownStream_predictor(4,
                                 [6016, 6617, 4539, 7460],
                                 64,
                                 [2048, 512],
                                 [512, 2048],
                                 pretrain_model_path,
                                 task, omics_data_type,
                                 fixed, omics,
                                 0.01
                                )

    torch.cuda.set_device(device_id)
    model.cuda()
    param_groups = [
        {'params': model.cross_encoders.parameters(), 'lr': 0.00001, 'weight_decay': 1e-5},
        {'params': model.downstream_predictor.parameters(), 'lr': 0.0001, 'weight_decay': 1e-5},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )

    best_acc = 0
    best_f1 = 0
    classification_score = []
    patience = 25
    counter = 0
    best_model_state = None

    torch.cuda.empty_cache()

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_classification(train_dataloader, model, epoch, 'pancancer', fold, optimizer, omics, criterion)

        model.eval()
        acc, precision, recall, f1 = test_classification(test_dataloader, model, epoch, 'pancancer', fold, optimizer,
                                                         omics, criterion)
        scheduler.step(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            classification_score = [[fold, acc, precision, recall, f1]]
            counter = 0
            best_model_state = model.state_dict().copy()
            print(f'New best f1!: {best_f1:.4f} at epoch {epoch}')
            # torch.save(best_model_state,
            #            os.path.join(base_path, f'model/best_classification_model_fold{fold}.pt'))
        else:
            counter += 1
            print(f'No improvement for {counter} epochs. Best f1: {best_f1:.4f}')

            if counter >= patience:
                print(f'Early stopping at epoch {epoch}。Best f1: {best_f1:.4f}')
                break

        classification_score_df = pd.DataFrame(classification_score,
                                               columns=['fold', 'acc', 'precision', 'recall', 'f1'])
        classification_score_df.to_csv(os.path.join(base_path, f'model/classification_score_fold{fold}.csv'))

        print(f'fold {fold} time used: ', time.time() - start_time)

        torch.cuda.empty_cache()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Loaded best model with f1: {best_f1:.4f}, acc: {best_acc:.4f}')

    del model
    del optimizer
    del scheduler
    del train_dataloader
    del test_dataloader
    torch.cuda.empty_cache()
    print(f'Classification training completed for fold {fold}. Best F1: {best_f1:.4f}')

#   example of TMO-Net five fold training

# TCGA_Dataset_pretrain(fold = 0, epochs = 300, device_id = 0)
cancer_type_list = ['BLCA', 'BRCA', 'CESC', 'GBM', 'HNSC', 'KIRC', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'PAAD', 'SARC']
# cancer_type_list = ['BLCA']

TCGA_Dataset_survival_prediction(0, 300, cancer_type_list,
    f"D:\TMO-Netplus\model\model_dict/TCGA_pancancer_pretrain_model_fold0_dim64.pt",
                False, 0)
# # # # #
TCGA_Dataset_classification(0, 300,
    f'D:\TMO-Netplus\model\model_dict/TCGA_pancancer_pretrain_model_fold0_dim64.pt',
                False, 0)

# for i in range(5):
#     TCGA_Dataset_pretrain(i, 50, 0)

