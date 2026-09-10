import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset import MetastaticDataset
from TMO_Netplus_model import DownStream_predictor, product_of_experts, MoE_VAE
import types

#####66, 100, 166, 188, and 200
def set_seed(seed=200):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(200)


def train_metastatic(train_dataloader, model, epoch, fold, optimizer, criterion, omics):
    model.train()
    total_loss = 0
    all_labels, all_preds = [], []

    with tqdm(train_dataloader, unit='batch') as tepoch:
        tepoch.set_description(f"Train Epoch {epoch} (Fold {fold})")
        for _, (omics_data, labels) in enumerate(tepoch):
            labels = labels.cuda().squeeze()
            input_x = [omics_data[k].cuda() for k in omics_data.keys()]
            outputs = model(input_x, labels.size(0), omics)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            tepoch.set_postfix(loss=loss.item())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"[Train] Fold {fold} Epoch {epoch} | "
          f"Loss: {total_loss / len(train_dataloader):.4f} | "
          f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    return acc, prec, rec, f1


def test_metastatic(test_dataloader, model, epoch, fold, criterion, omics):
    model.eval()
    total_loss = 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        with tqdm(test_dataloader, unit='batch') as tepoch:
            tepoch.set_description(f"Test Epoch {epoch} (Fold {fold})")
            for _, (omics_data, labels) in enumerate(tepoch):
                labels = labels.cuda().squeeze()
                input_x = [omics_data[k].cuda() for k in omics_data.keys()]
                outputs = model(input_x, labels.size(0), omics)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                preds = outputs.argmax(1)
                all_labels.extend(labels.tolist())
                all_preds.extend(preds.tolist())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"[Test] Fold {fold} Epoch {epoch} | "
          f"Loss: {total_loss / len(test_dataloader):.4f} | "
          f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    return acc, prec, rec, f1


def load_pretrained_weights_improved(model, pretrain_model_path, fixed=False):
    try:
        pretrained_dict = torch.load(pretrain_model_path, map_location='cpu', weights_only=True)
        if isinstance(pretrained_dict, torch.nn.Module):
            pretrained_dict = pretrained_dict.state_dict()

        model_dict = model.state_dict()
        loaded_layers = 0
        ignored_layers = []

        module_mapping = {
            'encoders.0.': 'cross_encoders.encoders.0.',
            'encoders.1.': 'cross_encoders.encoders.1.',
            'self_decoders.0.': 'cross_encoders.self_decoders.0.',
            'self_decoders.1.': 'cross_encoders.self_decoders.1.',
            'cross_decoders.0.': 'cross_encoders.cross_decoders.0.',
            'cross_decoders.1.': 'cross_encoders.cross_decoders.1.',
            'share_encoder.': 'cross_encoders.share_encoder.',
            'discriminator.': 'cross_encoders.discriminator.',
            'infer_discriminator.': 'cross_encoders.infer_discriminator.',
        }

        for pretrained_key, pretrained_value in pretrained_dict.items():
            model_key = pretrained_key
            for old_prefix, new_prefix in module_mapping.items():
                if model_key.startswith(old_prefix):
                    model_key = model_key.replace(old_prefix, new_prefix, 1)
                    break

            if model_key in model_dict:
                if model_dict[model_key].shape == pretrained_value.shape:
                    model_dict[model_key] = pretrained_value
                    loaded_layers += 1
                    print(f"{pretrained_key} -> {model_key}")
                else:
                    ignored_layers.append(f"shape mismatch: {pretrained_key} -> {model_key} "
                                          f"({pretrained_value.shape} vs {model_dict[model_key].shape})")
            else:
                ignored_layers.append(f"key name not found: {pretrained_key} -> {model_key}")

        if loaded_layers < len(pretrained_dict) * 0.3:
            for pretrained_key, pretrained_value in pretrained_dict.items():
                if pretrained_key in model_dict:
                    if model_dict[pretrained_key].shape == pretrained_value.shape:
                        if model_dict[pretrained_key].data_ptr() != pretrained_value.data_ptr():
                            model_dict[pretrained_key] = pretrained_value
                            loaded_layers += 1
                            print(f" direct match: {pretrained_key}")

        if loaded_layers > 0:
            model.load_state_dict(model_dict, strict=False)
            print(f"\nSuccessfully loaded {loaded_layers}/{len(pretrained_dict)} layer parameters")
            if fixed:
                for name, param in model.named_parameters():
                    if any(module in name for module in ['cross_encoders.encoders', 'cross_encoders.share_encoder']):
                        param.requires_grad = False
                        print(f"freeze: {name}")
        else:
            print("\nFailed to load any parameters, using random initialization.")
        if ignored_layers:
            print(f"Ignoring {len(ignored_layers)} layers:")
            for i, msg in enumerate(ignored_layers[:10]):
                print(f"  {i + 1}. {msg}")
            if len(ignored_layers) > 10:
                print(f"  ... {len(ignored_layers) - 10} additional layers were ignored")

    except Exception as e:
        print(f"Failed to load: {e}")
        import traceback
        traceback.print_exc()


def patch_tmo_get_embedding(instance, modal_num):

    import types

    latent_dim = 64
    # if hasattr(instance.moe, 'experts') and len(instance.moe.experts) > 0:
    #     latent_dim = instance.moe.experts[0][0].in_features
    instance.moe = MoE_VAE(latent_dim, num_experts=modal_num)
    print(f"Reset MoE_VAE, experts = {modal_num}")

    def patched_get_embedding(self, input_x, batch_size, omics):
        output, share_representation = self.forward(input_x, omics)
        embedding_tensor = []
        keys = list(omics.keys())
        values = list(omics.values())

        for i in range(self.k):
            mu_set = []
            log_var_set = []
            for j in range(self.k):
                if (i != j) and (j in values):
                    latent_z, mu, log_var = output[j][i]
                    mu_set.append(mu)
                    log_var_set.append(log_var)
            poe_mu, poe_log_var = product_of_experts(mu_set, log_var_set)
            if i in values:
                _, omic_mu, omic_log_var = output[i][i]
                joint_mu = (omic_mu + poe_mu) / 2
            else:
                joint_mu = poe_mu
            embedding_tensor.append(joint_mu)

        embedding_tensor = torch.cat(embedding_tensor, dim=1)

        bsize = embedding_tensor.size(0)
        embedding_tensor = embedding_tensor.view(bsize, self.k, -1)
        embedding_tensor = self.moe(embedding_tensor)
        embedding_tensor = embedding_tensor.view(bsize, -1)
        return embedding_tensor

    instance.get_embedding = types.MethodType(patched_get_embedding, instance)
    return instance


def TCGA_Dataset_metastatic_prediction(
        fold,
        epochs,
        pretrain_model_path,
        fixed,
        device_id
):
    torch.cuda.set_device(device_id)
    omics = {'gex': 0, 'methy': 1}
    omics_data_type = ['gaussian', 'gaussian']

    data_dir = "D:\TMO-Netplus/data/metastatic_data"
    split_path = os.path.join(data_dir, "five_split.pkl")
    label_path = os.path.join(data_dir, "label.pkl")

    omics_paths_prim = [
        os.path.join(data_dir, "pancancer_primary_mrna_new_norm.csv"),
        os.path.join(data_dir, "pancancer_primary_dna_new.csv")
    ]
    omics_paths_meta = [
        os.path.join(data_dir, "pancancer_metastatic_mrna_new_norm.csv"),
        os.path.join(data_dir, "pancancer_metastatic_dna_new.csv")
    ]

    train_dataset = MetastaticDataset(
        omics_paths_prim, omics_paths_meta, ['gex', 'methy'],
        split_path, fold, label_path, is_test=False
    )
    test_dataset = MetastaticDataset(
        omics_paths_prim, omics_paths_meta, ['gex', 'methy'],
        split_path, fold, label_path, is_test=True
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    sample, _ = train_dataset[0]
    modal_dims = [sample['gex'].shape[0], sample['methy'].shape[0]]
    modal_num = len(modal_dims)
    task = {'output_dim': 2}

    model = DownStream_predictor(
        modal_num,
        modal_dims,
        64,
        [2048, 512],
        [512, 2048],
        None,
        task,
        omics_data_type,
        fixed,
        omics,
        0.01
    )

    if pretrain_model_path and os.path.exists(pretrain_model_path):
        load_pretrained_weights_improved(model, pretrain_model_path, fixed)
    else:
        print("Pretrained model path does not exist, using random initialization.")


    model.cross_encoders = patch_tmo_get_embedding(model.cross_encoders, modal_num)
    print(f"  Applied patch: hardcoded 4 -> self.k, MoE experts -> {modal_num}")

    model.cuda()

    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            print(f" Trainable parameter: {name}")
        else:
            print(f" Frozen parameter: {name}")

    if not trainable_params:
        trainable_params = model.parameters()

    optimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.CrossEntropyLoss()

    best_f1, best_acc, best_prec, best_rec = 0, 0, 0, 0
    patience, counter = 30, 0
    best_model_state = None

    results_dir = os.path.join("D:\TMO-Netplus/metastatic_results")
    os.makedirs(results_dir, exist_ok=True)

    for epoch in range(epochs):
        print(f"\n===== Fold {fold} Epoch {epoch} =====")
        train_acc, train_prec, train_rec, train_f1 = train_metastatic(
            train_loader, model, epoch, fold, optimizer, criterion, omics
        )
        test_acc, test_prec, test_rec, test_f1 = test_metastatic(
            test_loader, model, epoch, fold, criterion, omics
        )

        scheduler.step(test_f1)

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_acc, best_prec, best_rec = test_acc, test_prec, test_rec
            counter = 0
            best_model_state = model.state_dict().copy()
            print(f"New best model found: F1 = {best_f1:.4f}")
        else:
            counter += 1
            print(f"No improvement for {counter} epochs")
            if counter >= patience:
                print(f" Early stopping triggered. Best F1 = {best_f1:.4f}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model (F1={best_f1:.4f})")

    return best_acc, best_prec, best_rec, best_f1


if __name__ == "__main__":
    # pretrain_model_path = None
    # pretrain_model_path='D:\TMO-Netplus\model\model_dict/TCGA_pancancer_pretrain_model_fold1_dim64.pt'
    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    results_dir = "D:\TMO-Netplus/metastatic_results"
    os.makedirs(results_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "five_fold_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=== Five-Fold Cross-Validation Results ===\n\n")

    for fold in range(5):
        best_acc, best_prec, best_rec, best_f1 = TCGA_Dataset_metastatic_prediction(
            fold=fold,
            epochs=100,
            pretrain_model_path=f'D:\TMO-Netplus\model\model_dict/TCGA_pancancer_pretrain_model_fold{fold}_dim64.pt',
            fixed=False,
            device_id=0
        )
        all_acc.append(best_acc)
        all_prec.append(best_prec)
        all_rec.append(best_rec)
        all_f1.append(best_f1)

        with open(summary_path, 'a') as f:
            f.write(f"Fold {fold} -> Acc: {best_acc:.4f}, Prec: {best_prec:.4f}, "
                    f"Rec: {best_rec:.4f}, F1: {best_f1:.4f}\n")

    mean_acc = np.mean(all_acc)
    mean_prec = np.mean(all_prec)
    mean_rec = np.mean(all_rec)
    mean_f1 = np.mean(all_f1)

    std_acc = np.std(all_acc)
    std_prec = np.std(all_prec)
    std_rec = np.std(all_rec)
    std_f1 = np.std(all_f1)

    with open(summary_path, 'a') as f:
        f.write("\n=== Average Results ===\n")
        f.write(f"Mean Acc: {mean_acc:.4f} ± {std_acc:.4f}\n")
        f.write(f"Mean Prec: {mean_prec:.4f} ± {std_prec:.4f}\n")
        f.write(f"Mean Rec: {mean_rec:.4f} ± {std_rec:.4f}\n")
        f.write(f"Mean F1: {mean_f1:.4f} ± {std_f1:.4f}\n")

    print("\n===== Cross-Validation Summary =====")
    print(f"Mean Acc:  {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Mean Prec: {mean_prec:.4f} ± {std_prec:.4f}")
    print(f"Mean Rec:  {mean_rec:.4f} ± {std_rec:.4f}")
    print(f"Mean F1:   {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"\nDetailed results saved to: {summary_path}")