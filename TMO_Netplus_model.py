import random
import torch.nn as nn
import torch
import torch.nn.functional as F
from loss_function import KL_loss, reconstruction_loss, KL_divergence
from functools import reduce
import numpy as np
from joblib.func_inspect import _clean_win_chars

def reparameterize(mean, logvar):
    std = torch.exp(logvar / 2)
    epsilon = torch.randn_like(std).cuda()
    return epsilon * std + mean

def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)

def un_dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        un_dfs_freeze(child)


def product_of_experts(mu_set_, log_var_set_):
    tmp = 0
    for i in range(len(mu_set_)):
        tmp += torch.div(1, torch.exp(log_var_set_[i]))

    poe_var = torch.div(1., tmp)
    poe_log_var = torch.log(poe_var)

    tmp = 0.
    for i in range(len(mu_set_)):
        tmp += torch.div(1., torch.exp(log_var_set_[i])) * mu_set_[i]
    poe_mu = poe_var * tmp
    return poe_mu, poe_log_var

class MoE_VAE(nn.Module):
    def __init__(self, latent_dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU())
            for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * num_experts, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, mu_set):
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_outputs.append(expert(mu_set[:, i]))

        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, M, D]


        gate_input = expert_outputs.view(expert_outputs.size(0), -1)  # [B, M*D]
        gate_weights = self.gate(gate_input)  # [B, M]
        combined_mu = gate_weights.unsqueeze(-1) * expert_outputs

        return combined_mu

class LinearLayer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 dropout: float = 0.2,
                 batchnorm: bool = False,
                 activation=None):
        super(LinearLayer, self).__init__()
        self.linear_layer = nn.Linear(input_dim, output_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.batchnorm = nn.BatchNorm1d(output_dim) if batchnorm else None

        self.activation = None
        if activation is not None:
            if activation == 'relu':
                self.activation = F.relu
            elif activation == 'sigmoid':
                self.activation = torch.sigmoid
            elif activation == 'tanh':
                self.activation = torch.tanh
            elif activation == 'leakyrelu':
                self.activation = torch.nn.LeakyReLU()

    def forward(self, input_x):
        x = self.linear_layer(input_x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, activation):
        super(encoder, self).__init__()
        self.FeatureEncoder = nn.ModuleList(
            [LinearLayer(input_dim, hidden_dims[0], batchnorm=True, activation=activation)])
        for i in range(len(hidden_dims) - 1):
            self.FeatureEncoder.append(
                LinearLayer(hidden_dims[i], hidden_dims[i + 1], batchnorm=True, activation=activation))

        self.identity_proj = nn.Linear(input_dim, hidden_dims[-1])
        self.activation1 = torch.nn.Sigmoid()

        self.mu_predictor = nn.Sequential(nn.Linear(hidden_dims[-1], latent_dim))

        self.log_var_predictor = nn.Sequential(nn.Linear(hidden_dims[-1], latent_dim))

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    def forward(self, x):
        identity = x
        for layer in self.FeatureEncoder:
            x = layer(x)

        identity_proj = self.identity_proj(identity)
        if identity_proj.size(1) != x.size(1):
            identity_proj = F.adaptive_avg_pool1d(identity_proj.unsqueeze(1), x.size(1)).squeeze(1)
        x = x * self.activation1(identity_proj)
        mu = self.mu_predictor(x)
        log_var = self.log_var_predictor(x)
        latent_z = self.reparameterize(mu, log_var)
        return latent_z, mu, log_var


class decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims, activation):
        super(decoder, self).__init__()
        self.FeatureDecoder = nn.ModuleList([LinearLayer(latent_dim, hidden_dims[0],
                                                         dropout=0.1, batchnorm=True,
                                                         activation=activation)])
        for i in range(len(hidden_dims) - 1):
            self.FeatureDecoder.append(LinearLayer(hidden_dims[i], hidden_dims[i + 1],
                                                   dropout=0.1, batchnorm=True,
                                                   activation=activation))

        self.ReconsPredictor = LinearLayer(hidden_dims[-1], output_dim)

    def forward(self, latent_z):
        for layer in self.FeatureDecoder:
            latent_z = layer(latent_z)
        DataRecons = self.ReconsPredictor(latent_z)
        return DataRecons


class TMO_Netplus(nn.Module):
    def __init__(self,
                 #  number of multimodal
                 modal_num: int,
                 #  multimodal dimension list
                 modal_dim: list,
                 #  dimension of latent representation
                 latent_dim: int,
                 #  dimension of encoder hidden layer
                 encoder_hidden_dims: list,
                 #  dimension of decoder hidden layer
                 decoder_hidden_dims: list,
                 #  distribution/reconstruct loss of each modality
                 omics_data_type: list,
                 # weight of kl loss
                 kl_loss_weight: float,
                 pretrain=False):
        super(TMO_Netplus, self).__init__()

        self.k = modal_num
        self.encoders = nn.ModuleList(
            nn.ModuleList([encoder(modal_dim[i], latent_dim, encoder_hidden_dims, 'relu') for j in
                           range(self.k)]) for i in
            range(self.k))
        self.self_decoders = nn.ModuleList(
            [decoder(latent_dim, modal_dim[i], decoder_hidden_dims, 'relu') for i in range(self.k)])

        self.cross_decoders = nn.ModuleList(
            [decoder(latent_dim, modal_dim[i], decoder_hidden_dims, 'relu') for i in range(self.k)])

        #   modality-invariant representation
        self.share_encoder = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                           nn.BatchNorm1d(latent_dim),
                                           nn.ReLU())
        #   modal align
        self.discriminator = nn.Sequential(nn.Linear(latent_dim, 16),
                                           nn.BatchNorm1d(16),
                                           nn.ReLU(),
                                           nn.Linear(16, modal_num))

        #   infer modal and real modal align
        self.infer_discriminator = nn.ModuleList(nn.Sequential(nn.Linear(latent_dim, 16),
                                                               nn.BatchNorm1d(16),
                                                               nn.ReLU(),
                                                               nn.Linear(16, 2))
                                                 for i in range(self.k))

        self.classifier = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim),
                                        nn.BatchNorm1d(latent_dim),
                                        nn.ReLU(),
                                        nn.Linear(latent_dim, 32))

        self.moe = MoE_VAE(latent_dim)

        #   loss function hyperparameter
        self.loss_weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], requires_grad=True)

        self.omics_data_type = omics_data_type

        self.kl_loss_weight = kl_loss_weight
        #   choose to freeze the parameters of TMO-Net
        if pretrain:
            dfs_freeze(self.encoders)

    #   incomplete omics input
    def forward(self, input_x, omics):
        keys = omics.keys()
        values = list(omics.values())
        output = [[0 for _ in range(self.k)] for _ in range(self.k)]

        for (item, i) in enumerate(values):
            for j in range(self.k):
                output[i][j] = self.encoders[i][j](input_x[item])
        share_representation = self.share_representation(output, omics)

        return output, share_representation

    def compute_generate_loss(self, input_x, batch_size, omics):
        values = list(omics.values())
        # mask_k = random.randint(0, self.k*5)
        mask_k = 10
        output = [[0 for _ in range(self.k)] for _ in range(self.k)]
        for (item, i) in enumerate(values):
            for j in range(self.k):
                output[i][j] = self.encoders[i][j](input_x[item])

        self_elbo = self.self_elbo([output[i][i] for i in range(self.k)], input_x, omics, mask_k)
        cross_elbo, cross_infer_dsc_loss = self.cross_elbo(output, input_x, batch_size, omics, mask_k)
        cross_infer_loss = self.cross_infer_loss(output, omics, mask_k)
        dsc_loss = self.adversarial_loss(batch_size, output, omics, mask_k)
        generate_loss = self_elbo + 0.1 * (cross_elbo + cross_infer_loss * cross_infer_loss) - (
                    dsc_loss + cross_infer_dsc_loss) * 0.01
        return generate_loss, self_elbo, cross_elbo, cross_infer_loss, dsc_loss

    def compute_dsc_loss(self, input_x, batch_size, omics):
        mask_k = random.randint(0, self.k * 5)
        values = list(omics.values())
        output = [[0 for i in range(self.k)] for j in range(self.k)]
        for (item, i) in enumerate(values):
            for j in range(self.k):
                output[i][j] = self.encoders[i][j](input_x[item])

        cross_elbo, cross_infer_dsc_loss = self.cross_elbo(output, input_x, batch_size, omics, mask_k)
        dsc_loss = self.adversarial_loss(batch_size, output, omics, mask_k)
        return cross_infer_dsc_loss, dsc_loss

    def share_representation(self, output, omics):
        values = omics.values()
        share_features = [self.share_encoder(output[i][i][1]) for i in values]
        return share_features

    def self_elbo(self, input_x, input_omic, omics, mask_k):
        self_vae_elbo = 0
        keys = omics.keys()
        values = list(omics.values())
        r_squared = []
        for item, i in enumerate(values):
            if i != mask_k:
                latent_z, mu, log_var = input_x[i]
                reconstruct_omic = self.self_decoders[i](latent_z)
                self_vae_elbo += (self.kl_loss_weight * KL_loss(mu, log_var, 1.0) +
                                  reconstruction_loss(input_omic[item], reconstruct_omic, 1.0, self.omics_data_type[i]))
        return self_vae_elbo

    def cross_elbo(self, input_x, input_omic, batch_size, omics, mask_k):
        cross_elbo = 0
        cross_infer_loss = 0
        cross_modal_KL_loss = 0
        cross_modal_dsc_loss = 0

        values = list(omics.values())
        for i in range(self.k):
            if i in values:
                real_latent_z, real_mu, real_log_var = input_x[i][i]
                mu_set = []
                log_var_set = []
                for j in range(self.k):
                    if (i != j) and (j in values) and (j != mask_k):
                        latent_z, mu, log_var = input_x[j][i]
                        mu_set.append(mu)
                        log_var_set.append(log_var)

                poe_mu, poe_log_var = product_of_experts(mu_set, log_var_set)
                poe_latent_z = reparameterize(poe_mu, poe_log_var)
                if i in values:
                    reconstruct_omic = self.self_decoders[i](poe_latent_z)

                    cross_elbo += (self.kl_loss_weight * KL_loss(poe_mu, poe_log_var, 1.0) +
                                   reconstruction_loss(input_omic[values.index(i)],
                                                       reconstruct_omic, 1.0, self.omics_data_type[i]))
                    cross_infer_loss += reconstruction_loss(real_mu, poe_mu, 1.0, 'gaussian')

                    cross_modal_KL_loss += KL_divergence(poe_mu, real_mu, poe_log_var, real_log_var)

                    real_modal = torch.tensor([1 for j in range(batch_size)]).cuda()
                    infer_modal = torch.tensor([0 for j in range(batch_size)]).cuda()
                    pred_real_modal = self.infer_discriminator[i](real_mu)
                    pred_infer_modal = self.infer_discriminator[i](poe_mu)

                    cross_modal_dsc_loss += F.cross_entropy(pred_real_modal, real_modal, reduction='none')
                    cross_modal_dsc_loss += F.cross_entropy(pred_infer_modal, infer_modal, reduction='none')

        cross_modal_dsc_loss = cross_modal_dsc_loss.sum(0) / (self.k * batch_size)
        return cross_elbo + cross_infer_loss + self.kl_loss_weight * cross_modal_KL_loss, cross_modal_dsc_loss

    def cross_infer_loss(self, input_x, omics, mask_k):
        values = list(omics.values())
        latent_mu = [0 for i in range(len(input_x))]
        for i in range(self.k):
            if i in values:
                latent_mu[i] = input_x[i][i][1]
        infer_loss = 0
        for i in range(len(input_x)):
            if (i in values) and (i != mask_k):
                for j in range(len(input_x)):
                    if (i != j) and (j in values):
                        latent_z_infer, latent_mu_infer, _ = input_x[j][i]
                        infer_loss += reconstruction_loss(latent_mu_infer, latent_mu[i], 1.0, 'gaussian')
        return infer_loss / len(values)

    def adversarial_loss(self, batch_size, output, omics, mask_k):
        dsc_loss = 0
        values = list(omics.values())
        for i in range(self.k):
            if (i in values) and (i != mask_k):
                latent_z, mu, log_var = output[i][i]
                shared_fe = mu

                real_modal = (torch.tensor([i for j in range(batch_size)])).cuda()
                pred_modal = self.discriminator(shared_fe)
                # print(i, pred_modal)
                dsc_loss += F.cross_entropy(pred_modal, real_modal, reduction='none')

        dsc_loss = dsc_loss.sum(0) / (self.k * batch_size)
        return dsc_loss

    def get_embedding(self, input_x, batch_size, omics):
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
        embedding_tensor = embedding_tensor.view(bsize, 4, -1)
        embedding_tensor = self.moe(embedding_tensor)
        embedding_tensor = embedding_tensor.view(bsize, -1)

        return embedding_tensor

    def cross_modal_generation(self, input_x, omics):
        values = list(omics.values())

        output = [[0 for _ in range(self.k)] for _ in range(self.k)]
        for (item, i) in enumerate(values):
            for j in range(self.k):
                output[i][j] = self.encoders[i][j](input_x[item])

        reconstruct_omics = []
        for i in range(self.k):
            if i in values:
                real_latent_z, real_mu, real_log_var = output[i][i]
                reconstruct_self_omic = self.self_decoders[i](real_latent_z)
                reconstruct_omics.append(reconstruct_self_omic)

            else:
                mu_set = []
                log_var_set = []
                for j in range(self.k):
                    if (i != j) and (j in values):
                        latent_z, mu, log_var = output[j][i]
                        mu_set.append(mu)
                        log_var_set.append(log_var)

                poe_mu, poe_log_var = product_of_experts(mu_set, log_var_set)
                poe_latent_z = reparameterize(poe_mu, poe_log_var)
                reconstruct_cross_omic = self.self_decoders[i](poe_latent_z)
                reconstruct_omics.append(reconstruct_cross_omic)

        return reconstruct_omics

    def classifier_loss(self, embeddings, labels):

        pred_tensor = self.classifier(embeddings)
        closs = F.cross_entropy(pred_tensor, labels, reduction='none')
        return closs

    @staticmethod
    def contrastive_loss(embeddings, labels, margin=1.0, distance='euclidean'):

        if distance == 'euclidean':
            distances = torch.cdist(embeddings, embeddings)
        elif distance == 'cosine':
            normed_embeddings = F.normalize(embeddings, p=2, dim=1)
            distances = 1 - torch.mm(normed_embeddings, normed_embeddings.transpose(0, 1))
        else:
            raise ValueError(f"Unknown distance type: {distance}")

        labels_matrix = labels.view(-1, 1) == labels.view(1, -1)

        positive_pair_distances = distances * labels_matrix.float()
        negative_pair_distances = distances * (1 - labels_matrix.float())

        positive_loss = positive_pair_distances.sum() / labels_matrix.float().sum()
        negative_loss = F.relu(margin - negative_pair_distances).sum() / (1 - labels_matrix.float()).sum()

        return positive_loss + negative_loss



class DownStream_predictor(nn.Module):
    def __init__(self, modal_num, modal_dim, latent_dim, encoder_hidden_dim, decoder_hidden_dim, pretrain_model_path,
                 task, omics_data_type, fixed, omics, kl_loss_weight):
        super(DownStream_predictor, self).__init__()
        self.k = modal_num
        #   cross encoders
        self.cross_encoders = TMO_Netplus(modal_num, modal_dim, latent_dim, encoder_hidden_dim, decoder_hidden_dim,
                                      omics_data_type, kl_loss_weight)
        if pretrain_model_path:
            print('load pretrain model')
            model_pretrain_dict = torch.load(pretrain_model_path, map_location='cpu')
            self.cross_encoders.load_state_dict(model_pretrain_dict)

        self.downstream_predictor = nn.Sequential(nn.Linear(latent_dim * self.k, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.Dropout(0.3),
                                                  nn.ReLU(),

                                                  nn.Linear(128, 64),
                                                  nn.BatchNorm1d(64),
                                                  nn.Dropout(0.3),
                                                  nn.ReLU(),

                                                  nn.Linear(64, task['output_dim']),
                                                  nn.Sigmoid())
        omics_values = set(omics.values())
        for i in range(self.k):
            if i not in omics_values:
                print('fix cross-modal encoders')
                dfs_freeze(self.cross_encoders.encoders[:][i])

        if fixed:
            dfs_freeze(self.cross_encoders)

    def un_dfs_freeze_encoder(self):
        un_dfs_freeze(self.cross_encoders)

    #   return embedding
    def get_embedding(self, input_x, batch_size, omics):
        output, share_representation = self.cross_encoders(input_x, batch_size)
        embedding_tensor = []
        keys = list(omics.keys())
        share_features = [share_representation[omics[key]] for key in keys]
        share_features = sum(share_features) / len(keys)
        for i in range(self.k):
            mu_set = []
            log_var_set = []
            for j in range(len(omics)):
                latent_z, mu, log_var = output[omics[keys[j]]][i]
                mu_set.append(mu)
                log_var_set.append(log_var)
            poe_mu, poe_log_var = product_of_experts(mu_set, log_var_set)
            poe_latent_z = reparameterize(poe_mu, poe_log_var)

            embedding_tensor.append(poe_mu)
        embedding_tensor = torch.cat(embedding_tensor, dim=1)
        # multi_representation = torch.concat((embedding_tensor, share_features), dim=1)
        return embedding_tensor

    def forward(self, input_x, batch_size, omics):
        multi_representation = self.cross_encoders.get_embedding(input_x, batch_size, omics)
        downstream_output = self.downstream_predictor(multi_representation)

        return downstream_output
