"""
PyTorch Implementation of EsiGCF
ESIGCF: Extremely
Simplified but Intent-enhanced Graph Collaborative Filtering for Recommendation. Yang et al. EAAI'25
For more information, please refer to:
"""

import torch
from torch import nn
import utility.utility_function.losses as losses
import utility.utility_function.tools as tools
import utility.utility_train.trainer as trainer
import utility.utility_data.data_graph
from utility.utility_data.load_data import *


class ESIGCF(nn.Module):
    def __init__(self, config, dataset, device):
        super(ESIGCF, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(self.config['reg_lambda'])
        self.ssl_lambda = float(self.config['ssl_lambda'])
        self.can_lambda = float(self.config['can_lambda'])
        self.temperature = float(self.config['temperature'])

        self.user_embedding = None   # Simplified user embeddings

        # Traditional user embeddings
        self.user_embedding_original = torch.nn.Embedding(num_embeddings=self.dataset.num_users,
                                                 embedding_dim=int(self.config['embedding_size']))

        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items,
                                                 embedding_dim=int(self.config['embedding_size']))

        # no pretrain
        nn.init.xavier_uniform_(self.user_embedding_original.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

        # JoGCN
        self.user_Graph = utility.utility_data.data_graph.joint_sparse_adjacency_matrix_R(self.dataset)
        # LightGCN
        # self.user_Graph = utility.utility_data.data_graph.sparse_adjacency_matrix_R(self.dataset)
        self.user_Graph = tools.convert_sp_mat_to_sp_tensor(self.user_Graph)
        self.user_Graph = self.user_Graph.coalesce().to(self.device)

        if config['dataset'] == "amazon-electronics":
            data_generator = Data(path=config['dataset_path'] + config['dataset'], batch_size=config['batch_size'])
            self.Graph = data_generator.get_adj_mat()
        else:
            # JoGCN
            self.Graph = utility.utility_data.data_graph.joint_sparse_adjacency_matrix(self.dataset)
            # self.Graph = utility.utility_data.data_graph.sparse_adjacency_matrix(self.dataset)
        self.Graph = tools.convert_sp_mat_to_sp_tensor(self.Graph)
        self.Graph = self.Graph.coalesce().to(self.device)

        self.concat_linear = nn.Linear(2 * int(self.config['embedding_size']), int(self.config['embedding_size']))
        self.activation_layer = nn.Tanh()
        self.activation = nn.Sigmoid()

    # Simplified JoGCN
    def parallel_aggregate(self):
        item_embedding = self.item_embedding.weight
        # Equation 14 (Eliminate the Tanh activation functions)
        # user_embedding = torch.sparse.mm(self.user_Graph, item_embedding)
        user_embedding = self.activation_layer(torch.sparse.mm(self.user_Graph, item_embedding))  # Equation 14
        all_embedding = torch.cat([user_embedding, item_embedding])

        all_embeddings = []

        for layer in range(int(self.config['GCN_layer'])):   # Equation 15
            all_embedding = self.activation_layer(torch.sparse.mm(self.Graph, all_embedding))
            all_embeddings.append(all_embedding)

        final_all_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.sum(final_all_embeddings, dim=1)

        users_emb, items_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])

        return users_emb, items_emb

    # Non-simplified JoGCN
    def parallel_aggregate1(self):
        user_embedding = self.user_embedding_original.weight
        item_embedding = self.item_embedding.weight
        all_embedding = torch.cat([user_embedding, item_embedding])

        all_embeddings = []

        for layer in range(int(self.config['GCN_layer'])):
            all_embedding = self.activation_layer(torch.sparse.mm(self.Graph, all_embedding))
            all_embeddings.append(all_embedding)

        final_all_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.sum(final_all_embeddings, dim=1)

        users_emb, items_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])

        return users_emb, items_emb

    def forward(self, user, positive, negative):
        all_user_embeddings, all_item_embeddings = self.parallel_aggregate()      # Simplified JoGCN
        # all_user_embeddings, all_item_embeddings = self.parallel_aggregate1()   # Non-simplified JoGCN

        user_embedding = all_user_embeddings[user.long()]
        pos_embedding = all_item_embeddings[positive.long()]
        neg_embedding = all_item_embeddings[negative.long()]

        ego_pos_emb = self.item_embedding(positive)
        ego_neg_emb = self.item_embedding(negative)

        ################################################################################################################
        # Hadamard + LeakyReLU
        can_neg_emb = torch.nn.functional.leaky_relu(pos_embedding * neg_embedding)

        # # Concatenation + LeakyReLU
        # concat_emb = torch.cat([pos_embedding, neg_embedding], dim=-1)  # [B, 2D]
        # can_neg_emb = torch.nn.functional.leaky_relu(self.concat_linear(concat_emb))  # [B, D]

        # # Weighted mean
        # can_neg_emb = torch.nn.functional.leaky_relu(0.5 * (pos_embedding + neg_embedding))

        # # Weighted Sum
        # can_neg_emb = torch.nn.functional.leaky_relu((pos_embedding + neg_embedding))

        # # Learnable scalar weight
        # alpha = torch.nn.Parameter(torch.tensor(0.5))  # 初始化为0.5
        # can_neg_emb = torch.nn.functional.leaky_relu(alpha * pos_embedding + (1 - alpha) * neg_embedding)
        ################################################################################################################

        bpr_loss = losses.get_bpr_loss(user_embedding, pos_embedding, neg_embedding)  # Equation 4

        reg_loss = losses.get_reg_loss(ego_pos_emb, ego_neg_emb)
        reg_loss = self.reg_lambda * reg_loss

        # ESIGCF
        ssl_intent_loss = losses.get_InfoNCE_loss(user_embedding, pos_embedding, self.temperature)   # Equation 19
        ssl_loss = self.ssl_lambda * ssl_intent_loss
        can_item_loss = -losses.get_InfoNCE_loss(pos_embedding, can_neg_emb, self.temperature)  # Equation 20
        can_item_loss = self.can_lambda * can_item_loss
        loss_list = [bpr_loss, reg_loss, ssl_loss, can_item_loss]

        ################################################################################################################
        # # # 1. JoGCN + $\mathcal{L}_\text{BPR}$ + $\mathcal{L}_{\text{BPR}}^\text{intent}$
        # # ssl_intent_loss = losses.get_bpr_loss(user_embedding, pos_embedding, neg_embedding)
        # # ssl_loss = self.ssl_lambda * ssl_intent_loss
        # # loss_list = [bpr_loss, reg_loss, ssl_loss]
        #
        # # 2. JoGCN + $\mathcal{L}_\text{BPR}$ + $\mathcal{L}_{\text{BPR}}^\text{item}$
        # can_item_loss = losses.get_bpr_loss(pos_embedding, can_neg_emb, neg_embedding)
        # can_item_loss = self.can_lambda * can_item_loss
        # loss_list = [bpr_loss, reg_loss, can_item_loss]
        #
        # # # 3. JoGCN + $\mathcal{L}_\text{BPR}$ + $\mathcal{L}_{\text{BPR}}^\text{intent}$ + $\mathcal{L}_{\text{BPR}}^\text{item}$
        # # ssl_intent_loss = losses.get_bpr_loss(user_embedding, pos_embedding, neg_embedding)
        # # ssl_loss = self.ssl_lambda * ssl_intent_loss
        # # can_item_loss = losses.get_bpr_loss(pos_embedding, can_neg_emb, neg_embedding)
        # # can_item_loss = self.can_lambda * can_item_loss
        # # loss_list = [bpr_loss, reg_loss, ssl_loss, can_item_loss]
        ################################################################################################################
        return loss_list

    def get_rating_for_test(self, user):
        all_user_embeddings, all_item_embeddings = self.parallel_aggregate()
        # all_user_embeddings, all_item_embeddings = self.parallel_aggregate1()

        user_embeddings = all_user_embeddings[user.long()]

        rating = self.activation(torch.matmul(user_embeddings, all_item_embeddings.t()))
        return rating


class Trainer():
    def __init__(self, args, config, dataset, device, logger):
        self.model = ESIGCF(config, dataset, device)
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset
        self.logger = logger

    # This function provides a universal training and testing process
    # that can be customized according to actual situations.
    def train(self):
        trainer.universal_trainer(self.model, self.args, self.config, self.dataset, self.device, self.logger)
