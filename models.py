import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''Marginal GRU-ODE Cell'''
class cross_ltv(nn.Module):
    def __init__(self, args, feat_sizes, device):
        super(cross_ltv, self).__init__()
        # compute the vocab size for each categorical variable
        categorical_feature_sizes = feat_sizes['cat_feat_sizes']
        numeric_feature_size = feat_sizes['num_feat_sizes']
        numeric_dim = 10 * numeric_feature_size

        self.num_tasks = 20
        self.num_experts = args.num_experts

        # Create embedding layers for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=size, embedding_dim=int(np.log2(size)))
            for size in categorical_feature_sizes
        ])

        # Linear layer for numeric features
        self.numeric = nn.Linear(numeric_feature_size, numeric_dim)

        # Total size of total embeddings
        total_emb_size = sum(e.embedding_dim * self.num_tasks for e in self.embeddings)
        input_size = total_emb_size + numeric_dim

        # Define experts
        expert_units = input_size // 2
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(input_size, expert_units),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        ) for _ in range(self.num_experts)])

        # Define gates for each task
        self.gates = nn.ModuleList([nn.Linear(input_size, args.num_experts) for _ in range(self.num_tasks)])

        # Define task-specific layers
        self.task_layers = nn.ModuleList([nn.Linear(expert_units, 3) for _ in range(self.num_tasks)])

        self.device = device

    def compute_loss(self, x_all, y_all):
        criterion = nn.BCEWithLogitsLoss()
        classification_loss, regression_loss = 0, 0

        for i in range(self.num_tasks):

            x, y = x_all[i].to(self.device), y_all[:, i].to(self.device)

            # get the predict parameters
            prob, loc, scale = x[:, 0], x[:, 1], x[:, 2]
            softplus = nn.Softplus()
            scale = softplus(scale)

            # get the targets
            positive = torch.tensor(y > 0, dtype=torch.float32)
            safe_labels = positive * y + (1 - positive) * torch.ones_like(y)

            # compute loss
            classification_loss += criterion(prob, positive)
            regression_loss += -torch.mean(
                positive * torch.distributions.LogNormal(loc=loc, scale=scale).log_prob(safe_labels))

        return classification_loss/self.num_tasks, regression_loss/self.num_tasks

    def predict(self, x_all):
        pred_list = []
        for i in range(self.num_tasks):
            x = x_all[i]
            prob, loc, scale = x[:, 0], x[:, 1], x[:, 2]
            softplus = nn.Softplus()
            scale = softplus(scale)
            preds = prob * torch.exp(loc + 0.5 * torch.pow(scale, 2))
            pred_list.append(preds)

        return pred_list

    def forward(self, x_cat_list, x_num):
        embeddings_list = []
        for i, x_cat in enumerate(x_cat_list):
            x_cat = x_cat.to(self.device)
            embeddings = [self.embeddings[i](x_cat[:, j]) for j in range(x_cat.shape[1])]
            embeddings_list.append(torch.cat(embeddings, dim=1))
        x_cat = torch.cat(embeddings_list, dim=1)

        x_num = x_num.to(self.device)
        x_num = self.numeric(x_num)

        # Combine categorical and numerical features
        x = torch.cat([x_cat, x_num], dim=1)

        # Apply each expert to the input
        expert_outputs = [F.relu(expert(x)) for expert in self.experts]

        # Compute gate values and apply them to expert outputs
        task_outputs = []
        for i in range(self.num_tasks):
            gate_outputs = F.softmax(self.gates[i](x), dim=1)
            weighted_expert_output = sum(
                [gate_outputs[:, j:j + 1] * expert_outputs[j] for j in range(self.num_experts)])
            task_outputs.append(self.task_layers[i](weighted_expert_output))

        return task_outputs


class single_ltv(nn.Module):
    def __init__(self, args, feat_sizes, device):
        super(single_ltv, self).__init__()

        # compute the vocab size for each categorical variable
        categorical_feature_sizes = feat_sizes['cat_feat_sizes']
        numeric_feature_size = feat_sizes['num_feat_sizes']
        numeric_dim = 10 * numeric_feature_size

        # Create embedding layers for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=size, embedding_dim=min(50, (size + 1) // 2))
            for size in categorical_feature_sizes
        ])

        # Linear layer for numeric features
        self.numeric = nn.Linear(numeric_feature_size, numeric_dim)

        # Total size of total embeddings
        total_emb_size = sum(e.embedding_dim for e in self.embeddings)

        # Construct the model
        self.model = nn.Sequential(
            nn.Linear(total_emb_size + numeric_dim, (total_emb_size + numeric_dim) // 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear((total_emb_size + numeric_dim) // 2, 3)
        )

        self.device = device

    def compute_loss(self, x, y):
        x, y = x.to(self.device), y.to(self.device)

        criterion = nn.BCEWithLogitsLoss()

        # get the predict parameters
        prob, loc, scale = x[:, 0], x[:, 1], x[:, 2]
        softplus = nn.Softplus()
        scale = softplus(scale)

        # get the targets
        positive = torch.tensor(y > 0, dtype=torch.float32)
        safe_labels = positive * y + (1 - positive) * torch.ones_like(y)

        # compute loss
        classification_loss = criterion(prob, positive)
        regression_loss = -torch.mean(
            positive * torch.distributions.LogNormal(loc=loc, scale=scale).log_prob(safe_labels))

        return classification_loss, regression_loss

    def predict(self, x):
        prob, loc, scale = x[:, 0], x[:, 1], x[:, 2]
        softplus = nn.Softplus()
        scale = softplus(scale)
        preds = prob * torch.exp(loc + 0.5 * torch.pow(scale, 2))
        return preds

    def forward(self, x_cat, x_num):
        x_cat, x_num = x_cat.to(self.device), x_num.to(self.device)

        embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_cat = torch.cat(embeddings, dim=1)

        x_num = self.numeric(x_num)  # Ensure numeric data is float for the linear layer

        # Combine categorical and numerical features
        x = torch.cat([x_cat, x_num], dim=1)

        # Pass the combined features through the output layer
        x = self.model(x)

        return x

