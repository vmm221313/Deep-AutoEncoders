import torch
import torch.nn as nn


class AE_3D_500cone_bn_custom(nn.Module):
    def __init__(self, hidden_dim_1, hidden_dim_2, hidden_dim_3, n_features):
        super(AE_3D_500cone_bn_custom, self).__init__()
        self.en1 = nn.Linear(n_features, hidden_dim_1)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1)
        self.en2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)
        self.en3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.bn3 = nn.BatchNorm1d(hidden_dim_3)
        self.en4 = nn.Linear(hidden_dim_3, 3)
        self.bn5 = nn.BatchNorm1d(3)
        self.de1 = nn.Linear(3, hidden_dim_1)
        self.bn6 = nn.BatchNorm1d(hidden_dim_1)
        self.de2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.bn7 = nn.BatchNorm1d(hidden_dim_2)
        self.de3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.bn8 = nn.BatchNorm1d(hidden_dim_3)
        self.de4 = nn.Linear(hidden_dim_3, n_features)
        self.relu = nn.ReLU()

    def encode(self, x):
        h1 = self.bn1(self.relu(self.en1(x)))
        h2 = self.bn2(self.relu(self.en2(h1)))
        h3 = self.bn3(self.relu(self.en3(h2)))
        z = self.en4(h3)
        return z

    def decode(self, x):
        h5 = self.bn6(self.relu(self.de1(self.bn5(self.relu(x)))))
        h6 = self.bn7(self.relu(self.de2(h5)))
        h7 = self.bn8(self.relu(self.de3(h6)))
        return self.de4(h7)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        pass


# +
#AE_3D_500cone_bn_custom
#loss = 0.013833339 (50 epochs)
# -


