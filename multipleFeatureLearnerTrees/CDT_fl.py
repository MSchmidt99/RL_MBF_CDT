import torch
import torch.nn as nn
import numpy as np

"""
learner_args = {
    "feature_learner_args"          : [
        {
            "num_intermediate_variables"    : 8,
            "feature_learning_depth"        : 2,
            "weight_decay"                  : 0.001,
            "exp_scheduler_gamma"           : 1.,
            "beta_fl"                       : 1,
        },
        {
            "num_intermediate_variables"    : 8,
            "feature_learning_depth"        : 2,
            "weight_decay"                  : 0.001,
            "exp_scheduler_gamma"           : 1.,
            "beta_fl"                       : 1,
        }
    ]
    "decision_depth"                : 4,
    "lr"                            : 1e-3,
    "K_epoch"                       : 1,
    "weight_decay"                  : 0.001,
    "gamma"                         : 0.0,
    "exp_scheduler_gamma"           : 1.,
    "device"                        : "cuda:0" if torch.cuda.is_available() else "cpu",
    "greatest_path_probability"     : 0,
    "beta_dc"                       : 1,
    "classification"                : 1,
    "clip_sample_weights"           : [0.25, 4.],
    "learn_probabilities"           : 1,
    "max_loss_for_weight"           : None,
    "fixed_model_weight"            : 1,
}
"""

class CDT_fl(nn.Module):
    def __init__(self, args, input_dim, dc_args, name=""):
        super().__init__()
        self.args = args
        self.input_dim = input_dim
        # print('CDT parameters: ', args)
        self.device = torch.device(dc_args.get('device', 'cpu'))
        self.name = name

        self.sigmoid = nn.Sigmoid()

        self.feature_learning_init()

        self.max_leaf_idx = None

        if dc_args.get("momentum") is not None:
            momentum = dc_args.get("momentum")
            self.optimizer = torch.optim.SGD(
                self.parameters(),
                lr=dc_args['lr'],
                weight_decay=dc_args['weight_decay'],
                momentum=momentum
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=dc_args['lr'],
                weight_decay=dc_args['weight_decay']
            )

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=dc_args['exp_scheduler_gamma']
        )
        
    def get_tree_weights(self, Bias=False):
        """Return tree weights as a list"""
        if Bias:
            return (
                self._to_numpy(self.fl_inner_nodes.weight)
            )
        else:  # no bias
            return (
                self._to_numpy(
                    self.fl_inner_nodes.weight[:, 1:]
                )
            )
        
    def get_feature_weights(self):
        return self._to_numpy(
            self.fl_leaf_weights
        ).reshape(
            self.num_fl_leaves,
            self.args['num_intermediate_variables'],
            self.input_dim
        )
    
    def feature_learning_init(self):
        self.num_fl_inner_nodes = 2 ** self.args['feature_learning_depth'] - 1
        self.num_fl_leaves = self.num_fl_inner_nodes + 1
        self.fl_inner_nodes = nn.Linear(
            self.input_dim + 1,
            self.num_fl_inner_nodes,
            bias=False
        ).to(self.device)
        # coefficients of feature combinations
        fl_leaf_weights = torch.randn(
            self.num_fl_leaves * self.args['num_intermediate_variables'],
            self.input_dim
        ).to(self.device)
        self.fl_leaf_weights = nn.Parameter(fl_leaf_weights)

        # learnable temperature term
        if self.args['beta_fl'] is True or self.args['beta_fl'] == 1:
            # use different beta_fl for each node
            beta_fl = torch.randn(self.num_fl_inner_nodes).to(self.device)
            self.beta_fl = nn.Parameter(beta_fl)
        elif self.args['beta_fl'] is False or self.args['beta_fl'] == 0:
            # or use one beta_fl across all nodes
            self.beta_fl = torch.ones(1).to(self.device)
        else:  # pass in value for beta_fl
            self.beta_fl = torch.tensor(self.args['beta_fl']).to(self.device)

    def feature_learning_forward(self, data, batch_size):
        """
        Forward the tree for feature learning.
        Return the probabilities for reaching each leaf.
        """
        self.batch_size = batch_size
        aug_data = self._data_augment_(data)
        # (batch_size, num_fl_inner_nodes)
        path_prob = self.sigmoid(
            self.beta_fl * self.fl_inner_nodes(aug_data)
        )

        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1-path_prob), dim=2)
        _mu = aug_data.data.new(self.batch_size, 1, 1).fill_(1.)

        begin_idx = 0
        end_idx = 1
        for layer_idx in range(0, self.args['feature_learning_depth']):
            _path_prob = path_prob[:, begin_idx:end_idx, :]

            _mu = _mu.view(self.batch_size, -1, 1).repeat(1, 1, 2)
            _mu = _mu * _path_prob
            begin_idx = end_idx  # index for each layer
            end_idx = begin_idx + 2 ** (layer_idx + 1)
        mu = _mu.view(self.batch_size, self.num_fl_leaves)
        return mu
    
    def intermediate_features_construct(self, data):
        """
        Construct the intermediate features for decision making, with learned
        feature combinations from feature learning module.
        """
        features = (
            self.fl_leaf_weights.view(-1, self.input_dim)
            @ data.transpose(0, 1)  # data: (batch_size, feature_dim)
        )  # return: (num_fl_leaves * num_intermediate_variables, batch)
        features = (
            features.contiguous().view(
                self.num_fl_leaves,
                self.args['num_intermediate_variables'],
                -1
            ).permute(2, 0, 1).contiguous().view(
                -1,
                self.args['num_intermediate_variables']
            )
        )  # return: (batch_size * num_fl_leaves, num_intermediate_variables)
        return features
        
    """ Add constant 1 onto the front of each instance, serving as the bias """
    def _data_augment_(self, input):
        batch_size = input.size()[0]
        input = input.view(batch_size, -1)
        bias = torch.ones(batch_size, 1).to(self.device)
        input = torch.cat((bias, input), 1)
        return input
        
    def _to_numpy(self, torch_tensor):
        return np.copy(
            torch_tensor.detach().cpu().numpy()
        )