# -*- coding: utf-8 -*-
"""
Modified from the following:
https://github.com/quantumiracle/Cascading-Decision-Tree/blob/d660c442175c3a05bc70c1a45eb3eb9d91242260/src/cdt/CDT.py

Seperated feature learning tree from the rest of the tree pipeline (now CDT_fl.py), parameterized
the inclusion of N feature learning trees in a cascading orientation, and modified importance
calculations to account for each of the N feature learning trees.
"""
import torch
import torch.nn as nn
from .CDT_fl import CDT_fl
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
    "learn_probabilities"           : 1,
    "fixed_model_weight"            : 1,
}
"""


class CDT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # print('CDT parameters: ', args)
        self.device = torch.device(self.args.get('device', 'cpu'))

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.feature_learning_init(self.args['feature_learner_args'])
        self.decision_init()

        self.max_leaf_idx = None

        if self.args.get("momentum") is not None:
            momentum = self.args.get("momentum")
            self.optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.args['lr'],
                weight_decay=self.args['weight_decay'],
                momentum=momentum
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.args['lr'],
                weight_decay=self.args['weight_decay']
            )

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=self.args['exp_scheduler_gamma']
        )

    def get_tree_weights(self, Bias=False):
        """Return tree weights as a list"""
        if Bias:
            if isinstance(self.feature_learners, list):
                return [
                    [
                        feature_learner.get_tree_weights(Bias=Bias)
                        for feature_learner in self.feature_learners
                    ],
                    self._to_numpy(self.dc_inner_nodes.weight)
                ]
            else:
                return [
                    self.feature_learners.get_tree_weights(Bias=Bias),
                    self._to_numpy(self.dc_inner_nodes.weight)
                ]
        else:  # no bias
            if isinstance(self.feature_learners, list):
                return [
                    [
                        feature_learner.get_tree_weights(Bias=Bias)
                        for feature_learner in self.feature_learners
                    ],
                    self._to_numpy(
                        self.dc_inner_nodes.weight[:, 1:]
                    )
                ]
            else:
                return [
                    self.feature_learners.get_tree_weights(Bias=Bias),
                    self._to_numpy(
                        self.dc_inner_nodes.weight[:, 1:]
                    )
                ]

    def get_fl_input_weights(self):
        if isinstance(self.feature_learners, list):
            return [
                feature_learner.get_feature_weights()
                for feature_learner in self.feature_learners
            ]

    def feature_learning_init(self, fl_args):
        if isinstance(fl_args, dict):
            self.feature_learners = CDT_fl(
                fl_args,
                self.args['input_dim'],
                self.args
            )
            next_dim = fl_args['num_intermediate_variables']
        else:
            self.feature_learners = []
            for series_idx, _fl_args in enumerate(fl_args):
                if series_idx == 0:
                    self.feature_learners.append(
                        CDT_fl(
                            _fl_args,
                            self.args['input_dim'],
                            self.args,
                            name=f"FL_{series_idx}"
                        )
                    )
                    next_dim = _fl_args['num_intermediate_variables']
                else:
                    self.feature_learners.append(
                        CDT_fl(
                            _fl_args,
                            next_dim,
                            self.args,
                            name=f"FL_{series_idx}"
                        )
                    )
                    next_dim = _fl_args['num_intermediate_variables']
        self.args['num_intermediate_variables'] = next_dim

    def feature_learning_forward(self):
        if isinstance(self.feature_learners, list):
            probs = []
            self.features = self.data
            batch_size = self.batch_size
            for feature_learner in self.feature_learners:
                probs.append(
                    feature_learner.feature_learning_forward(
                        self.features, batch_size
                    )
                )
                self.features = feature_learner \
                    .intermediate_features_construct(self.features)
                batch_size *= feature_learner.num_fl_leaves
            return probs
        else:
            probs = self.feature_learners \
                        .feature_learning_forward(self.data, self.batch_size)
            self.features = self.feature_learners \
                                .intermediate_features_construct(self.data)
            return probs

    def decision_init(self):
        self.num_dc_inner_nodes = 2 ** self.args['decision_depth'] - 1
        self.num_dc_leaves = self.num_dc_inner_nodes + 1
        self.dc_inner_nodes = nn.Linear(
            self.args['num_intermediate_variables'] + 1,
            self.num_dc_inner_nodes,
            bias=False
        )

        dc_leaves = torch.randn(
            self.num_dc_leaves,
            self.args['output_dim']
        )
        self.dc_leaves = nn.Parameter(dc_leaves)

        # learnable temperature term
        if self.args['beta_dc'] is True or self.args['beta_dc'] == 1:
            # use different beta_dc for each node
            beta_dc = torch.randn(self.num_dc_inner_nodes)
            self.beta_dc = nn.Parameter(beta_dc)
        elif self.args['beta_dc'] is False or self.args['beta_dc'] == 0:
            # or use one beta_dc across all nodes
            self.beta_dc = torch.ones(1).to(self.device)
        else:  # pass in value for beta_dc
            self.beta_dc = torch.tensor(self.args['beta_dc']).to(self.device)

    def decision_forward(self):
        """
        Forward the differentiable decision tree
        """
        # add bias to self.features
        # (batch_size * num_fl_leaves, num_intermediate_variables + 1)

        aug_features = self._data_augment_(self.features)
        # (batch_size * num_fl_leaves, num_dc_inner_nodes)
        path_prob = self.sigmoid(
            self.beta_dc * self.dc_inner_nodes(aug_features)
        )
        # batch_size * num_fl_leaves
        feature_batch_size = self.features.shape[0]

        path_prob = torch.unsqueeze(path_prob, dim=2)
        # (batch_size * num_fl_leaves, num_dc_inner_nodes, 2)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)
        # (batch_size * num_fl_leaves, 1, 1)
        _mu = aug_features.data.new(feature_batch_size, 1, 1).fill_(1.)

        begin_idx = 0
        end_idx = 1
        for layer_idx in range(0, self.args['decision_depth']):
            _path_prob = path_prob[:, begin_idx:end_idx, :]

            _mu = _mu.view(feature_batch_size, -1, 1).repeat(1, 1, 2)
            _mu = _mu * _path_prob
            begin_idx = end_idx  # index for each layer
            end_idx = begin_idx + 2 ** (layer_idx + 1)
        # (batch_size * num_fl_leaves, num_dc_inner_nodes)
        mu = _mu.view(feature_batch_size, self.num_dc_leaves)
        return mu

    def decision_leaves(self, p):
        if self.args["classification"]:
            # (num_dc_leaves, output_dim)
            distribution_per_leaf = self.softmax(self.dc_leaves)
            # sum(probability of each leaf * leaf distribution)
            average_distribution = torch.mm(p, distribution_per_leaf)
            return average_distribution
        else:
            return torch.mm(p, self.dc_leaves)

    def forward(self, data, LogProb=False):
        self.data = data
        self.batch_size = data.size()[0]
        # (batch_size, num_fl_leaves)
        fl_probs = self.feature_learning_forward()
        dc_probs = self.decision_forward()

        if isinstance(fl_probs, list):
            _fl_probs = [fl_probs[0]]
            for i in range(1, len(fl_probs)):
                dc_probs = dc_probs.view(
                    self.batch_size,
                    self.feature_learners[i-1].num_fl_leaves,
                    -1
                )
                dc_probs = torch.bmm(
                    _fl_probs[i-1].unsqueeze(1),
                    dc_probs
                ).squeeze(1)
                _fl_probs.append(
                    torch.bmm(
                        _fl_probs[i-1].unsqueeze(1),
                        fl_probs[i].view(
                            self.batch_size,
                            -1,
                            self.feature_learners[i].num_fl_leaves
                        )
                    ).squeeze(1)
                )
            dc_probs = dc_probs.view(
                self.batch_size, self.feature_learners[-1].num_fl_leaves, -1
            )
            _mu = torch.bmm(_fl_probs[-1].unsqueeze(1), dc_probs).squeeze(1)
        else:
            # (batch_size, num_fl_leaves, num_dc_leaves)
            dc_probs = dc_probs.view(
                self.batch_size,
                self.feature_learners.num_fl_leaves,
                -1
            )
            # (batch_size, num_dc_leaves)
            _mu = torch.bmm(fl_probs.unsqueeze(1), dc_probs).squeeze(1)

        output = self.decision_leaves(_mu)

        prediction = output

        if LogProb:
            output = torch.log(output)
            prediction = torch.log(prediction)
        return prediction, output, 0

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

    def save_model(self, model_path, id=''):
        torch.save(self.state_dict(), model_path+id)

    def load_model(self, model_path, id=''):
        self.load_state_dict(torch.load(model_path+id, map_location='cpu'))
        self.eval()


if __name__ == '__main__':
    learner_args = {
        'num_intermediate_variables': 3,
        'feature_learning_depth': 2,
        'decision_depth': 2,
        'input_dim': 8,
        'output_dim': 4,
    }
