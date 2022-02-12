"""
PPO with cascading decision tree (CDT) as policy function approximator

Taken from the following:
https://github.com/quantumiracle/Cascading-Decision-Tree/blob/d660c442175c3a05bc70c1a45eb3eb9d91242260/src/cdt/deprecated/cdt_rl_train.py

Modified loss function
Added wrapper function which strings together CDTs into a boosted forest
"""
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.distributions import Categorical
import numpy as np
import pandas as pd
from .CDT_hierarchical_fl import CDT
import sys
sys.path.append("..")
# from rl import StateNormWrapper


class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, learner_args,
                 sample_weights=None, state_mask=None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_path = learner_args.get('model_path', None)
        self.device = learner_args.get('device', None)
        self.learning_rate = learner_args.get('lr', 0.0005)
        self.gamma = learner_args.get('gamma', 0.98)
        self.K_epoch = learner_args.get('K_epoch', 3)
        self.beta = learner_args.get('beta', 1.0)
        self.weight_decay = learner_args.get('weight_decay', 0.1)
        self.exp_gamma = learner_args.get('exp_scheduler_gamma', 0.98)
        self.learn_probabilities = learner_args.get('learn_probabilities', 0)

        self.sample_weights = (
            self._to_tensor(sample_weights)
            if not isinstance(sample_weights, type(None)) else None
        )
        if not isinstance(state_mask, type(None)):
            self.state_mask = self._to_tensor(state_mask, dtype=torch.bool)
            args = learner_args.copy()
            args['input_dim'] = state_dim
            self.cdt = CDT(args).to(self.device)
        else:
            self.state_mask = None
            self.cdt = CDT(learner_args).to(self.device)

        self.softmax = nn.Softmax(dim=1)
        self.calc_probs = lambda x: self.cdt.forward(x)[1]

        if learner_args.get("momentum") is not None:
            momentum = learner_args.get("momentum")
            self.optimizer = optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=momentum
            )
        else:
            self.optimizer = optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=self.exp_gamma
        )

    def make_batch(self):
        batch = (
            self.train[["s", "rs", "s'"]].copy()
        )
        self.train = []
        return batch

    def train_net(self):
        batch = self.make_batch()

        for epoch in range(self.K_epoch):
            for i, row in batch.iterrows():
                s, rs, s_prime = (
                    self._to_tensor([row["s"]]),
                    self._to_tensor([row["rs"]]),
                    self._to_tensor([row["s'"]])
                )
                loss = self.calc_loss(s, rs, s_prime, i=i)

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

    def calc_loss(self, s, rs, s_prime, i=None):
        if isinstance(self.state_mask, torch.Tensor):
            # apply estimator level state mask
            s = s.masked_select(self.state_mask).view(s.shape[0], -1)
            s_prime = s_prime.masked_select(
                self.state_mask
            ).view(s.shape[0], -1)

        # get prediction and target
        q_vals = self.calc_probs(s)
        td_target = self.softmax(rs) + self.gamma * self.calc_probs(s_prime)
        td_target = td_target / torch.sum(td_target, dim=1)
        max_target_mask = torch.eq(td_target, td_target.max(dim=1)[0]).float()

        # calc cross entropy loss using target mask or real probabilities
        if not self.learn_probabilities:
            loss = self.softXEnt(q_vals, max_target_mask)
        else:
            loss = self.softXEnt(q_vals, td_target)

        # mask loss for largest decision signal in this sample
        loss = torch.sum(loss * max_target_mask, dim=1)
        if isinstance(self.sample_weights, torch.Tensor) and i is not None:
            # apply sample weights
            loss *= self.sample_weights[i]

        return loss

    def softXEnt(self, input, target):
        logprobs = torch.log(input)
        inv_logprobs = torch.log(1 - input)
        loss_per_action = - (
            (target * logprobs) + ((1-target) * inv_logprobs)
        )
        return loss_per_action

    def choose_action(self, s):
        s = torch.from_numpy(s).unsqueeze(0).float().to(self.device)
        if isinstance(self.state_mask, torch.Tensor):
            s = s.masked_select(self.state_mask).view(s.shape[0], -1)
        # ensure input state shape is correct
        probs = self.calc_probs(s).squeeze()
        a = torch.argmax(probs, dim=-1).item()
        return a, probs

    def load_model(self):
        self.load_state_dict(torch.load(self.model_path))

    def _to_tensor(self, arr, dtype=torch.float):
        return torch.tensor(arr, dtype=dtype).to(self.device)


class BoostedPPO:
    def __init__(self, model, state_dim, action_dim, learner_args,
                 n_estimators=50, max_features=0.80, **model_kwargs):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.model = model
        self.model_kwargs = model_kwargs
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.learner_args = learner_args
        self.max_loss_for_weight = learner_args.get(
            'max_loss_for_weight', None
        )
        self.fixed_model_weight = learner_args.get('fixed_model_weight', 0)
        self.clip_sample_weights = learner_args.get(
            'clip_sample_weights', [0.8, 1.2]
        )
        self.learn_probabilities = learner_args.get('learn_probabilities', 0)
        self.estimators = []
        
        if isinstance(self.max_features, float):
            self.max_features = int(self.state_dim * self.max_features)

    def train_net(self):
        sample_weights = np.ones((len(self.train), 1))
        for i in range(self.n_estimators):
            print(f"Estimator {i+1} of {self.n_estimators}.")
            state_mask = np.array([False] * self.state_dim)
            state_mask[
                np.random.choice(
                    np.arange(self.state_dim),
                    size=self.max_features,
                    replace=False
                )
            ] = True

            model = self.model(
                np.sum(state_mask), self.action_dim, self.learner_args,
                sample_weights=sample_weights, state_mask=state_mask,
                **self.model_kwargs
            ).to(self.learner_args['device'])

            # insert train data and train model
            model.train = self.train
            model.train_net()

            # eval model on train for per sample loss
            loss_per_sample = []
            for idx, row in self.train.iterrows():
                s, rs, s_prime = (
                    model._to_tensor([row["s"]]),
                    model._to_tensor([row["rs"]]),
                    model._to_tensor([row["s'"]])
                )
                train_loss = self._to_numpy(
                    model.calc_loss(s, rs, s_prime, idx)
                )
                loss_per_sample.append(train_loss)

            # eval model on val for model weight
            model_loss = None
            for idx, row in self.val.iterrows():
                s, rs, s_prime = (
                    model._to_tensor([row["s"]]),
                    model._to_tensor([row["rs"]]),
                    model._to_tensor([row["s'"]])
                )
                val_loss = self._to_numpy(model.calc_loss(s, rs, s_prime))
                if isinstance(model_loss, type(None)):
                    model_loss = val_loss
                else:
                    model_loss += val_loss

            model_loss = (model_loss / len(self.val))[0]
            if not self.fixed_model_weight:
                if self.max_loss_for_weight is not None:
                    model_weight = 0.5 * np.log(
                        (
                            (2 * self.max_loss_for_weight) - model_loss
                        ) / model_loss
                    )
                    model_weight = max(model_weight, 1e-9)
                else:
                    model_weight = np.exp(-(model_loss + 0.1)) / (
                        1 - np.exp(-(model_loss + 0.1))
                    )
            else:
                model_weight = 1

            # save estimator to ensemble data
            self.estimators.append({
                'estimator': model,
                'weight': np.array(model_weight),
                'q_value losses': np.array(model_loss),
                'loss per sample': np.array(loss_per_sample),
                'features mask': state_mask
            })

            # calculate metrics for progress logging
            total_val_loss = np.zeros((1,))
            total_weights = np.zeros((1,))
            for estimator_data in self.estimators:
                total_val_loss += (
                    estimator_data['q_value losses'] * estimator_data['weight']
                )
                total_weights += estimator_data['weight']

            weighted_training_loss = np.average(
                np.array(loss_per_sample), axis=0
            )[0]
            training_loss = np.mean(
                loss_per_sample / sample_weights, axis=0
            )[0]
            validation_loss = (
                np.array(total_val_loss) / np.array(total_weights)
            )
            print("\tEstimator:")
            print(
                f"\tweighted training loss: {weighted_training_loss:.4f} "
                f"training loss: {training_loss:.4f} "
                f"validation loss: {np.array(model_loss):.4f} "
                f"weight: {np.array(model_weight):.4f} "
            )
            print(f"Ensemble validation loss: {validation_loss[0]:.4f}")

            # calc loss of ensemble for new sample_weights
            ensemble_loss_per_sample = self.calc_ensemble_loss_per_sample()
            sample_weights = np.clip(
                ensemble_loss_per_sample / np.mean(
                    ensemble_loss_per_sample,
                    axis=0
                ),
                self.clip_sample_weights[0], self.clip_sample_weights[1]
            )

    def calc_ensemble_loss_per_sample(self):
        model_weights = np.array([
            estimator_data['weight']
            for estimator_data in self.estimators
        ])  # (n_estimators,)
        model_weights = model_weights / np.sum(model_weights, axis=0)
        loss_per_sample_per_estimator = np.array([
            estimator_data['loss per sample']
            for estimator_data in self.estimators
        ])  # (n_estimators, 1, n_samples)

        return np.sum(
            np.einsum(
                'i...,i->i...',
                loss_per_sample_per_estimator,  # (n_estimators, 1, n_samples)
                model_weights  # (n_estimators,)
            ),
            axis=0
        )

    def calc_probs_per_estimator(self, s):
        probs_list = []
        model_weights = []
        for estimator_data in self.estimators:
            _, probs = estimator_data['estimator'].choose_action(s)
            model_weights.append(estimator_data['weight'])
            probs_list.append(self._to_numpy(probs))
        return np.array(probs_list), np.array(model_weights)

    def calc_mean_of_probs(self, s):
        probs_per_estimator, model_weights = self.calc_probs_per_estimator(s)
        decision_per_estimator = (
            np.equal(
                probs_per_estimator,
                probs_per_estimator.max(axis=1)[:, np.newaxis]
            )
        ).astype(float)
        return np.matmul(
            model_weights, decision_per_estimator
        ) / np.sum(model_weights)

    def calc_mean_sd_of_probs(self, s, mask_max_prob=False):
        probs_per_estimator, model_weights = self.calc_probs_per_estimator(s)
        if mask_max_prob:
            decision_per_estimator = (
                np.equal(
                    probs_per_estimator,
                    probs_per_estimator.max(axis=1)[:, np.newaxis]
                )
            ).astype(float)
            probs_per_estimator = decision_per_estimator
        average = np.matmul(
            model_weights, probs_per_estimator
        ) / np.sum(model_weights)
        variance = np.average(
            (probs_per_estimator-average) ** 2,
            axis=0,
            weights=model_weights
        )
        return average, np.sqrt(variance)

    def get_feature_importances(self, use_estimator_weights=True):
        feature_weights = []
        for estimator in self.estimators:
            tree_weights = estimator['estimator'].cdt.get_tree_weights()
            fl_weights = estimator['estimator'].cdt.get_fl_input_weights()

            dc_tree_imp = np.mean(np.abs(tree_weights[1]), axis=0)
            feat_tree_imp = dc_tree_imp[np.newaxis, :]
            for i in range(-1, -len(fl_weights)-1, -1):
                fl_leaf_imp = np.mean(np.abs(fl_weights[i]), axis=0)
                fl_feat_imp = feat_tree_imp @ fl_leaf_imp
                fl_tree_imp = np.mean(np.abs(tree_weights[0][i]), axis=0)
                feat_tree_imp = fl_feat_imp * fl_tree_imp

            feat_imp = feat_tree_imp.flatten()
            feat_imp /= np.sum(feat_imp)

            if use_estimator_weights:
                feat_imp *= estimator['weight']
            # expand back to state_dim using state_mask
            feat_imp_full = np.zeros(
                estimator['features mask'].shape
            )
            np.place(  # place is inplace
                feat_imp_full,
                estimator['features mask'],
                feat_imp
            )
            feature_weights.append(feat_imp_full)

        sum_feature_weights = np.sum(np.array(feature_weights), axis=0)
        # normalize at the ensemble level
        return sum_feature_weights / np.sum(sum_feature_weights)

    def _to_numpy(self, torch_tensor):
        return np.copy(
            torch_tensor.detach().cpu().numpy()
        )


def run(
    train_env, learner_args, train=False, test=False,
    train_ratio=0.80, n_estimators=30, max_features=0.80,
):
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    learner_args["input_dim"] = state_dim
    learner_args["output_dim"] = action_dim

    print(state_dim, action_dim)
    model = BoostedPPO(
        PPO,  # model
        state_dim=state_dim, action_dim=action_dim, learner_args=learner_args,
        n_estimators=n_estimators, max_features=max_features,  # boosting args
    )
    data = []

    s = train_env.reset()
    done = False
    while not done:
        if train:
            a, q_vals = None, None
            s_prime, rs, done, _ = train_env.step_all()
            data.append([s, rs, s_prime, q_vals, a])
        else:
            q_vals = model.calc_mean_of_q_vals(s)
            a = np.argmax(q_vals)
            s_prime, r, done, a = train_env.step(a)
            data.append([s, r, s_prime, q_vals, a])
        if done:
            break
        s = s_prime

    data = pd.DataFrame(data, columns=["s", "rs", "s'", "Qs", "a"])

    if train:
        model.train = data.iloc[: int(len(data) * train_ratio)]
        model.val = data.iloc[int(len(data) * train_ratio): -1]
        model.train_net()

    train_env.close()
    return model, data