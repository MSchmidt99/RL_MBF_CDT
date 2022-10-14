"""
PPO with cascading decision tree (CDT) as policy function approximator

Modified from the following:
https://github.com/quantumiracle/Cascading-Decision-Tree/blob/d660c442175c3a05bc70c1a45eb3eb9d91242260/src/cdt/deprecated/cdt_rl_train.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.distributions import Categorical
import numpy as np
import pandas as pd
from .CDT import CDT
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
        self.loss_batch_size = learner_args.get('loss_batch_size', 1)

        self.sample_weights = (
            torch.tensor(sample_weights, dtype=torch.float).to(self.device)
            if not isinstance(sample_weights, type(None)) else None
        )
        if not isinstance(state_mask, type(None)):
            self.state_mask = (
                torch.tensor(state_mask, dtype=torch.bool).to(self.device)
            )
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
        loss_batch = []
        for epoch in range(self.K_epoch):
            for i, row in batch.iterrows():
                s, rs, s_prime = (
                    self._to_tensor([row["s"]]),
                    self._to_tensor([row["rs"]]),
                    self._to_tensor([row["s'"]])
                )
                loss = self.calc_loss(s, rs, s_prime, i=i)
                loss_batch.append(loss)
                
                if i % self.loss_batch_size == 0 or i == len(batch) - 1:
                    self.optimizer.zero_grad()
                    torch.cat(loss_batch, 0).mean().backward()
                    loss_batch = []
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

        # orient loss so minimum is 0, only effects learn_probabilities
        loss -= self.softXEnt(td_target, td_target)

        # mask loss for largest decision signal in this sample
        loss = torch.sum(loss * max_target_mask, dim=1)
        if isinstance(self.sample_weights, torch.Tensor) and i is not None:
            # apply sample weights
            loss *= self.sample_weights[i]

        return loss

    def softXEnt(self, input, target):
        # https://www.desmos.com/calculator/zlzedjdfif
        # log will only be nan when input is 0 or 1
        logprobs = torch.log(input + 1e-44)
        inv_logprobs = torch.log(1 - input + 1e-44)
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
        self.fixed_model_weight = learner_args.get('fixed_model_weight', 0)
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
            loss_per_sample = [
                self._to_numpy(
                    model.calc_loss(
                        model._to_tensor([row["s"]]),
                        model._to_tensor([row["rs"]]),
                        model._to_tensor([row["s'"]]),
                        idx
                    )
                )
                for idx, row in self.train.iterrows()
            ]
            train_loss_unweighted = loss_per_sample / sample_weights

            # eval model on val for model weight
            val_loss = np.sum([
                self._to_numpy(
                    model.calc_loss(
                        model._to_tensor([row["s"]]),
                        model._to_tensor([row["rs"]]),
                        model._to_tensor([row["s'"]])
                    )
                )
                for idx, row in self.val.iterrows()
            ], axis=0)
            val_loss = (val_loss / len(self.val))[0]

            # remove effect of sample weights then mean
            training_loss = np.mean(
                train_loss_unweighted, axis=0
            )[0]
            
            # calc loss of ensemble for new sample_weights
            EPS = 1e-10
            # loss function used ranges from [0,inf)
            # thus alpha calculation must be modified.
            alpha = 0.5 * np.log(
                1 / np.minimum(training_loss + EPS, 1)
            )
            sample_weights *= np.exp(-alpha)
            sample_weights /= np.mean(sample_weights)
            model_weight = alpha if not self.fixed_model_weight else 1

            # save estimator to ensemble data
            self.estimators.append({
                'estimator': model,
                'weight': np.array(model_weight),
                'q_value losses': np.array(val_loss),
                'loss per sample': np.array(train_loss_unweighted),
                'features mask': state_mask
            })
            
            # --------------------------------------
            # calculate metrics for progress logging
            total_val_loss = np.zeros((1,))
            total_weights = np.zeros((1,))
            for estimator_data in self.estimators:
                total_val_loss += (
                    estimator_data['q_value losses'] * estimator_data['weight']
                )
                total_weights += estimator_data['weight']

            weighted_training_loss = np.mean(
                np.array(loss_per_sample), axis=0
            )[0]
            validation_loss = (
                np.array(total_val_loss) / np.array(total_weights)
            )
            print("\tEstimator:")
            print(
                f"\tweighted training loss: {weighted_training_loss:.4f} "
                f"training loss: {training_loss:.4f} "
                f"validation loss: {np.array(val_loss):.4f} "
                f"weight: {np.array(model_weight):.4f} "
            )
            print(f"Ensemble validation loss: {validation_loss[0]:.4f}")

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
            feature_learner_weights = estimator[
                'estimator'
            ].cdt.get_tree_weights()[0]
            sum_feature_learner_weights = np.sum(
                np.abs(feature_learner_weights),
                axis=0
            )
            # normalize at the estimator level
            sum_feature_learner_weights = (
                sum_feature_learner_weights /
                np.sum(sum_feature_learner_weights)
            )
            if use_estimator_weights:
                sum_feature_learner_weights *= estimator['weight']
            # expand back to state_dim using state_mask
            feature_learner_weights_full = np.zeros(
                estimator['features mask'].shape
            )
            np.place(  # place is inplace
                feature_learner_weights_full,
                estimator['features mask'],
                sum_feature_learner_weights
            )
            feature_weights.append(feature_learner_weights_full)

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