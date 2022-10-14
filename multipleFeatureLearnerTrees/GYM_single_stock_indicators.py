import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
# https://github.com/bukosabino/ta
from ta import add_all_ta_features
import warnings


class TimedTrailingStop:
    """
    Example derivative_type for a trailing stop loss timed order.

    This order functions as follows:
     - When make_order is called with short or long order type it
       will enter a position with the initialized stop_loss.
     - If the position increases in value, the stop_loss will lag
       the price by 1-stop_loss percent.
     - If the price goes below the stop_loss the position will close.
     - If the end of the lookahead is reached the order will close.
    """
    def __init__(self, stop_loss=0.95):
        self.stop_loss = stop_loss
    
    def make_order(self, lookahead, order_type):
        # lookahead is scaled by 1 / lookahead.iloc[0]['Close'],
        # therefore each price at the following time steps are exactly
        # the returns of the position if it is promptly closed.
        # order_type 1 -> long position, -1 -> short position
        maximum_profit_col = (
            'High' * (order_type == 1) +
            'Low' * (order_type == -1)
        )
        maximum_loss_col = (
            'Low' * (order_type == 1) +
            'High' * (order_type == -1)
        )

        trailing_stop_loss = self.stop_loss
        max_profit_seen = 1
        for i, row in lookahead.iterrows():
            # Based on order type, calculate returns at extrema of
            # the current step.
            max_loss_at_step = row[maximum_loss_col] ** order_type
            max_profit_at_step = row[maximum_profit_col] ** order_type
            # If trailing_stop_loss is reached then close the position.
            if max_loss_at_step < trailing_stop_loss:
                return max_loss_at_step
            # Set max_profit_seen to max_profit_at_step if it is the
            # largest profit seen, then adjust trailing_stop_loss to
            # follow it.
            elif max_profit_at_step > max_profit_seen:
                max_profit_seen = max_profit_at_step
                trailing_stop_loss = max_profit_seen - (1 - self.stop_loss)
        # If the trailing_stop_loss has not been hit through the duration
        # of the position, then position will be closed at the returns of
        # lookahead.iloc[-1]['Close'].
        return lookahead.iloc[-1]['Close']


class DerivativesTradingEnv(gym.Env):
    """A derivatives trading environment for OpenAI gym"""

    def __init__(
        self, df, train_end_ratio=None, n_lookback=30,
        trade_length=5, derivative_type=TimedTrailingStop(),
        ignore_indicators=[]
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.ignore_indicators = ignore_indicators
        self.train_end_idx = (
            int( (len(df) - n_lookback) * train_end_ratio ) + n_lookback
            if not train_end_ratio is None
            else None
        )
        self.test = False
        self.n_lookback = n_lookback
        self.trade_length = trade_length # action space? (0, np.inf)
        self.derivative_type = derivative_type

        # Actions of the format [Q-Sell, Q-Do nothing (=0), Q-Buy]
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float16
        )

        self.observation_cols = self.reset(return_pandas=True).index
        observation_len = len(self.observation_cols)
        
        observation_bounds = {
            "low": np.array([-np.inf] * observation_len),
            "high": np.array([np.inf] * observation_len)
        }
        self.observation_space = spaces.Box(
            low=observation_bounds['low'],
            high=observation_bounds['high'],
            shape=(observation_len,),
            dtype=np.float16
        )

    def _next_observation(self, return_pandas=False):
        # Get the stock data points for the last n_lookback steps
        # and scale data
        return self._observation_calculations(
            self.df.loc[
                self.current_step - self.n_lookback : self.current_step
            ].copy(),
            return_pandas=return_pandas
        )

    def _take_action(self, action):
        if action == 1:
            return 0
        # current_price used to normalize lookahead prices passed
        # to derivative_type.make_order
        current_price = self.df.loc[self.current_step]["Close"]
        action_type = action - 1
        
        lookahead = self.df.loc[
            self.current_step : self.current_step + self.trade_length
        ]
        # long or short the stock using the make_order which
        # corresponds to the derivative_type used.
        # returned as price of closing order / order open price
        return (self.derivative_type.make_order(
            lookahead / current_price,
            order_type=action_type
        ) - 1) * 100

    def step(self, action):
        # Execute one time step within the environment
        reward = self._take_action(action)
        
        self.current_step += 1
        if not (self.train_end_idx is None) and not self.test:
            done = self.current_step > self.train_end_idx
        else:
            done = self.current_step > (
                len(self.df) - self.trade_length
            )
        obs = self._next_observation() if not done else None

        return obs, reward, done, {}
    
    def step_all(self):
        # With derivatives we are in a lucky spot where the
        # previous decision does not impact the returns of
        # the next decision (bar the effect that the trade
        # had on the market), thus we can use this step_all
        # method to build out a lookup table of the rewards
        # for each possible decision.
        reward0 = self._take_action(0)
        reward1 = self._take_action(1)
        reward2 = self._take_action(2)

        self.current_step += 1
        if not self.train_end_idx is None and not self.test:
            done = self.current_step > self.train_end_idx
        else:
            done = self.current_step > (
                len(self.df) - self.trade_length
            )
        obs = self._next_observation() if not done else None

        return obs, [reward0, reward1, reward2], done, {}

    def reset(self, test=False, return_pandas=False):
        self.test = test
        # Set the current step to 0
        if not (self.train_end_idx is None) and self.test:
            self.current_step = self.train_end_idx
        else:
            self.current_step = 0 + self.n_lookback
        return self._next_observation(return_pandas=return_pandas)
    
    def _observation_calculations(self, frame, return_pandas=False):
        frame = frame.reset_index(drop=True)
        frame.index -= self.n_lookback
        # normalize
        frame[["Open", "High", "Low", "Close"]] /= frame.loc[0]['Close']
        frame["Volume"] /= frame.loc[0]['Volume']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = add_all_ta_features(
                frame,
                open="Open", high="High",
                low="Low", close="Close",
                volume="Volume"
            ).loc[0].fillna(0)
        # since Close[0] and Volume[0] are normalized by themselves,
        # they are both always 1.0.
        frame = frame.drop(["Close", "Volume"])
        if len(self.ignore_indicators) > 0:
            frame = frame.drop(self.ignore_indicators)
        if return_pandas:
            return frame
        return frame.values

    def get_state_column_names(self):
        return self.observation_cols
