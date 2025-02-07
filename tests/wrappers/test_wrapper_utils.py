import gym
import numpy as np
import pytest
from stable_baselines3.common.atari_wrappers import AtariWrapper

from models.reward.mlp import MlpRewardModel
from wrappers.external.indirect_feedback_remover import IndirectFeedbackRemover
from wrappers.internal.reward_predictor import RewardPredictor
from wrappers.internal.reward_standardizer import RewardStandardizer
from wrappers.utils import add_external_env_wrappers, create_env, is_atari_env, is_wrapped, add_internal_env_wrappers


@pytest.fixture()
def envs():
    cartpole_env = create_env('CartPole-v1', termination_penalty=0)
    pong_env = create_env('Pong-v0', termination_penalty=0)
    return cartpole_env, pong_env


def test_converts_to_stacked_env():
    env = gym.make('CartPole-v1')
    frame_stack_depth = 5
    shp = env.observation_space.shape
    env = add_external_env_wrappers(env, frame_stack_depth=frame_stack_depth, termination_penalty=0.)

    assert len(env.observation_space.shape) == len(np.hstack([frame_stack_depth, shp]))
    assert np.all(env.observation_space.shape == np.hstack([frame_stack_depth, shp]))


def test_does_not_convert_to_stacked_env():
    env = gym.make('CartPole-v1')
    frame_stack_depth = 0
    env = add_external_env_wrappers(env, frame_stack_depth=frame_stack_depth, termination_penalty=0.)

    assert not is_wrapped(env, gym.wrappers.FrameStack)


def test_is_atari_env(envs):
    cartpole_env, pong_env = envs
    assert not is_atari_env(cartpole_env)
    assert is_atari_env(pong_env)


def test_is_wrapped(envs):
    cartpole_env, pong_env = envs
    assert not is_wrapped(cartpole_env, AtariWrapper)
    assert is_wrapped(pong_env, AtariWrapper)


def test_wrap_external_environment(envs):
    _, pong_env = envs
    assert is_wrapped(pong_env, AtariWrapper)
    assert is_wrapped(pong_env, IndirectFeedbackRemover)


def test_wrap_internal_environment(cartpole_env):
    reward_model = MlpRewardModel(cartpole_env)

    wrapped_env = add_internal_env_wrappers(cartpole_env, reward_model=reward_model)

    assert is_wrapped(wrapped_env, RewardPredictor) and is_wrapped(wrapped_env, RewardStandardizer)
