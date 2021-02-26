import gym
import pytest
from gym.wrappers import FrameStack
from stable_baselines3 import A2C

from agent import LearningAgent
from data_generation.experience import Experience
from data_generation.preference_data_generator import PreferenceDataGenerator
from data_generation.preference_label import PreferenceLabel
from reward_modeling.choice_model import ChoiceModel
from reward_modeling.reward_model import RewardModel
from reward_modeling.reward_wrapper import RewardWrapper


@pytest.fixture()
def cartpole_env():
    env = gym.make('CartPole-v1')
    env = FrameStack(env, num_stack=4)
    return env


@pytest.fixture(params=('CartPole-v1', 'Pong-v0'))
def env(request):
    env_id = request.param
    env = gym.make(env_id)
    env = FrameStack(env, num_stack=4)
    return env


@pytest.fixture()
def reward_model(cartpole_env):
    return RewardModel(cartpole_env)


@pytest.fixture()
def reward_wrapper(cartpole_env, reward_model):
    return RewardWrapper(env=cartpole_env, reward_model=reward_model, trajectory_buffer_size=100)


@pytest.fixture()
def learning_agent(reward_wrapper):
    return LearningAgent(reward_wrapper, segment_length=4, simulation_steps_per_policy_update=5)


@pytest.fixture()
def segment_samples():
    segment_1 = [Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 1}),
                 Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 1})]
    segment_2 = [Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 25}),
                 Experience(observation=1, action=1, reward=1, done=1, info={"original_reward": 25})]

    return [segment_1, segment_2]


@pytest.fixture()
def preference_data_generator(policy_model):
    return PreferenceDataGenerator(policy_model=policy_model, segment_length=3)


@pytest.fixture()
def policy_model(reward_wrapper):
    return A2C('MlpPolicy', env=reward_wrapper, n_steps=10)


@pytest.fixture()
def preference(env):
    # TODO: Return a fixed segment (without running the env!) to make it faster and deterministic
    segment_length = 6
    experiences = []
    env.reset()
    for i in range(segment_length * 2):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        experiences.append(Experience(observation, action, reward, done, info))
    query = [experiences[:segment_length], experiences[segment_length:]]
    return query, PreferenceLabel.LEFT


@pytest.fixture()
def choice_model(reward_model):
    return ChoiceModel(reward_model)
