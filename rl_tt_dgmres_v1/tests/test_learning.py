"""Behavioral checks for finite-episode Double-DQN, without solver training."""

import copy

import numpy as np
import pytest
import torch

from rl_tt_dgmres_v1.learning import QAgent, bellman_targets


def make_agent(**kwargs):
    options = dict(observation_dim=3, actions=[0.25, 0.5, 1.0], seed=17,
                   batch_size=4, capacity=16, hidden=8, target_update=3)
    options.update(kwargs)
    return QAgent(**options)


def populate(agent, count=12):
    for index in range(count):
        state = [index / 12.0, 0.5, -0.2]
        agent.observe(state, index % 3, -2.0 - index / 10.0,
                      [state[0] + 0.1, 0.4, -0.1], index % 4 == 3)


def test_double_dqn_uses_online_selection_and_masks_terminal():
    rewards = torch.tensor([-2.0, -3.0, -4.0], requires_grad=True)
    dones = torch.tensor([False, True, False])
    online = torch.tensor([[9.0, 1.0], [0.0, 2.0], [1.0, 3.0]], requires_grad=True)
    target = torch.tensor([[5.0, 100.0], [7.0, float("inf")], [100.0, 8.0]], requires_grad=True)
    result = bellman_targets(rewards, dones, online, target)
    torch.testing.assert_close(result, torch.tensor([3.0, -3.0, 4.0]))
    assert not result.requires_grad


def test_seed_is_local_and_greedy_does_not_advance_rng():
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    agent = make_agent()
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state)
    after_numpy_state = np.random.get_state()
    np.testing.assert_array_equal(after_numpy_state[1], numpy_state[1])
    assert after_numpy_state[2:] == numpy_state[2:]
    before = copy.deepcopy(agent.rng.bit_generator.state)
    actions = [agent.select_action([0.0, 0.0, 0.0]) for _ in range(8)]
    assert len(set(actions)) == 1
    assert before == agent.rng.bit_generator.state
    assert not agent.online.training
    assert all(parameter.grad is None for parameter in agent.online.parameters())


def test_epsilon_exploration_is_reproducible_and_covers_actions():
    first, second = make_agent(), make_agent()
    choices = [first.select_action([1.0, 0.0, 0.0], epsilon=1.0) for _ in range(100)]
    assert choices == [second.select_action([1.0, 0.0, 0.0], epsilon=1.0) for _ in range(100)]
    assert set(choices) == {0, 1, 2}
    with torch.no_grad():
        for parameter in first.online.parameters():
            parameter.zero_()
    assert first.select_action([0.0, 0.0, 0.0]) == 0


def test_training_updates_parameters_and_syncs_target():
    agent = make_agent()
    assert agent.train_steps(2)["updates"] == 0
    populate(agent)
    before = [parameter.detach().clone() for parameter in agent.online.parameters()]
    stats = agent.train_steps(3)
    assert stats["updates"] == 3 and stats["gradient_steps"] == 3
    assert np.isfinite(stats["loss_mean"]) and stats["loss_mean"] > 0
    assert any(not torch.equal(old, new) for old, new in zip(before, agent.online.parameters()))
    for online, target in zip(agent.online.parameters(), agent.target.parameters()):
        torch.testing.assert_close(online, target)
        assert target.grad is None and not target.requires_grad


def test_replay_is_copied_and_inference_collection_can_train():
    agent = make_agent()
    with torch.inference_mode():
        state = torch.tensor([0.1, 0.2, 0.3])
        for _ in range(4):
            action = agent.select_action(state)
            agent.observe(state, action, -1.0, state, True)
        state.fill_(99)
        with pytest.raises(RuntimeError, match="after the solver"):
            agent.train_steps()
    np.testing.assert_allclose(agent.replay.states[:4], [[0.1, 0.2, 0.3]] * 4)
    assert agent.replay.states.dtype == np.float32
    assert agent.train_steps(2)["updates"] == 2


def test_save_load_preserves_policy_metadata_and_exact_training_resume(tmp_path):
    agent = make_agent()
    populate(agent, 25)  # Exercise a wrapped ring, not just an initial prefix.
    agent.train_steps(2)
    path = tmp_path / "learner.pt"
    agent.save(path, metadata={"actions": agent.actions, "experiment": "unit-test"})
    restored = QAgent.load(path)
    assert restored.actions == agent.actions
    assert restored.metadata["experiment"] == "unit-test"
    assert restored.gradient_steps == 2
    assert restored.replay.position == agent.replay.position
    assert len(restored.replay) == len(agent.replay)
    probes = [[0.0, 0.1, 0.2], [2.0, 0.0, -1.0], [-1.0, 3.0, 2.0]]
    assert [agent.select_action(s) for s in probes] == [restored.select_action(s) for s in probes]
    assert agent.train_steps(4) == restored.train_steps(4)
    for original, loaded in zip(agent.online.parameters(), restored.online.parameters()):
        torch.testing.assert_close(original, loaded, rtol=0, atol=0)
    assert [agent.select_action(probes[0], 1.0) for _ in range(20)] == [
        restored.select_action(probes[0], 1.0) for _ in range(20)]


def test_metadata_action_order_cannot_drift(tmp_path):
    agent = make_agent()
    path = tmp_path / "learner.pt"
    with pytest.raises(ValueError, match="metadata actions"):
        agent.save(path, {"actions": list(reversed(agent.actions))})
    agent.save(path)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["metadata"]["actions"] = list(reversed(agent.actions))
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="metadata actions"):
        QAgent.load(path)


@pytest.mark.parametrize("bad_observation", [[float("nan"), 0.0, 0.0], [0.0], [1e300, 0.0, 0.0]])
def test_nonfinite_or_wrong_shape_observations_rejected(bad_observation):
    agent = make_agent()
    with pytest.raises(ValueError, match="observation"):
        agent.select_action(bad_observation)
    with pytest.raises(ValueError, match="observation"):
        agent.observe([0.0] * 3, 0, 0.0, bad_observation, False)
    assert len(agent.replay) == 0


@pytest.mark.parametrize("bad_action", [-1, 3, 0.5, True])
def test_invalid_action_indices_rejected(bad_action):
    with pytest.raises(ValueError, match="action index"):
        make_agent().observe([0.0] * 3, bad_action, 0.0, [0.0] * 3, False)


@pytest.mark.parametrize("bad_reward", [float("nan"), float("inf"), 1e300])
def test_invalid_rewards_rejected(bad_reward):
    with pytest.raises(ValueError, match="reward"):
        make_agent().observe([0.0] * 3, 0, bad_reward, [0.0] * 3, False)


@pytest.mark.parametrize("bad_epsilon", [-0.1, 1.1, float("nan")])
def test_invalid_epsilon_rejected(bad_epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        make_agent().select_action([0.0] * 3, bad_epsilon)
