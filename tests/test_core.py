import pytest
from src.agent import LinUCB
from src.utils import calculate_reward

def test_linucb_initialization():
    agent = LinUCB(n_arms=3, n_features=5)
    assert agent.n_arms == 3
    assert len(agent.A_inv) == 3
    assert agent.A_inv[0].shape == (5, 5)

def test_linucb_select_arm():
    agent = LinUCB(n_arms=2, n_features=2)
    context = [1.0, 0.5]
    arm = agent.select_arm(context)
    assert arm in [0, 1]

def test_calculate_reward_success():
    # base=100, comp=60, latency=500ms, valid=True
    reward, saving, lat_pen, fail_pen = calculate_reward(100, 60, 500, True)
    assert saving == 0.4
    assert lat_pen == 0.1
    assert fail_pen == 0.0
    # 1.5 * 0.4 - 0.1 - 0 = 0.5
    assert reward == pytest.approx(0.5)

def test_calculate_reward_failure():
    reward, saving, lat_pen, fail_pen = calculate_reward(100, 60, 500, False)
    assert fail_pen == 2.5
    assert reward == pytest.approx(1.5 * 0.4 - 0.1 - 2.5)
