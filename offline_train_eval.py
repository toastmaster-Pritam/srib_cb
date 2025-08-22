# offline_train_eval.py

import numpy as np
from obp.dataset import SyntheticBanditDataset
from obp.policy import UniformPolicy
from obp.ope import InverseProbabilityWeighting, DoublyRobust, RegressionModel

# ==== 1. Generate Synthetic Bandit Dataset ====
n_actions = 5
dim_context = 5
n_rounds = 10000

dataset = SyntheticBanditDataset(
    n_actions=n_actions,
    dim_context=dim_context,
    reward_type="continuous",
    reward_function=None,
    random_state=12345,
)

bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)

# ==== 2. Define Composite Reward Function ====
def compute_composite_reward(reward_matrix, decay=0.9, horizon=10):
    """
    reward_matrix: shape (n_rounds, horizon)
        short-term rewards for each day
    decay: float
        exponential decay applied across days
    horizon: int
        number of days considered
    """
    n_rounds = reward_matrix.shape[0]
    composite = np.zeros(n_rounds)
    for i in range(n_rounds):
        decayed_sum = 0.0
        for t in range(horizon):
            decayed_sum += (decay ** t) * reward_matrix[i, t]
        composite[i] = decayed_sum
    return composite

# Create synthetic "10-day" short-term rewards
short_term_rewards = np.random.rand(n_rounds, 10)
long_term_rewards = compute_composite_reward(short_term_rewards)

# Replace reward in bandit feedback with long-term composite reward
bandit_feedback["reward"] = long_term_rewards
bandit_feedback["position"] = np.zeros(n_rounds, dtype=int)  # needed for some OPE estimators

# ==== 3. Define Baseline Policy ====
uniform_policy = UniformPolicy(n_actions=n_actions, random_state=42)
action_dist = uniform_policy.compute_batch_action_dist(n_rounds=n_rounds)

# ==== 4. Train Reward Regressor ====
reg_model = RegressionModel(
    n_actions=n_actions,
    base_model="logistic",  # you can also try "linear" or "random_forest"
    random_state=42,
)
estimated_rewards = reg_model.fit_predict(
    context=bandit_feedback["context"],
    action=bandit_feedback["action"],
    reward=bandit_feedback["reward"],
    n_folds=3,
    action_dist=action_dist,
)

# ==== 5. Offline Policy Evaluation ====
ipw = InverseProbabilityWeighting()
dr = DoublyRobust()

ipw_value = ipw.estimate_policy_value(
    reward=bandit_feedback["reward"],
    action=bandit_feedback["action"],
    pscore=bandit_feedback["pscore"],
    action_dist=action_dist,
)

dr_value = dr.estimate_policy_value(
    reward=bandit_feedback["reward"],
    action=bandit_feedback["action"],
    pscore=bandit_feedback["pscore"],
    action_dist=action_dist,
    estimated_rewards_by_reg_model=estimated_rewards,
)

# ==== 6. Print Results ====
print("=== Offline Policy Evaluation Results ===")
print(f"IPW Estimated Policy Value: {ipw_value:.4f}")
print(f"DR Estimated Policy Value:  {dr_value:.4f}")