# drl/agent/config.py
from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass(frozen=True)
class RecurrentPPOConfig:
    """
    Recurrent PPO configurations.
    """
    n_steps: int
    epoch: int
    seq_len: int
    seq_mini_batch_size: int
    padding_value: float = 0.0
    gamma: float = 0.99
    lam: float = 0.95
    epsilon_clip: float = 0.2
    critic_loss_coef: float = 0.5
    entropy_coef: float = 0.001

@dataclass(frozen=True)
class RecurrentPPORNDConfig:
    """
    Recurrent PPO with RND configurations.
    """
    n_steps: int
    epoch: int
    seq_len: int
    seq_mini_batch_size: Optional[int] = None
    padding_value: float = 0.0
    gamma: float = 0.99
    gamma_n: float = 0.99 # Discount factor for non-episodic (intrinsic) rewards
    nonepi_adv_coef: float = 1.0 # Coefficient for non-episodic advantage
    lam: float = 0.95
    epsilon_clip: float = 0.2
    critic_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    rnd_pred_exp_proportion: float = 0.25
    init_norm_steps: Optional[int] = 50
    obs_norm_clip_range: Tuple[float, float] = (-5.0, 5.0)
    hidden_state_norm_clip_range: Tuple[float, float] = (-5.0, 5.0)

# --- New SAC Configuration ---
@dataclass(frozen=True)
class RecurrentSACConfig:
    """
    Recurrent Soft Actor-Critic (SAC) configurations.
    """
    # --- SAC Specific ---
    gamma: float = 0.99 # Discount factor
    tau: float = 0.005 # Coefficient for soft update of target networks
    alpha: float = 0.2 # Initial entropy regularization coefficient
    actor_lr: float = 3e-4 # Learning rate for the actor network
    critic_lr: float = 3e-4 # Learning rate for the critic network(s)
    alpha_lr: float = 3e-4 # Learning rate for the entropy coefficient alpha (if learned)
    target_update_interval: int = 1 # Frequency of target network updates (in terms of gradient steps)
    learn_alpha: bool = True # Whether to automatically tune the entropy coefficient alpha
    target_entropy_ratio: float = 0.98 # Target entropy ratio (relative to max entropy: -log(1/|A|)*ratio)

    # --- Buffer & Training ---
    buffer_size: int = 100000 # Size of the replay buffer
    batch_size: int = 256 # Minibatch size for sampling from the replay buffer
    learning_starts: int = 1000 # Number of steps to collect before starting training
    gradient_steps: int = 1 # Number of gradient updates per environment step

    # --- Recurrent Specific (Adjust based on network needs) ---
    # seq_len might not be needed as SAC learns from transitions, not sequences like PPO
    # seq_len: int = 1

    # --- Common Agent Params (May have different meaning/usage in SAC) ---
    n_steps: int = 1 # Number of environment steps per agent interaction cycle (often 1 for SAC)

    # --- Optional RND Integration (If combining SAC with RND) ---
    # Add RND-specific parameters here if needed, similar to RecurrentPPORNDConfig
    # gamma_n: float = 0.99
    # nonepi_adv_coef: float = 1.0
    # rnd_pred_exp_proportion: float = 0.25
    # init_norm_steps: Optional[int] = 50
    # obs_norm_clip_range: Tuple[float, float] = (-5.0, 5.0)
    # hidden_state_norm_clip_range: Tuple[float, float] = (-5.0, 5.0)
