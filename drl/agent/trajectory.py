from dataclasses import dataclass
import random
from typing import Optional, List, Tuple , Any, Dict

import torch
import numpy as np

@dataclass(frozen=True)
class RecurrentPPOExperience:
    obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    action_log_prob: torch.Tensor
    state_value: torch.Tensor
    hidden_state: torch.Tensor
    
class RecurrentPPOTrajectory:
    def __init__(self, n_steps: int) -> None:
        self._n_steps = n_steps
        self.reset()
        
    @property
    def reached_n_steps(self) -> bool:
        return self._recent_idx + 1 >= self._n_steps
        
    def reset(self):
        self._recent_idx = -1
        
        self._obs_buffer = self._make_buffer()
        self._action_buffer = self._make_buffer()
        self._reward_buffer = self._make_buffer()
        self._terminated_buffer = self._make_buffer()
        self._action_log_prob_buffer = self._make_buffer()
        self._state_value_buffer = self._make_buffer()
        self._hidden_state_buffer = self._make_buffer()
        
        self._final_next_obs = None
        
    def add(self, exp: RecurrentPPOExperience):
        self._recent_idx += 1
        
        self._obs_buffer[self._recent_idx] = exp.obs
        self._action_buffer[self._recent_idx] = exp.action
        self._reward_buffer[self._recent_idx] = exp.reward
        self._terminated_buffer[self._recent_idx] = exp.terminated
        self._action_log_prob_buffer[self._recent_idx] = exp.action_log_prob
        self._state_value_buffer[self._recent_idx] = exp.state_value
        self._hidden_state_buffer[self._recent_idx] = exp.hidden_state
        
        self._final_next_obs = exp.next_obs
        
    def sample(self) -> RecurrentPPOExperience:
        self._obs_buffer.append(self._final_next_obs)
        exp_batch = RecurrentPPOExperience(
            torch.concat(self._obs_buffer[:-1]),
            torch.concat(self._action_buffer),
            torch.concat(self._obs_buffer[1:]),
            torch.concat(self._reward_buffer),
            torch.concat(self._terminated_buffer),
            torch.concat(self._action_log_prob_buffer),
            torch.concat(self._state_value_buffer),
            torch.concat(self._hidden_state_buffer, dim=1),
        )
        self.reset()
        return exp_batch
        
    def _make_buffer(self) -> list:
        return [None] * self._n_steps
    
@dataclass(frozen=True)
class RecurrentPPORNDExperience:
    obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
    reward: torch.Tensor
    rnd_int_reward: torch.Tensor
    terminated: torch.Tensor
    action_log_prob: torch.Tensor
    epi_state_value: torch.Tensor
    nonepi_state_value: torch.Tensor
    hidden_state: torch.Tensor
    next_hidden_state: torch.Tensor
    
class RecurrentPPORNDTrajectory:
    def __init__(self, n_steps: int) -> None:
        self._n_steps = n_steps
        self.reset()
        
    @property
    def reached_n_steps(self) -> bool:
        return self._recent_idx + 1 >= self._n_steps
        
    def reset(self):
        self._recent_idx = -1
        
        self._obs_buffer = self._make_buffer()
        self._action_buffer = self._make_buffer()
        self._reward_buffer = self._make_buffer()
        self._rnd_int_reward_buffer = self._make_buffer()
        self._terminated_buffer = self._make_buffer()
        self._action_log_prob_buffer = self._make_buffer()
        self._epi_state_value_buffer = self._make_buffer()
        self._nonepi_state_value_buffer = self._make_buffer()
        self._hidden_state_buffer = self._make_buffer()
        
        self._final_next_obs = None
        self._final_next_hidden_state = None
        
    def add(self, exp: RecurrentPPORNDExperience):
        self._recent_idx += 1
        
        self._obs_buffer[self._recent_idx] = exp.obs
        self._action_buffer[self._recent_idx] = exp.action
        self._reward_buffer[self._recent_idx] = exp.reward
        self._rnd_int_reward_buffer[self._recent_idx] = exp.rnd_int_reward
        self._terminated_buffer[self._recent_idx] = exp.terminated
        self._action_log_prob_buffer[self._recent_idx] = exp.action_log_prob
        self._epi_state_value_buffer[self._recent_idx] = exp.epi_state_value
        self._nonepi_state_value_buffer[self._recent_idx] = exp.nonepi_state_value
        self._hidden_state_buffer[self._recent_idx] = exp.hidden_state
        
        self._final_next_obs = exp.next_obs
        self._final_next_hidden_state = exp.next_hidden_state
        
    def sample(self) -> RecurrentPPORNDExperience:
        self._obs_buffer.append(self._final_next_obs)
        self._hidden_state_buffer.append(self._final_next_hidden_state)
        exp_batch = RecurrentPPORNDExperience(
            torch.concat(self._obs_buffer[:-1]),
            torch.concat(self._action_buffer),
            torch.concat(self._obs_buffer[1:]),
            torch.concat(self._reward_buffer),
            torch.concat(self._rnd_int_reward_buffer),
            torch.concat(self._terminated_buffer),
            torch.concat(self._action_log_prob_buffer),
            torch.concat(self._epi_state_value_buffer),
            torch.concat(self._nonepi_state_value_buffer),
            torch.concat(self._hidden_state_buffer[:-1], dim=1),
            torch.concat(self._hidden_state_buffer[1:], dim=1)
        )
        self.reset()
        return exp_batch
    
    def _make_buffer(self) -> list:
        return [None] * self._n_steps # type: ignore

@dataclass(frozen=True)
class RecurrentPPOEpisodicRNDExperience:
    obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
    ext_reward: torch.Tensor
    int_reward: torch.Tensor
    terminated: torch.Tensor
    action_log_prob: torch.Tensor
    state_value: torch.Tensor
    hidden_state: torch.Tensor
    next_hidden_state: torch.Tensor
    
class RecurrentPPOEpisodicRNDTrajectory:
    def __init__(self, n_steps: int) -> None:
        self._n_steps = n_steps
        self.reset()
        
    @property
    def reached_n_steps(self) -> bool:
        return self._recent_idx + 1 >= self._n_steps
        
    def reset(self):
        self._recent_idx = -1
        
        self._obs_buffer = self._make_buffer()
        self._action_buffer = self._make_buffer()
        self._reward_buffer = self._make_buffer()
        self._int_reward_buffer = self._make_buffer()
        self._terminated_buffer = self._make_buffer()
        self._action_log_prob_buffer = self._make_buffer()
        self._state_value_buffer = self._make_buffer()
        self._hidden_state_buffer = self._make_buffer()
        
        self._final_next_obs = None
        self._final_next_hidden_state = None
        
    def add(self, exp: RecurrentPPOEpisodicRNDExperience):
        self._recent_idx += 1
        
        self._obs_buffer[self._recent_idx] = exp.obs
        self._action_buffer[self._recent_idx] = exp.action
        self._reward_buffer[self._recent_idx] = exp.ext_reward
        self._int_reward_buffer[self._recent_idx] = exp.int_reward
        self._terminated_buffer[self._recent_idx] = exp.terminated
        self._action_log_prob_buffer[self._recent_idx] = exp.action_log_prob
        self._state_value_buffer[self._recent_idx] = exp.state_value
        self._hidden_state_buffer[self._recent_idx] = exp.hidden_state
        
        self._final_next_obs = exp.next_obs
        self._final_next_hidden_state = exp.next_hidden_state
        
    def sample(self) -> RecurrentPPOEpisodicRNDExperience:
        self._obs_buffer.append(self._final_next_obs)
        self._hidden_state_buffer.append(self._final_next_hidden_state)
        exp_batch = RecurrentPPOEpisodicRNDExperience(
            torch.concat(self._obs_buffer[:-1]),
            torch.concat(self._action_buffer),
            torch.concat(self._obs_buffer[1:]),
            torch.concat(self._reward_buffer),
            torch.concat(self._int_reward_buffer),
            torch.concat(self._terminated_buffer),
            torch.concat(self._action_log_prob_buffer),
            torch.concat(self._state_value_buffer),
            torch.concat(self._hidden_state_buffer[:-1], dim=1),
            torch.concat(self._hidden_state_buffer[1:], dim=1)
        )
        self.reset()
        return exp_batch
    
    def _make_buffer(self) -> list:
        return [None] * self._n_steps # type: ignore




# --- New SAC Experience and Replay Buffer ---

@dataclass(frozen=True)
class RecurrentSACTransition:
    """
    Represents a single transition for Recurrent SAC.
    Stores numpy arrays for efficiency in the buffer.
    Includes hidden states.
    """
    obs: np.ndarray # (obs_shape)
    action: np.ndarray # (action_shape)
    reward: float
    next_obs: np.ndarray # (obs_shape)
    terminated: bool
    # Store hidden states *before* taking the action
    actor_hidden_state: np.ndarray # (num_layers * D, H_actor)
    critic_hidden_state: np.ndarray # (num_layers * D, H_critic)

class ReplayBuffer:
    """
    Replay buffer for off-policy RL algorithms like SAC.
    Handles recurrent hidden states.
    """
    def __init__(self, buffer_size: int, num_envs: int, obs_shape: Tuple, action_shape: Tuple, actor_hidden_shape: Tuple, critic_hidden_shape: Tuple, device: torch.device):
        self._buffer_size = buffer_size
        self._num_envs = num_envs
        self._obs_shape = obs_shape
        self._action_shape = action_shape
        self._actor_hidden_shape = actor_hidden_shape
        self._critic_hidden_shape = critic_hidden_shape
        self.device = device

        # Calculate total size needed per environment
        self.obs = np.zeros((self._buffer_size, num_envs) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((self._buffer_size, num_envs) + action_shape, dtype=np.int64) # Assuming discrete actions (indices)
        self.rewards = np.zeros((self._buffer_size, num_envs), dtype=np.float32)
        self.next_obs = np.zeros((self._buffer_size, num_envs) + obs_shape, dtype=np.float32)
        self.terminated = np.zeros((self._buffer_size, num_envs), dtype=np.float32) # Use float for multiplication
        self.actor_hidden_states = np.zeros((self._buffer_size, num_envs) + actor_hidden_shape, dtype=np.float32)
        self.critic_hidden_states = np.zeros((self._buffer_size, num_envs) + critic_hidden_shape, dtype=np.float32)

        self._pos = 0
        self._full = False

    def add(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray, next_obs: np.ndarray, terminated: np.ndarray, actor_hidden: np.ndarray, critic_hidden: np.ndarray) -> None:
        """
        Add a new transition to the buffer.
        Expects inputs for all environments at a single time step.
        Args:
            obs: (num_envs, *obs_shape)
            action: (num_envs, *action_shape)
            reward: (num_envs,)
            next_obs: (num_envs, *obs_shape)
            terminated: (num_envs,)
            actor_hidden: (num_envs, num_layers*D, H_actor) - Hidden state *before* action
            critic_hidden: (num_envs, num_layers*D, H_critic) - Hidden state *before* action
        """
        if actor_hidden.shape[1:] != self._actor_hidden_shape or critic_hidden.shape[1:] != self._critic_hidden_shape:
             raise ValueError(f"Unexpected hidden state shapes. Actor: got {actor_hidden.shape}, expected {self._actor_hidden_shape}. Critic: got {critic_hidden.shape}, expected {self._critic_hidden_shape}")


        self.obs[self._pos] = obs
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.next_obs[self._pos] = next_obs
        self.terminated[self._pos] = terminated.astype(np.float32)
        self.actor_hidden_states[self._pos] = actor_hidden
        self.critic_hidden_states[self._pos] = critic_hidden

        self._pos += 1
        if self._pos == self._buffer_size:
            self._full = True
            self._pos = 0

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions from the buffer.
        Args:
            batch_size: The number of transitions to sample.
        Returns:
            A dictionary containing tensors for obs, actions, rewards, next_obs, terminated,
            actor_hidden_states, and critic_hidden_states.
        """
        upper_bound = self._buffer_size if self._full else self._pos
        if upper_bound < batch_size:
             print(f"Warning: Buffer size ({upper_bound}) is smaller than batch size ({batch_size}). Sampling with replacement or fewer samples might occur implicitly if indices repeat.")
             # Consider raising an error or sampling with replacement explicitly if needed.

        # Sample time steps
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        # Sample environments for each time step
        env_inds = np.random.randint(0, self._num_envs, size=batch_size)

        return self._get_samples(batch_inds, env_inds)

    def _get_samples(self, batch_inds: np.ndarray, env_inds: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Retrieve samples from the buffer based on indices.
        """
        # Retrieve data using advanced indexing
        obs_batch = self.obs[batch_inds, env_inds]
        actions_batch = self.actions[batch_inds, env_inds]
        rewards_batch = self.rewards[batch_inds, env_inds]
        next_obs_batch = self.next_obs[batch_inds, env_inds]
        terminated_batch = self.terminated[batch_inds, env_inds]
        actor_hidden_batch = self.actor_hidden_states[batch_inds, env_inds]
        critic_hidden_batch = self.critic_hidden_states[batch_inds, env_inds]

        # Convert to tensors
        data = {
            "obs": torch.as_tensor(obs_batch, device=self.device),
            "actions": torch.as_tensor(actions_batch, device=self.device), # Keep as int64 for embedding/indexing
            "rewards": torch.as_tensor(rewards_batch, device=self.device).unsqueeze(-1), # Add dim for consistency
            "next_obs": torch.as_tensor(next_obs_batch, device=self.device),
            "terminated": torch.as_tensor(terminated_batch, device=self.device).unsqueeze(-1), # Add dim for consistency
            "actor_hidden_states": torch.as_tensor(actor_hidden_batch, device=self.device),
            "critic_hidden_states": torch.as_tensor(critic_hidden_batch, device=self.device),
        }
        return data

    def __len__(self) -> int:
        return self._buffer_size if self._full else self._pos

    def _make_buffer(self) -> list:
        # This method might not be needed for the numpy-based buffer
        # Kept for potential compatibility if switching storage later
        return [None] * self._buffer_size # type: ignore

# --- Optional: Define RecurrentSACExperience if needed for clarity ---
# Although the buffer stores transitions, you might define this for type hinting
# or if the agent's update logic expects this structure.
@dataclass(frozen=True)
class RecurrentSACExperience:
    """
    Dataclass to potentially represent a batch sampled from the ReplayBuffer.
    Contains tensors ready for training.
    """
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    terminated: torch.Tensor
    actor_hidden_states: torch.Tensor
    critic_hidden_states: torch.Tensor