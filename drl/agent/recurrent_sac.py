# drl/agent/recurrent_sac.py
from typing import Dict, Optional, Tuple, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import drl.util.func as util_f
from drl.agent.agent import Agent, agent_config
from drl.agent.config import RecurrentSACConfig
# Import the INTERFACE and the specific IMPLEMENTATION
from drl.agent.net import RecurrentSACNetwork
from train.net import SelfiesRecurrentSACNet # Import the concrete network class
from drl.agent.trajectory import ReplayBuffer # Import ReplayBuffer
from drl.exp import Experience
from drl.net import Network, Trainer # Keep Trainer if used for optimizers
from drl.util import IncrementalMean

@agent_config("Recurrent SAC")
class RecurrentSAC(Agent):
    def __init__(
        self,
        config: RecurrentSACConfig,
        network: SelfiesRecurrentSACNet, # Expect the concrete implementation
        # trainer: Trainer, # SAC manages its own optimizers
        num_envs: int,
        obs_shape: Tuple, # Pass obs_shape explicitly
        action_shape: Tuple, # Pass action_shape explicitly (usually (1,) for discrete)
        num_actions: int, # Pass num_actions explicitly for target entropy
        device: Optional[str] = None
    ) -> None:
        # Initialize Agent base class with the network
        # The network passed in should already be the concrete SelfiesRecurrentSACNet
        super().__init__(num_envs, network, device)

        self._config = config
        self._network = network # This is the main (online) network instance
        self._num_actions = num_actions

        # --- Initialize Actor Critic Networks ---
        # These are part of the main network instance
        self._actor = self._network
        self._critic = self._network

        # --- Target Networks ---
        # Create target networks by copying the structure and loading state dict
        # Use the concrete class SelfiesRecurrentSACNet for instantiation
        self._critic_target = SelfiesRecurrentSACNet( # Use concrete class
            # Pass necessary args for network initialization
             obs_shape=obs_shape[0], # Get vocab size from obs_shape
             num_actions=num_actions,
             # Get other params from the main network instance for consistency
             hidden_dim=network._hidden_dim,
             n_recurrent_layers=network._n_recurrent_layers,
             # Temperature is usually associated with the policy, not critic target
             # temperature=network._actor_head._temperature # If needed, but unlikely
        ).to(self.device) # Move target network to the correct device

        self._critic_target.load_state_dict(self._critic.state_dict())
        # Freeze target network parameters
        for param in self._critic_target.parameters():
            param.requires_grad = False

        # --- Optimizers ---
        # Ensure parameters() method exists and returns correct parameters
        self._actor_optimizer = optim.Adam(self._network.actor_parameters(), lr=config.actor_lr)
        self._critic_optimizer = optim.Adam(self._network.critic_parameters(), lr=config.critic_lr)

        # --- Entropy Coefficient (Alpha) ---
        self._learn_alpha = config.learn_alpha
        if self._learn_alpha:
            # Target entropy: H = -|A| * ratio
            # For discrete actions, max entropy is log(|A|)
            # Target entropy is typically negative, e.g., -dim(A) * ratio
            # Use num_actions directly for discrete case
            max_entropy = np.log(num_actions) if num_actions > 0 else 1.0 # Avoid log(0)
            self._target_entropy = -max_entropy * config.target_entropy_ratio

            self._log_alpha = torch.tensor(np.log(config.alpha), dtype=torch.float32, requires_grad=True, device=self.device)
            self._alpha_optimizer = optim.Adam([self._log_alpha], lr=config.alpha_lr)
            self._alpha = self._log_alpha.exp().item() # Keep track of current alpha value
        else:
            self._log_alpha = torch.tensor(np.log(config.alpha), dtype=torch.float32, device=self.device)
            self._alpha = config.alpha
            self._alpha_optimizer = None
            self._target_entropy = 0.0 # Not used if alpha is fixed

        # --- Replay Buffer ---
        # Get hidden state shapes from the network instance
        actor_hidden_shape_tuple = self._network.hidden_state_shape() # e.g., (layers*D, H_lstm)
        critic_hidden_shape_tuple = self._network.hidden_state_shape() # Assuming same for critic for now

        self._replay_buffer = ReplayBuffer(
            buffer_size=config.buffer_size,
            num_envs=num_envs,
            obs_shape=obs_shape,
            action_shape=action_shape, # Should be (1,) for discrete index
            actor_hidden_shape=actor_hidden_shape_tuple, # Pass the tuple
            critic_hidden_shape=critic_hidden_shape_tuple, # Pass the tuple
            device=self.device
        )

        # --- Hidden States Tracking ---
        # Shape: (num_envs, num_layers*D, H_lstm) - Batch dimension first
        self._current_actor_hidden_state = torch.zeros((num_envs,) + actor_hidden_shape_tuple, device=self.device)
        self._current_critic_hidden_state = torch.zeros((num_envs,) + critic_hidden_shape_tuple, device=self.device)
        self._next_actor_hidden_state = torch.zeros((num_envs,) + actor_hidden_shape_tuple, device=self.device)
        self._next_critic_hidden_state = torch.zeros((num_envs,) + critic_hidden_shape_tuple, device=self.device)

        # Shape: (num_envs, 1)
        self._prev_terminated = torch.zeros(self._num_envs, 1, device=self.device, dtype=torch.float32)

        # --- Logging ---
        self._actor_avg_loss = IncrementalMean()
        self._critic_avg_loss = IncrementalMean()
        self._alpha_avg_loss = IncrementalMean()
        self._alpha_value_log = IncrementalMean() # To log the actual alpha value

    @property
    def config_dict(self) -> dict:
        # Return a dictionary representation of the config
        cfg = self._config.__dict__.copy()
        # Add runtime info if needed
        cfg['current_alpha'] = self._alpha
        return cfg

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """ Select action greedily or sample from policy """
        # Reset hidden state if previous step was terminal
        # Hidden state shape is (num_envs, num_layers*D, H)
        # Need to reshape for LSTM: (num_layers*D, num_envs, H)
        # Termination shape: (num_envs, 1) -> broadcast to hidden state shape
        term_mask = (1.0 - self._prev_terminated).unsqueeze(-1) # Shape (num_envs, 1, 1)

        actor_h_batch_first = self._next_actor_hidden_state * term_mask
        critic_h_batch_first = self._next_critic_hidden_state * term_mask # Track critic state too

        # Permute to (num_layers*D, num_envs, H) for LSTM input
        actor_h_lstm = actor_h_batch_first.permute(1, 0, 2).contiguous()
        critic_h_lstm = critic_h_batch_first.permute(1, 0, 2).contiguous()


        # Store current hidden states (before this action) for the buffer
        # Store in (batch, layers*D, H) format
        self._current_actor_hidden_state = actor_h_batch_first
        self._current_critic_hidden_state = critic_h_batch_first

        # Add sequence dimension (seq_len=1)
        obs_seq = obs.unsqueeze(1)

        # Actor forward pass requires hidden state in LSTM format
        policy_dist, next_actor_h_lstm = self._network.forward_actor(obs_seq, actor_h_lstm)

        # Sample action from the policy distribution
        action = policy_dist.sample() # Shape: (batch_size, seq_len=1, action_dim=1)

        # Critic forward pass to get next hidden state (needed for next step)
        # Action needs to be LongTensor for embedding lookup in critic
        action_long = action.long()
        _, _, next_critic_h_lstm = self._network.forward_critic(obs_seq, action_long, critic_h_lstm)

        # Store next hidden states (resulting from this action) for the next step
        # Permute back to (batch, layers*D, H)
        self._next_actor_hidden_state = next_actor_h_lstm.permute(1, 0, 2).contiguous()
        self._next_critic_hidden_state = next_critic_h_lstm.permute(1, 0, 2).contiguous()

        # Remove sequence dimension from action
        return action.squeeze(1) # Return shape (batch_size, 1)

    def update(self, exp: Experience) -> Optional[dict]:
        """ Add experience to buffer and perform gradient updates """
        # Store previous terminated state for hidden state resetting in select_action
        self._prev_terminated = exp.terminated.clone() # Shape (num_envs, 1)

        # Add experience to replay buffer (use numpy arrays)
        # Ensure hidden states stored are the ones *before* the action was taken
        self._replay_buffer.add(
            exp.obs.cpu().numpy(),
            exp.action.cpu().numpy(), # Shape (num_envs, 1)
            exp.reward.cpu().numpy().squeeze(-1), # Shape (num_envs,)
            exp.next_obs.cpu().numpy(),
            exp.terminated.cpu().numpy().squeeze(-1), # Shape (num_envs,)
            # Pass hidden states in (num_envs, layers*D, H) format
            self._current_actor_hidden_state.cpu().numpy(),
            self._current_critic_hidden_state.cpu().numpy()
        )

        # Perform gradient steps if buffer is large enough
        if len(self._replay_buffer) < self._config.learning_starts:
            return None

        metrics = {}
        for _ in range(self._config.gradient_steps):
            # Sample a batch from the replay buffer
            batch = self._replay_buffer.sample(self._config.batch_size)
            step_metrics = self._learn(batch)
            # Aggregate metrics if needed (e.g., average over gradient steps)
            metrics.update(step_metrics) # Simplest: just keep last step's metrics

        # Log aggregated metrics
        log_data = {}
        if metrics:
             log_data["Training/Critic Loss"] = (self._critic_avg_loss.mean, self.training_steps)
             log_data["Training/Actor Loss"] = (self._actor_avg_loss.mean, self.training_steps)
             if self._learn_alpha:
                 log_data["Training/Alpha Loss"] = (self._alpha_avg_loss.mean, self.training_steps)
             log_data["Training/Alpha"] = (self._alpha_value_log.mean, self.training_steps)

             # Reset incremental means after logging
             self._critic_avg_loss.reset()
             self._actor_avg_loss.reset()
             self._alpha_avg_loss.reset()
             self._alpha_value_log.reset()

        return {"metric": log_data} if log_data else None


    def _learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """ Perform one gradient update step """
        obs = batch["obs"]
        actions = batch["actions"] # Shape (batch_size, 1)
        rewards = batch["rewards"] # Shape (batch_size, 1)
        next_obs = batch["next_obs"]
        terminated = batch["terminated"] # Shape (batch_size, 1)
        # Hidden states from buffer are for the *start* of the transition (obs)
        # Shape: (batch_size, num_layers*D, H)
        actor_h_batch = batch["actor_hidden_states"]
        critic_h_batch = batch["critic_hidden_states"]

        # Permute hidden states to LSTM format: (num_layers*D, batch_size, H)
        actor_h_lstm = actor_h_batch.permute(1, 0, 2).contiguous()
        critic_h_lstm = critic_h_batch.permute(1, 0, 2).contiguous()

        # Add sequence dimension (seq_len=1) for network inputs
        obs_seq = obs.unsqueeze(1)
        next_obs_seq = next_obs.unsqueeze(1)
        # Unsqueeze actions to add sequence dimension: (batch_size, 1) -> (batch_size, 1, 1)
        actions_seq = actions.unsqueeze(1) # Add sequence dimension

        # --- Critic Update ---
        with torch.no_grad():
            # Get next action and log prob from *current* policy using next_obs
            # We need the hidden state that *results* from processing next_obs
            # First, get the next hidden state from the actor using the initial hidden state from the buffer
            _, next_actor_h_lstm = self._actor.forward_actor(next_obs_seq, actor_h_lstm)
            # Then, use this resulting hidden state to get the policy distribution for next_obs
            next_policy_dist, _ = self._actor.forward_actor(next_obs_seq, next_actor_h_lstm)

            next_actions_pi = next_policy_dist.sample() # Shape (batch, 1, 1)
            next_log_prob = next_policy_dist.log_prob(next_actions_pi) # Shape (batch, 1, 1)

            # Get Q values from target critic network for next_obs and next_actions_pi
            # We need the hidden state that *results* from processing next_obs with the critic
            _, _, next_critic_h_lstm = self._critic_target.forward_critic(next_obs_seq, next_actions_pi.long(), critic_h_lstm)
            # Use this resulting hidden state to get the target Q values
            q1_next_target, q2_next_target, _ = self._critic_target.forward_critic(next_obs_seq, next_actions_pi.long(), next_critic_h_lstm)
            # Shape: (batch, 1, 1)

            # Remove sequence dimension
            q1_next_target = q1_next_target.squeeze(1) # Shape (batch, 1)
            q2_next_target = q2_next_target.squeeze(1) # Shape (batch, 1)
            next_log_prob = next_log_prob.squeeze(1)   # Shape (batch, 1)

            min_q_next_target = torch.min(q1_next_target, q2_next_target)
            # TD target value
            next_q_value = min_q_next_target - self._alpha * next_log_prob
            target_q = rewards + (1.0 - terminated) * self._config.gamma * next_q_value # Shape (batch, 1)

        # Get current Q estimates using obs and actions from buffer
        # Use the initial hidden state from the buffer
        # Pass actions_seq which now has shape (batch_size, 1, 1)
        current_q1, current_q2, _ = self._critic.forward_critic(obs_seq, actions_seq.long(), critic_h_lstm)
        # Shape: (batch, 1, 1)

        # Remove sequence dimension
        current_q1 = current_q1.squeeze(1) # Shape (batch, 1)
        current_q2 = current_q2.squeeze(1) # Shape (batch, 1)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize the critic
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        # Optional: Gradient clipping for critic
        # torch.nn.utils.clip_grad_norm_(self._network.critic_parameters(), max_norm=...)
        self._critic_optimizer.step()

        # --- Actor Update ---
        # <<<< REMOVE CRITIC FREEZING >>>>
        # for param in self._critic.parameters():
        #     param.requires_grad = False

        # Get actions and log probs from current policy using obs
        # Use the initial hidden state from the buffer
        policy_dist, _ = self._actor.forward_actor(obs_seq, actor_h_lstm)
        actions_pi = policy_dist.sample() # Shape (batch, 1, 1)
        log_prob = policy_dist.log_prob(actions_pi) # Shape (batch, 1, 1)

        # Get Q values for the policy's actions using obs
        # Use the initial hidden state from the buffer
        # Pass actions_pi which has shape (batch_size, 1, 1)
        # Detach actions_pi before passing to critic to prevent gradients flowing through action
        q1_pi, q2_pi, _ = self._critic.forward_critic(obs_seq, actions_pi.long().detach(), critic_h_lstm)
        # Shape: (batch, 1, 1)

        # Remove sequence dimension
        q1_pi = q1_pi.squeeze(1) # Shape (batch, 1)
        q2_pi = q2_pi.squeeze(1) # Shape (batch, 1)
        log_prob = log_prob.squeeze(1) # Shape (batch, 1)

        min_q_pi = torch.min(q1_pi, q2_pi)
        # Detach min_q_pi to ensure loss gradient only comes from log_prob term w.r.t actor params
        actor_loss = (self._alpha * log_prob - min_q_pi.detach()).mean() # <<< DETACH Q VALUE

        # Optimize the actor
        self._actor_optimizer.zero_grad()
        actor_loss.backward() # <<< Error occurred here previously
        # Optional: Gradient clipping for actor
        # torch.nn.utils.clip_grad_norm_(self._network.actor_parameters(), max_norm=...)
        self._actor_optimizer.step()

        # <<<< REMOVE CRITIC UNFREEZING >>>>
        # for param in self._critic.parameters():
        #     param.requires_grad = True

        # --- Alpha Update (Optional) ---
        alpha_loss_val = 0.0
        if self._learn_alpha and self._alpha_optimizer:
            # Use the log_prob calculated for the actor update
            # Detach log_prob here as alpha loss shouldn't affect policy parameters
            alpha_loss = - (self._log_alpha * (log_prob + self._target_entropy).detach()).mean()

            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()
            # Clamp log_alpha to avoid extreme values if necessary
            # self._log_alpha.data.clamp_(min_log_alpha, max_log_alpha)
            self._alpha = self._log_alpha.exp().item() # Update tracked alpha value
            alpha_loss_val = alpha_loss.item()
            self._alpha_avg_loss.update(alpha_loss_val)


        # --- Target Network Update ---
        # Note: training_steps should be incremented *after* the check
        if self.training_steps % self._config.target_update_interval == 0:
            self._polyak_update(self._critic, self._critic_target, self._config.tau)

        # --- Logging & Step Increment ---
        self._tick_training_steps() # Increment training steps *after* potential target update
        self._critic_avg_loss.update(critic_loss.item())
        self._actor_avg_loss.update(actor_loss.item())
        self._alpha_value_log.update(self._alpha)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss_val,
            "alpha": self._alpha,
        }

    def _polyak_update(self, source_net: nn.Module, target_net: nn.Module, tau: float) -> None:
        """ Soft update target network parameters """
        with torch.no_grad():
            for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
                target_param.data.mul_(1.0 - tau)
                torch.add(target_param.data, source_param.data, alpha=tau, out=target_param.data)

    def inference_agent(self, num_envs: int = 1, device: Optional[str] = None) -> Agent:
        # Create an inference-specific agent instance
        from .recurrent_sac_inference import RecurrentSACInference # Avoid circular import
        inference_device = device or str(self.device)
        # Pass the *main* network instance (which contains the actor logic)
        return RecurrentSACInference(self._network, num_envs, inference_device)

    @property
    def log_data(self) -> Dict[str, tuple]:
        """ Returns log data and resets incremental means """
        # This is handled within the update method now to ensure
        # logs are generated only when updates happen.
        return {} # Return empty dict as logging is done in update

    @property
    def state_dict(self) -> dict:
        """Returns the state dict of the agent."""
        sd = super().state_dict # Get base state_dict (training_steps, model)
        sd.update({
            'critic_target_state_dict': self._critic_target.state_dict(),
            'actor_optimizer_state_dict': self._actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self._critic_optimizer.state_dict(),
        })
        if self._learn_alpha and self._alpha_optimizer:
            sd['log_alpha_state_dict'] = self._log_alpha.detach().cpu() # Save tensor value on CPU
            sd['alpha_optimizer_state_dict'] = self._alpha_optimizer.state_dict()
        return sd

    def load_state_dict(self, state_dict: dict):
        """Load the state dict."""
        super().load_state_dict(state_dict) # Load base state_dict (training_steps, model)
        self._critic_target.load_state_dict(state_dict['critic_target_state_dict'])
        self._actor_optimizer.load_state_dict(state_dict['actor_optimizer_state_dict'])
        self._critic_optimizer.load_state_dict(state_dict['critic_optimizer_state_dict'])
        if self._learn_alpha and self._alpha_optimizer:
             # Load log_alpha tensor and update optimizer state
             # Ensure it's loaded back to the correct device and requires grad
            self._log_alpha.data = state_dict['log_alpha_state_dict'].to(self.device)
            self._log_alpha.requires_grad = True
            self._alpha_optimizer.load_state_dict(state_dict['alpha_optimizer_state_dict'])
            # Re-link log_alpha to the optimizer (important!)
            self._alpha_optimizer.param_groups[0]['params'] = [self._log_alpha]
            self._alpha = self._log_alpha.exp().item() # Update tracked alpha
