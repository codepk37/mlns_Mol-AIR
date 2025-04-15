# train/train.py
import time
from queue import Queue
from typing import Optional, List, Dict # Added Dict

import numpy as np
import pandas as pd
import torch

import drl
import drl.agent as agent
from metric import MolMetric
from drl.util import IncrementalMean
from envs import Env
from util import CSVSyncWriter, TextInfoBox, logger, to_smiles, try_create_dir
import os

class Train:
    _TRACE_ENV: int = 0

    def __init__(
        self,
        env: Env,
        agent: agent.Agent,
        id: str,
        total_time_steps: int,
        summary_freq: Optional[int] = None,
        agent_save_freq: Optional[int] = None,
        inference_env: Optional[Env] = None,
        n_inference_episodes: int = 1,
        smiles_or_selfies_refset: Optional[List[str]] = None,
    ) -> None:
        self._env = env
        self._agent = agent

        self._id = id
        self._total_time_steps = total_time_steps
        self._summary_freq = total_time_steps // 20 if summary_freq is None else summary_freq
        self._agent_save_freq = self._summary_freq * 10 if agent_save_freq is None else agent_save_freq
        self._inference_env = inference_env
        self._n_inference_episodes = n_inference_episodes
        self._smiles_or_selfies_refset = smiles_or_selfies_refset

        self._dtype = torch.float32
        self._device = self._agent.device
        self._best_score = float("-inf")

        self._time_steps = 0
        self._episodes = 0
        self._episode_len = 0
        self._real_start_time = time.time()
        self._real_time = 0.0

        self._cumulative_reward_mean = IncrementalMean()
        self._episode_len_mean = IncrementalMean()
        self._mol_metric = MolMetric()

        # helps to synchronize final molecule results of each episode
        self._metric_csv_sync_writer_dict: Dict[str, CSVSyncWriter] = {} # Type hint added
        # self._episode_molecule_sync_buffer_dict = defaultdict(lambda: SyncFixedBuffer(max_size=self._env.num_envs, callback=self._record_molecule))
        # to save final molecule results periodically
        # self._molecule_queue = Queue() # (episode, env_id, score, selfies) - Consider using CSV writer instead
        # self._best_molecule = None
        # self._best_molecule_queue = Queue() # (episode, best_score, selfies) - Consider using CSV writer instead
        # self._intrinsic_reward_queue = Queue() # (episode, env_id, time_step, intrinsic_rewards...) - Consider using CSV writer instead

        self._enabled = True

    def train(self) -> "Train":
        if not self._enabled:
            raise RuntimeError("Train is already closed.")

        if not logger.enabled():
            logger.enable(self._id, enable_log_file=False)

        if self._time_steps == self._total_time_steps:
            # self._save_train() # Save even if already finished? Maybe not needed.
            print(f"[{self._id}] Training already completed at {self._total_time_steps} steps.")
            return self

        self._load_train() # Load previous state if exists

        if self._time_steps >= self._total_time_steps:
            logger.print(f"Training is already finished at step {self._time_steps}.")
            return self

        logger.disable() # Disable temporary logger used for loading
        logger.enable(self._id, enable_log_file=True) # Enable logger with file output

        try:
            if self._smiles_or_selfies_refset is not None:
                logger.print(f"Preprocessing the SMILES reference set ({len(self._smiles_or_selfies_refset)}) for the molecular metric...")
                smiles_refset = to_smiles(self._smiles_or_selfies_refset)
                self._mol_metric.preprocess(smiles_refset=smiles_refset)

            self._print_train_info()
        except KeyboardInterrupt:
            logger.print(f"Training setup interrupted.")
            self.close() # Ensure resources are closed
            return self

        try:
            obs = self._env.reset()
            cumulative_reward = np.zeros(self._env.num_envs) # Track per environment
            last_agent_save_t = self._time_steps # Track last save step

            # Main training loop
            for current_step in range(self._time_steps, self._total_time_steps):
                # take action and observe
                obs_tensor = self._numpy_to_tensor(obs)
                action_tensor = self._agent.select_action(obs_tensor)
                # Ensure action is on CPU and numpy for env step
                action_np = action_tensor.detach().cpu().numpy()

                next_obs, reward, terminated, real_final_next_obs, env_info = self._env.step(action_np)

                # update the agent
                # Use real_final_next_obs for terminated environments
                real_next_obs_np = next_obs.copy()
                if np.any(terminated): # Avoid indexing if nothing terminated
                     real_next_obs_np[terminated] = real_final_next_obs

                real_next_obs_tensor = self._numpy_to_tensor(real_next_obs_np)

                exp = drl.Experience(
                    obs_tensor,
                    action_tensor, # Pass tensor action to agent update
                    real_next_obs_tensor,
                    self._numpy_to_tensor(reward[..., np.newaxis]),
                    self._numpy_to_tensor(terminated[..., np.newaxis]),
                )
                # Agent update step
                agent_info = self._agent.update(exp)

                # process info dicts from env and agent
                self._process_info_dict(env_info, agent_info)

                # take next step
                obs = next_obs
                cumulative_reward += reward # Update cumulative reward per env

                # Handle terminated environments for logging
                terminated_envs = np.where(terminated)[0]
                for env_idx in terminated_envs:
                     if env_idx == self._TRACE_ENV: # Log only for trace env
                         self._cumulative_reward_mean.update(cumulative_reward[env_idx])
                         # Episode length needs careful tracking if envs reset at different times
                         # self._episode_len_mean.update(self._episode_len) # This needs rework for async envs
                         self._tick_episode() # Increment total episode count
                     cumulative_reward[env_idx] = 0.0 # Reset reward for terminated env

                self._tick_time_steps() # Increment global time step counter

                # summary
                if self._time_steps % self._summary_freq == 0:
                    self._summary_train()

                # save the agent periodically and based on inference score
                if self._time_steps % self._agent_save_freq == 0:
                    score = self._inference(self._n_inference_episodes)
                    self._save_train(score) # Pass score to potentially save best agent
                    last_agent_save_t = self._time_steps

            logger.print(f"Training is finished at step {self._time_steps}.")
            # Save final agent if it wasn't saved on the last step
            if self._time_steps > last_agent_save_t:
                final_score = self._inference(self._n_inference_episodes)
                self._save_train(final_score)

        except KeyboardInterrupt:
            logger.print(f"Training interrupted at the time step {self._time_steps}.")
            self._save_train() # Save progress on interrupt
        finally:
             # Ensure writers are closed if they exist
             for writer in self._metric_csv_sync_writer_dict.values():
                  if writer is not None:
                       # Add a close method to CSVSyncWriter if needed, or ensure files are flushed
                       pass
             self.close() # Close environments and logger

        return self

    def close(self):
        if not self._enabled:
            return # Already closed
        self._enabled = False
        self._env.close()

        if self._inference_env is not None:
            self._inference_env.close()

        if logger.enabled():
            logger.disable()

    def _make_csv_sync_writer(self, metric_name: str, metric_info_dict: dict):
        # Ensure keys and values exist before accessing .keys()
        key_fields = metric_info_dict.get("keys", {}).keys()
        value_fields = metric_info_dict.get("values", {}).keys()

        # Add time_step if not already present (useful for env metrics)
        all_value_fields = list(value_fields)
        if "time_step" not in key_fields and "time_step" not in value_fields:
             all_value_fields.append("time_step")


        return CSVSyncWriter(
            file_path=f"{logger.dir()}/{metric_name}.csv",
            key_fields=key_fields,
            value_fields=tuple(all_value_fields), # Use potentially extended list
        )

    def _write_metric_dicts(self, metric_dicts, include_time_step=False):
        """ Writes metrics from an iterable of structured metric dictionaries. """
        if metric_dicts is None:
             return

        for metric_dict in metric_dicts:
            if metric_dict is None:
                continue
            # Expected structure: {'metric_type': {'keys': {...}, 'values': {...}}}
            for metric_name, metric_info in metric_dict.items():
                 # Validate structure before proceeding
                 if not isinstance(metric_info, dict) or "keys" not in metric_info or "values" not in metric_info:
                      logger.print(f"Warning: Skipping malformed metric entry for '{metric_name}': {metric_info}")
                      continue

                 # Add time step to values if requested
                 if include_time_step:
                     metric_info["values"]["time_step"] = self._time_steps

                 # Initialize writer if needed
                 if metric_name not in self._metric_csv_sync_writer_dict:
                     self._metric_csv_sync_writer_dict[metric_name] = self._make_csv_sync_writer(metric_name, metric_info)

                 current_writer = self._metric_csv_sync_writer_dict[metric_name]

                 # Check if new value fields need to be added to the header (should be rare after init)
                 new_fields = set(metric_info["values"].keys()) - set(current_writer.value_fields)
                 if new_fields:
                      # This indicates an unexpected change in metric structure - log a warning
                      logger.print(f"Warning: New value fields {new_fields} detected for metric '{metric_name}'. CSV header might become inconsistent if file already exists.")
                      # Optionally, could try to rewrite the CSV header, but safer to warn.
                      # current_writer.value_fields += tuple(new_fields) # Update writer fields (use setter if implemented)

                 # Add data row
                 current_writer.add(
                     keys=metric_info["keys"],
                     values=metric_info["values"],
                 )

    def _process_info_dict(self, env_info: dict, agent_info: Optional[dict]):
        """ Processes info dicts from environment and agent. """
        # Process environment metrics (expected to be list/tuple of dicts)
        if "metric" in env_info and env_info["metric"] is not None:
            # env_info['metric'] comes from AsyncEnv merging, might contain None entries
            valid_env_metrics = [m for m in env_info["metric"] if m is not None]
            if valid_env_metrics:
                 self._write_metric_dicts(valid_env_metrics, include_time_step=True)

        # Process agent metrics (expected to be a single dict: {'metric': {'Loss':(v,t), ...}})
        if agent_info is not None and "metric" in agent_info and agent_info["metric"]:
            agent_metrics = agent_info["metric"] # This is the dict like {'Loss':(v,t), ...}
            if isinstance(agent_metrics, dict):
                 for key, (value, step) in agent_metrics.items():
                      # Use logger's direct method for simple scalar logging
                      logger.log_data(key, value, step)
            else:
                 logger.print(f"Warning: Unexpected agent metric format: {agent_metrics}")


    def _tick_time_steps(self):
        # self._episode_len += 1 # Episode length tracking needs rework for async envs
        self._time_steps += 1 # Use internal counter consistent with loop
        self._real_time = time.time() - self._real_start_time

    def _tick_episode(self):
        # self._episode_len = 0 # Reset per-env episode length on termination
        self._episodes += 1 # Increment total episodes completed across all envs

    def _agent_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(device=self._device, dtype=self._dtype)

    def _numpy_to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x).to(device=self._device, dtype=self._dtype)

    def _print_train_info(self):
        text_info_box = TextInfoBox() \
            .add_text(f"[{self._id}] Training Start!") \
            .add_line(marker="=") \
            .add_text(f"ID: {self._id}") \
            .add_text(f"Output Path: {logger.dir()}")

        # display environment info
        env_config_dict = self._env.config_dict
        if len(env_config_dict.keys()) > 0:
            text_info_box.add_line() \
                .add_text(f"Environment INFO:")
            for key, value in env_config_dict.items():
                text_info_box.add_text(f"    {key}: {value}")

        # display training info
        text_info_box.add_line() \
            .add_text(f"Training INFO:") \
            .add_text(f"    number of environments: {self._env.num_envs}") \
            .add_text(f"    total time steps: {self._total_time_steps}") \
            .add_text(f"    summary frequency: {self._summary_freq}") \
            .add_text(f"    agent save frequency: {self._agent_save_freq}")

        # display agent info
        agent_config_dict = self._agent.config_dict
        # Add device info if not already present
        if 'device' not in agent_config_dict:
             agent_config_dict["device"] = str(self._device) # Convert device object to string

        text_info_box.add_line() \
            .add_text(f"Agent ({self._agent.name}):") # Add agent name if available
        for key, value in agent_config_dict.items():
            text_info_box.add_text(f"    {key}: {value}")

        logger.print(text_info_box.make(), prefix="")
        logger.print("", prefix="") # Add empty line for spacing

    def _summary_train(self):
        """ Logs summary statistics during training. """
        try:
            # Try reading the main metric file (e.g., episode_metric.csv)
            metric_file = f"{logger.dir()}/episode_metric.csv"
            if not os.path.exists(metric_file):
                 logger.print(f"Summary ({self._time_steps}/{self._total_time_steps}): Metric file not found yet.")
                 return

            metric_df = pd.read_csv(metric_file)
            # Filter metrics within the current summary window
            lower_bound = self._time_steps - self._summary_freq
            current_metric_df = metric_df[metric_df["time_step"] >= lower_bound]
            # Ensure 'score' and 'smiles' columns exist and drop rows with NaNs in them
            if "score" not in current_metric_df.columns or "smiles" not in current_metric_df.columns:
                 logger.print(f"Summary ({self._time_steps}/{self._total_time_steps}): 'score' or 'smiles' not found in metrics.")
                 return

            current_metric_df = current_metric_df.dropna(subset=["score", "smiles"])

            if len(current_metric_df) == 0:
                info = "No valid molecules generated in this summary window."
            else:
                score = current_metric_df["score"].mean()
                # Use the class instance of MolMetric
                self._mol_metric.preprocess(smiles_generated=current_metric_df["smiles"].tolist())
                diversity = self._mol_metric.calc_diversity()
                uniqueness = self._mol_metric.calc_uniqueness()
                info = f"score: {score:.3f}, diversity: {diversity:.3f}, uniqueness: {uniqueness:.3f}"
                logger.log_data("Environment/Score", score, self._time_steps)
                logger.log_data(f"Environment/Diversity", diversity, self._time_steps)
                logger.log_data(f"Environment/Uniqueness", uniqueness, self._time_steps)

                # Calculate novelty only if reference set was provided
                if self._mol_metric.smiles_refset is not None:
                    try:
                        novelty = self._mol_metric.calc_novelty()
                        info += f", novelty: {novelty:.3f}"
                        logger.log_data(f"Environment/Novelty", novelty, self._time_steps)
                    except ValueError as e:
                        logger.print(f"Warning: Could not calculate novelty - {e}")
                else:
                    # Log novelty as NaN or skip if no refset
                    logger.log_data(f"Environment/Novelty", float('nan'), self._time_steps)


            # Log average reward and episode length if available
            if self._cumulative_reward_mean.count > 0:
                 avg_reward = self._cumulative_reward_mean.mean
                 logger.log_data("Environment/Avg Reward (Trace Env)", avg_reward, self._time_steps)
                 info += f", avg_reward: {avg_reward:.3f}"
                 self._cumulative_reward_mean.reset() # Reset after logging

            # if self._episode_len_mean.count > 0:
            #      avg_ep_len = self._episode_len_mean.mean
            #      logger.log_data("Environment/Avg Ep Length (Trace Env)", avg_ep_len, self._time_steps)
            #      info += f", avg_ep_len: {avg_ep_len:.1f}"
            #      self._episode_len_mean.reset() # Reset after logging

            logger.print(f"Summary ({self._time_steps}/{self._total_time_steps}): {info}")

        except FileNotFoundError:
             logger.print(f"Summary ({self._time_steps}/{self._total_time_steps}): Waiting for metric file...")
        except pd.errors.EmptyDataError:
             logger.print(f"Summary ({self._time_steps}/{self._total_time_steps}): Metric file is empty.")
        except Exception as e:
             logger.print(f"Error during summary: {e}") # Log other potential errors

        # Plot logs regardless of summary success
        try:
            logger.plot_logs()
        except Exception as e:
             logger.print(f"Error plotting logs: {e}")


    def _save_train(self, score=None):
        """ Saves the training state and agent models. """
        train_dict = dict(
            time_steps=self._time_steps,
            episodes=self._episodes,
            # episode_len=self._episode_len, # Less meaningful for async
            best_score=self._best_score, # Save best score achieved so far
        )
        state_dict = dict(
            train=train_dict,
            agent=self._agent.state_dict,
        )

        agent_save_path = f"{logger.dir()}/agent.pt"
        torch.save(state_dict, agent_save_path)

        agent_ckpt_dir = f"{logger.dir()}/agent_ckpt"
        try_create_dir(agent_ckpt_dir)
        torch.save(state_dict, f"{agent_ckpt_dir}/agent_{self._time_steps}.pt")

        saved_best = False
        if score is not None and score > self._best_score:
            self._best_score = score
            best_agent_save_path = f"{logger.dir()}/best_agent.pt"
            torch.save(state_dict, best_agent_save_path)
            logger.print(f"Agent saved ({self._time_steps} steps). New best score: {score:.4f}. Saved to: {best_agent_save_path}")
            saved_best = True

        if not saved_best:
             logger.print(f"Agent saved ({self._time_steps} steps): {agent_save_path}")

        # self._env.save_data(logger.dir()) # If env needs saving
        # self._save_molecules(logger.dir()) # If using queue-based saving

    def _inference(self, n_episodes: int):
        """ Runs inference using the current agent policy. """
        if self._inference_env is None or n_episodes <= 0:
            return None # Return None if no inference env or episodes

        logger.print(f"--- Starting inference ({self._time_steps} steps) for {n_episodes} episodes ---")

        episodes_done = np.zeros((self._inference_env.num_envs,), dtype=int)
        total_episodes_target = n_episodes * self._inference_env.num_envs # Aim for roughly n_episodes per env if possible
        score_list = []
        smiles_list = []

        # Get a dedicated inference agent instance
        inference_agent = self._agent.inference_agent(self._inference_env.num_envs)
        inference_agent.model.eval() # Set model to evaluation mode

        obs = self._inference_env.reset()

        # Loop until enough episodes are collected across all inference envs
        pbar = tqdm(total=n_episodes, desc="Inference Episodes", leave=False)
        collected_episodes = 0
        while collected_episodes < n_episodes:
            obs_tensor = self._numpy_to_tensor(obs)
            with torch.no_grad():
                action_tensor = inference_agent.select_action(obs_tensor) # Use inference agent

            action_np = action_tensor.detach().cpu().numpy()
            next_obs, _, terminated, real_final_next_obs, env_info = self._inference_env.step(action_np)

            # Update inference agent's internal state (e.g., hidden state)
            # No learning update needed for inference agent usually
            real_next_obs_np = next_obs.copy()
            if np.any(terminated):
                 real_next_obs_np[terminated] = real_final_next_obs
            real_next_obs_tensor = self._numpy_to_tensor(real_next_obs_np)
            exp = drl.Experience(
                 obs_tensor, action_tensor, real_next_obs_tensor,
                 torch.zeros_like(self._numpy_to_tensor(terminated[..., np.newaxis])), # Dummy reward/term
                 self._numpy_to_tensor(terminated[..., np.newaxis])
            )
            with torch.no_grad():
                 _ = inference_agent.update(exp)

            obs = next_obs
            episodes_done += terminated.astype(int)

            # Process metrics from terminated environments during inference
            if "metric" in env_info and env_info["metric"] is not None:
                inf_scores, inf_smiles = self._inference_metric(env_info) # Extract scores/smiles
                score_list.extend(inf_scores)
                smiles_list.extend(inf_smiles)

            # Update progress bar based on completed episodes
            newly_finished = terminated.sum()
            collected_episodes += newly_finished
            pbar.update(newly_finished)
            if collected_episodes >= n_episodes:
                 break # Stop if target number of episodes is reached

        pbar.close()

        # Ensure model is back in training mode after inference
        self._agent.model.train()

        if not score_list: # Check if any valid scores were collected
            logger.print(f"--- Inference ({self._time_steps} steps) -> No valid molecules generated. ---")
            return None

        avg_score = np.mean(score_list)
        valid_smiles_list = [s for s in smiles_list if s] # Filter out potential None/empty smiles
        if not valid_smiles_list:
             logger.print(f"--- Inference ({self._time_steps} steps) -> Score: {avg_score:.3f}, No valid SMILES found. ---")
             diversity, uniqueness, novelty = 0.0, 0.0, 0.0 # Or NaN
        else:
             self._mol_metric.preprocess(smiles_generated=valid_smiles_list)
             diversity = self._mol_metric.calc_diversity()
             uniqueness = self._mol_metric.calc_uniqueness()

             info = f"Score: {avg_score:.3f}, Diversity: {diversity:.3f}, Uniqueness: {uniqueness:.3f}"
             logger.log_data("Inference/Score", avg_score, self._time_steps)
             logger.log_data("Inference/Diversity", diversity, self._time_steps)
             logger.log_data("Inference/Uniqueness", uniqueness, self._time_steps)

             novelty = float('nan') # Default if no refset
             if self._mol_metric.smiles_refset is not None:
                 try:
                     novelty = self._mol_metric.calc_novelty()
                     info += f", Novelty: {novelty:.3f}"
                 except ValueError as e:
                     logger.print(f"Warning: Could not calculate novelty during inference - {e}")
             logger.log_data("Inference/Novelty", novelty, self._time_steps) # Log NaN if no refset

             logger.print(f"--- Inference ({self._time_steps} steps) -> {info} ({len(valid_smiles_list)} valid mols) ---")

        # Plot logs after inference summary
        try:
            logger.plot_logs()
        except Exception as e:
             logger.print(f"Error plotting logs after inference: {e}")

        return avg_score # Return average score for best model tracking


    def _inference_metric(self, env_info: dict):
        """ Extracts scores and smiles from inference environment info. """
        scores = []
        smiles = []

        if "metric" not in env_info or env_info["metric"] is None:
            return scores, smiles

        # env_info['metric'] should be an iterable (list/tuple) from AsyncEnv
        for metric_dict in env_info["metric"]:
            if metric_dict is None or "episode_metric" not in metric_dict:
                continue

            # Ensure 'values' exists and is a dictionary
            values = metric_dict["episode_metric"].get("values")
            if not isinstance(values, dict):
                 continue

            score = values.get("score") # Use .get for safety
            smile = values.get("smiles")

            # Append only if both score and smiles are valid (not None)
            if score is not None and smile is not None:
                scores.append(score)
                smiles.append(smile)

        return scores, smiles


    def _load_train(self):
        """ Loads training state from checkpoint file. """
        load_path = f"{logger.dir()}/agent.pt"
        try:
            # Ensure loading to the correct device (CPU might be safer for loading)
            state_dict = torch.load(load_path, map_location='cpu')
            logger.print(f"Loading training state from: {load_path}")

            train_dict = state_dict.get("train", {})
            self._time_steps = train_dict.get("time_steps", 0)
            self._episodes = train_dict.get("episodes", 0)
            # self._episode_len = train_dict.get("episode_len", 0) # Less relevant now
            self._best_score = train_dict.get("best_score", float("-inf")) # Load best score

            agent_state = state_dict.get("agent")
            if agent_state:
                 self._agent.load_state_dict(agent_state)
                 logger.print(f"Agent state loaded successfully. Resuming from step {self._time_steps}.")
            else:
                 logger.print("Warning: Agent state not found in checkpoint.")

        except FileNotFoundError:
            logger.print("No checkpoint file found. Starting training from scratch.")
        except Exception as e:
             logger.print(f"Error loading checkpoint from {load_path}: {e}. Starting from scratch.")
             # Reset state variables if loading fails
             self._time_steps = 0
             self._episodes = 0
             self._best_score = float("-inf")

