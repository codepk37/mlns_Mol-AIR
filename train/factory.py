# train/factory.py
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.optim as optim

import drl
import drl.agent as agent # Import agent module
import train.net as train_net # Import train network implementations
import util
from envs import Env
from envs.chem_env import make_async_chem_env
from train.train import Train
from train.pretrain import Pretrain, SelfiesDataset
from util import instance_from_dict
from train.inference import Inference
# from train.net import SelfiesPretrainedNet # Already imported via train_net
import os

class ConfigParsingError(Exception):
    pass

def yaml_to_config_dict(file_path: str) -> Tuple[str, dict]:
    try:
        config_dict = util.load_yaml(file_path)
    except FileNotFoundError:
        raise ConfigParsingError(f"Config file not found: {file_path}")

    try:
        config_id = tuple(config_dict.keys())[0]
    except IndexError: # Changed from generic Exception
        raise ConfigParsingError("YAML config file must start with the training ID.")
    config = config_dict[config_id]
    return config_id, config

@dataclass(frozen=True)
class CommonConfig:
    num_envs: int = 1
    seed: Optional[int] = None
    device: Optional[str] = None
    # lr: float = 1e-3 # LR is now agent-specific (actor_lr, critic_lr, alpha_lr)
    grad_clip_max_norm: float = 5.0 # Keep for potential use, though less common in SAC
    pretrained_path: Optional[str] = None
    num_inference_envs: int = 0

class MolRLTrainFactory:
    """
    Factory class creates a Train instance from a dictionary config.
    """
    @staticmethod
    def from_yaml(file_path: str) -> "MolRLTrainFactory":
        """ Create a `MolRLTrainFactory` from a YAML file. """
        config_id, config = yaml_to_config_dict(file_path)
        return MolRLTrainFactory(config_id, config)

    def __init__(self, id: str, config: dict):
        self._id = id
        self._config = config
        self._agent_config_dict = self._config.get("Agent", {}) # Store raw dict
        self._env_config = self._config.get("Env", {})
        self._train_config = self._config.get("Train", {})
        self._count_int_reward_config = self._config.get("CountIntReward", {})
        # Use train_config for common settings like seed, device etc.
        self._common_config = instance_from_dict(CommonConfig, self._train_config)
        self._pretrained = None

    def create_train(self) -> Train:
        self._train_setup()

        try:
            env = self._create_env()
            inference_env = self._create_inference_env()
        except TypeError as e:
            raise ConfigParsingError(f"Invalid Env config. Missing arguments or wrong type: {e}")
        except KeyError as e:
             raise ConfigParsingError(f"Missing required key in Env config: {e}")

        try:
            agent_instance = self._create_agent(env)
        except TypeError as e:
            raise ConfigParsingError(f"Invalid Agent config. Missing arguments or wrong type: {e}")
        except KeyError as e:
             raise ConfigParsingError(f"Missing required key in Agent config: {e}")
        except ValueError as e: # Catch agent type error
             raise ConfigParsingError(str(e))


        try:
            smiles_or_selfies_refset = util.load_smiles_or_selfies(self._train_config["refset_path"]) if "refset_path" in self._train_config else None
            # Pass necessary args from train_config to Train constructor
            train_args = {
                "env": env,
                "agent": agent_instance,
                "id": self._id,
                "inference_env": inference_env,
                "smiles_or_selfies_refset": smiles_or_selfies_refset,
                "total_time_steps": self._train_config.get("total_time_steps"),
                "summary_freq": self._train_config.get("summary_freq"),
                "agent_save_freq": self._train_config.get("agent_save_freq"),
                "n_inference_episodes": self._train_config.get("n_inference_episodes", 1) # Default if missing
            }
            # Filter out None values before passing to instance_from_dict
            train_args_filtered = {k: v for k, v in train_args.items() if v is not None}
            train = instance_from_dict(Train, train_args_filtered)

        except TypeError as e:
            raise ConfigParsingError(f"Invalid Train config. Missing arguments or wrong type: {e}")
        except KeyError as e:
             raise ConfigParsingError(f"Missing required key in Train config: {e}")

        return train

    def _train_setup(self):
        util.logger.enable(self._id, enable_log_file=False) # Disable file logging initially
        util.try_create_dir(util.logger.dir())
        config_to_save = {self._id: self._config}
        util.save_yaml(f"{util.logger.dir()}/config.yaml", config_to_save)

        if self._common_config.seed is not None:
            util.seed(self._common_config.seed)

        # --- Pretrained Model Loading ---
        pretrained_load_path = self._common_config.pretrained_path
        if pretrained_load_path is None and os.path.exists(f"{util.logger.dir()}/pretrained_models/best.pt"):
             pretrained_load_path = f"{util.logger.dir()}/pretrained_models/best.pt"

        if pretrained_load_path and os.path.exists(pretrained_load_path):
             try:
                 self._pretrained = torch.load(pretrained_load_path, map_location=self._common_config.device or 'cpu')
                 print(f"[Factory] Loaded pretrained model from: {pretrained_load_path}")
             except Exception as e:
                 print(f"[Factory] Warning: Failed to load pretrained model from {pretrained_load_path}: {e}")
                 self._pretrained = None
        elif pretrained_load_path:
             print(f"[Factory] Warning: Pretrained model path specified but not found: {pretrained_load_path}")


        # --- Vocabulary Loading ---
        vocab_load_path = self._env_config.get("vocab_path")
        if vocab_load_path is None and os.path.exists(f"{util.logger.dir()}/vocab.json"):
             vocab_load_path = f"{util.logger.dir()}/vocab.json"

        if vocab_load_path and os.path.exists(vocab_load_path):
            try:
                vocab, max_str_len = util.load_vocab(vocab_load_path)
                self._env_config["vocabulary"] = vocab # Add to env_config for env creation
                self._env_config["max_str_len"] = max_str_len
                print(f"[Factory] Loaded vocabulary from: {vocab_load_path}")
            except Exception as e:
                raise ConfigParsingError(f"Failed to load vocabulary from {vocab_load_path}: {e}")
        elif "vocabulary" not in self._env_config:
             # If no vocab path and not already in config, it might be an issue depending on the env
             print("[Factory] Warning: Vocabulary not found or specified. Environment might fail if it requires one.")


        util.logger.disable() # Disable logger after setup

    def _create_env(self) -> Env:
        # Ensure necessary keys are present before calling make_async_chem_env
        # 'vocabulary' and 'max_str_len' should be added during _train_setup if loaded
        required_keys = [] # Add keys absolutely required by make_async_chem_env
        for key in required_keys:
             if key not in self._env_config:
                  raise ConfigParsingError(f"Missing required key '{key}' in Env config for make_async_chem_env.")

        # Combine env config and count reward config
        full_env_config = {**self._env_config, **self._count_int_reward_config}

        env = make_async_chem_env(
            num_envs=self._common_config.num_envs,
            seed=self._common_config.seed,
            **full_env_config # Pass combined config
        )
        return env

    def _create_inference_env(self) -> Optional[Env]:
        if self._common_config.num_inference_envs <= 0: # Check for > 0
            return None

        # Use the same env_config as the training env, but without count reward config
        inference_env_config = self._env_config.copy()

        env = make_async_chem_env(
            num_envs=self._common_config.num_inference_envs,
            seed=self._common_config.seed, # Use same base seed, async env handles per-env seeding
            **inference_env_config
        )
        return env

    def _create_agent(self, env: Env) -> agent.Agent:
        agent_type = self._agent_config_dict.get("type", "").lower()
        if not agent_type:
             raise ValueError("Agent 'type' not specified in configuration.")

        obs_shape = env.obs_shape
        # Assuming discrete actions for SELFIES, action_shape is usually (1,)
        action_shape = (1,) # Or derive from env if it provides it
        num_actions = env.num_actions

        if agent_type == "ppo":
            return self._create_ppo_agent(env)
        elif agent_type == "rnd": # PPO + RND
            return self._create_rnd_agent(env)
        elif agent_type == "pretrained":
            return self._create_pretrained_agent(env)
        elif agent_type == "sac":
            return self._create_sac_agent(env, obs_shape, action_shape, num_actions)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def _create_ppo_agent(self, env: Env) -> agent.RecurrentPPO:
        config = instance_from_dict(agent.RecurrentPPOConfig, self._agent_config_dict)
        network = train_net.SelfiesRecurrentPPONet(
            env.obs_shape[0], # Assuming obs_shape[0] is vocab size for one-hot
            env.num_actions
        )
        if self._pretrained is not None and "model" in self._pretrained:
            network.load_state_dict(self._pretrained["model"], strict=False)
            print("[Factory] Loaded pretrained weights into PPO network.")

        # Use default LR from Train config if not specified in Agent config
        lr = self._train_config.get("lr", 1e-3) # Get default LR

        trainer = drl.Trainer(optim.Adam(
            network.parameters(),
            lr=lr
        )).enable_grad_clip(network.parameters(), max_norm=self._common_config.grad_clip_max_norm)

        return agent.RecurrentPPO(
            config=config,
            network=network,
            trainer=trainer,
            num_envs=self._common_config.num_envs,
            device=self._common_config.device
        )

    def _create_rnd_agent(self, env: Env) -> agent.RecurrentPPORND:
        config = instance_from_dict(agent.RecurrentPPORNDConfig, self._agent_config_dict)
        network = train_net.SelfiesRecurrentPPORNDNet(
            env.obs_shape[0],
            env.num_actions
        )
        if self._pretrained is not None and "model" in self._pretrained:
            network.load_state_dict(self._pretrained["model"], strict=False)
            print("[Factory] Loaded pretrained weights into RND network.")

        lr = self._train_config.get("lr", 1e-3)

        trainer = drl.Trainer(optim.Adam(
            network.parameters(),
            lr=lr
        )).enable_grad_clip(network.parameters(), max_norm=self._common_config.grad_clip_max_norm)

        return agent.RecurrentPPORND(
            config=config,
            network=network,
            trainer=trainer,
            num_envs=self._common_config.num_envs,
            device=self._common_config.device
        )

    def _create_pretrained_agent(self, env: Env) -> agent.PretrainedRecurrentAgent:
        # Pretrained agent typically uses a simpler network structure
        network = train_net.SelfiesPretrainedNet(
            env.num_actions, # Assumes input is vocab size
        )
        if self._pretrained is not None and "model" in self._pretrained:
            network.load_state_dict(self._pretrained["model"], strict=False)
            print("[Factory] Loaded pretrained weights into Pretrained network.")
        else:
             print("[Factory] Warning: Pretrained agent type selected, but no pretrained weights found or loaded.")


        return agent.PretrainedRecurrentAgent(
            network=network,
            num_envs=self._common_config.num_envs,
            device=self._common_config.device
        )

    def _create_sac_agent(self, env: Env, obs_shape, action_shape, num_actions) -> agent.RecurrentSAC:
        # Instantiate SAC config from the agent config dictionary
        config = instance_from_dict(agent.RecurrentSACConfig, self._agent_config_dict)

        # Instantiate the specific SAC network implementation
        network = train_net.SelfiesRecurrentSACNet(
            obs_shape=obs_shape[0], # Pass vocab size
            num_actions=num_actions,
            # hidden_dim=self._agent_config_dict.get("hidden_dim", 64), # Example: Get from config or use default
            # n_recurrent_layers=self._agent_config_dict.get("n_recurrent_layers", 2)
        )

        # Load pretrained weights if available (might need careful handling for actor/critic separation)
        if self._pretrained is not None and "model" in self._pretrained:
             # Loading into SAC requires care as architecture differs from PPO/Pretrained
             # Option 1: Load selectively if possible (e.g., only shared backbone)
             # Option 2: Load fully and accept potential mismatches (strict=False)
             try:
                 network.load_state_dict(self._pretrained["model"], strict=False)
                 print("[Factory] Loaded pretrained weights into SAC network (strict=False).")
             except Exception as e:
                  print(f"[Factory] Warning: Failed to load pretrained weights into SAC network: {e}")


        # SAC manages its own optimizers, so no Trainer is passed directly
        return agent.RecurrentSAC(
            config=config,
            network=network,
            num_envs=self._common_config.num_envs,
            obs_shape=obs_shape,
            action_shape=action_shape,
            num_actions=num_actions,
            device=self._common_config.device
        )


class MolRLInferenceFactory:
    """ Factory class creates an Inference instance from a dictionary config. """
    @staticmethod
    def from_yaml(file_path: str) -> "MolRLInferenceFactory":
        """ Create a `MolRLInferenceFactory` from a YAML file. """
        config_id, config = yaml_to_config_dict(file_path)
        return MolRLInferenceFactory(config_id, config)

    def __init__(self, id: str, config: dict):
        self._id = id
        self._config = config
        self._agent_config_dict = self._config.get("Agent", {})
        self._env_config = self._config.get("Env", {})
        # Train config might contain seed/device defaults if not in Inference section
        self._train_config_defaults = self._config.get("Train", {})
        self._inference_config = self._config.get("Inference", {}) # Specific inference settings
        # Prioritize inference config, fall back to train config, then use defaults
        common_seed = self._inference_config.get("seed", self._train_config_defaults.get("seed"))
        common_device = self._inference_config.get("device", self._train_config_defaults.get("device"))
        # Note: num_envs for inference comes from inference_config
        self._common_config = CommonConfig(
             num_envs=self._inference_config.get("num_envs", 1), # Default to 1 if not specified
             seed=common_seed,
             device=common_device
             # Other CommonConfig fields aren't typically needed for inference factory setup
        )
        self._agent_ckpt = None # To store loaded agent state dict

    def create_inference(self) -> Inference:
        self._inference_setup() # Load agent checkpoint here

        try:
            env = self._create_env() # Create env based on inference num_envs
        except TypeError as e:
            raise ConfigParsingError(f"Invalid Env config for inference. Missing arguments or wrong type: {e}")
        except KeyError as e:
             raise ConfigParsingError(f"Missing required key in Env config for inference: {e}")

        try:
            # Create the base agent structure first
            agent_instance = self._create_agent_structure(env)
            # Load the checkpoint state into the agent
            if self._agent_ckpt:
                 agent_instance.load_state_dict(self._agent_ckpt)
                 print(f"[Factory] Loaded agent checkpoint state into {agent_instance.name} agent.")
            else:
                 # This should have been caught in _inference_setup, but double-check
                 raise ConfigParsingError("Agent checkpoint not loaded during setup.")

            # Get the specific inference agent instance
            inference_agent = agent_instance.inference_agent(
                 num_envs=env.num_envs, # Use inference env's num_envs
                 device=self._common_config.device
            )

        except TypeError as e:
            raise ConfigParsingError(f"Invalid Agent config for inference. Missing arguments or wrong type: {e}")
        except KeyError as e:
             raise ConfigParsingError(f"Missing required key in Agent config for inference: {e}")
        except ValueError as e: # Catch agent type error
             raise ConfigParsingError(str(e))
        except FileNotFoundError as e: # Catch checkpoint loading error
             raise ConfigParsingError(str(e))


        try:
            # Determine reference set path (priority: inference config -> train config)
            refset_path = self._inference_config.get("refset_path", self._train_config_defaults.get("refset_path"))
            smiles_or_selfies_refset = util.load_smiles_or_selfies(refset_path) if refset_path else None

            inference_args = {
                "env": env,
                "agent": inference_agent, # Use the inference-specific agent
                "id": self._id,
                "smiles_or_selfies_refset": smiles_or_selfies_refset,
                "n_episodes": self._inference_config.get("n_episodes"),
                "n_unique_molecules": self._inference_config.get("n_unique_molecules"),
                # Add other inference-specific args from config if needed by Inference class
            }
            inference_args_filtered = {k: v for k, v in inference_args.items() if v is not None}
            inference_instance = instance_from_dict(Inference, inference_args_filtered)

        except TypeError as e:
            raise ConfigParsingError(f"Invalid Inference config. Missing arguments or wrong type: {e}")
        except KeyError as e:
             raise ConfigParsingError(f"Missing required key in Inference config: {e}")

        return inference_instance

    def _inference_setup(self):
        """ Loads vocabulary and agent checkpoint. """
        # Setup logger temporarily for finding paths relative to results dir
        util.logger.enable(self._id, enable_log_file=False)
        results_dir = util.logger.dir() # Get results dir path
        util.logger.disable() # Disable after getting path

        if self._common_config.seed is not None:
            util.seed(self._common_config.seed)

        # --- Vocabulary Loading (same as train setup) ---
        vocab_load_path = self._env_config.get("vocab_path")
        if vocab_load_path is None and os.path.exists(f"{results_dir}/vocab.json"):
             vocab_load_path = f"{results_dir}/vocab.json"

        if vocab_load_path and os.path.exists(vocab_load_path):
            try:
                vocab, max_str_len = util.load_vocab(vocab_load_path)
                self._env_config["vocabulary"] = vocab
                self._env_config["max_str_len"] = max_str_len
                print(f"[Factory] Loaded vocabulary for inference from: {vocab_load_path}")
            except Exception as e:
                raise ConfigParsingError(f"Failed to load vocabulary from {vocab_load_path} for inference: {e}")
        elif "vocabulary" not in self._env_config:
             print("[Factory] Warning: Vocabulary not found or specified for inference.")


        # --- Agent Checkpoint Loading ---
        ckpt_spec = self._inference_config.get("ckpt", "best") # Default to 'best'
        ckpt_path = None
        if ckpt_spec == "best":
            ckpt_path = f"{results_dir}/best_agent.pt"
        elif ckpt_spec == "final":
            ckpt_path = f"{results_dir}/agent.pt"
        elif isinstance(ckpt_spec, int):
            ckpt_path = f"{results_dir}/agent_ckpt/agent_{ckpt_spec}.pt"
        elif os.path.isabs(ckpt_spec) and util.file_exists(ckpt_spec): # Allow absolute path
             ckpt_path = ckpt_spec
        elif util.file_exists(os.path.join(results_dir, ckpt_spec)): # Allow relative path within results dir
             ckpt_path = os.path.join(results_dir, ckpt_spec)


        if ckpt_path and os.path.exists(ckpt_path):
            try:
                # Load the entire state dict, including agent sub-dict
                full_state_dict = torch.load(ckpt_path, map_location=self._common_config.device or 'cpu')
                if "agent" in full_state_dict:
                     self._agent_ckpt = full_state_dict["agent"] # Store only the agent's state dict
                     print(f"[Factory] Loaded agent checkpoint from: {ckpt_path}")
                else:
                     raise ConfigParsingError(f"Agent state dict not found within checkpoint file: {ckpt_path}")

            except FileNotFoundError:
                 raise ConfigParsingError(f"Specified checkpoint file not found: {ckpt_path}")
            except Exception as e:
                 raise ConfigParsingError(f"Error loading checkpoint from {ckpt_path}: {e}")
        else:
            raise ConfigParsingError(f"Could not find or determine checkpoint path from spec: '{ckpt_spec}' in directory {results_dir}")


    def _create_env(self) -> Env:
        """ Creates environment for inference using inference num_envs """
        # Inference env should not use count reward intrinsically
        inference_env_config = self._env_config.copy()

        env = make_async_chem_env(
            num_envs=self._common_config.num_envs, # Use inference num_envs
            seed=self._common_config.seed,
            **inference_env_config
        )
        return env

    def _create_agent_structure(self, env: Env) -> agent.Agent:
        """ Creates the agent object structure *without* loading the state dict. """
        # This logic is similar to _create_agent in MolRLTrainFactory,
        # but it just sets up the object. The state is loaded later.
        agent_type = self._agent_config_dict.get("type", "").lower()
        if not agent_type:
             raise ValueError("Agent 'type' not specified in configuration.")

        obs_shape = env.obs_shape
        action_shape = (1,) # Assume discrete
        num_actions = env.num_actions

        # We don't need trainers or optimizers for inference structure creation
        dummy_trainer = drl.Trainer(optim.Adam([torch.nn.Parameter(torch.empty(1))], lr=1e-4)) # Dummy optimizer

        if agent_type == "ppo":
            config = instance_from_dict(agent.RecurrentPPOConfig, self._agent_config_dict)
            network = train_net.SelfiesRecurrentPPONet(obs_shape[0], num_actions)
            # Pass dummy trainer
            return agent.RecurrentPPO(config, network, dummy_trainer, env.num_envs, self._common_config.device)
        elif agent_type == "rnd":
            config = instance_from_dict(agent.RecurrentPPORNDConfig, self._agent_config_dict)
            network = train_net.SelfiesRecurrentPPORNDNet(obs_shape[0], num_actions)
            # Pass dummy trainer
            return agent.RecurrentPPORND(config, network, dummy_trainer, env.num_envs, self._common_config.device)
        elif agent_type == "pretrained":
            network = train_net.SelfiesPretrainedNet(num_actions)
            return agent.PretrainedRecurrentAgent(network, env.num_envs, self._common_config.device)
        elif agent_type == "sac":
            config = instance_from_dict(agent.RecurrentSACConfig, self._agent_config_dict)
            network = train_net.SelfiesRecurrentSACNet(obs_shape[0], num_actions)
            # SAC agent constructor doesn't need trainer, manages optimizers internally
            return agent.RecurrentSAC(config, network, env.num_envs, obs_shape, action_shape, num_actions, self._common_config.device)
        else:
            raise ValueError(f"Unknown agent type for inference structure: {agent_type}")


class MolRLPretrainFactory:
    @staticmethod
    def from_yaml(file_path: str) -> "MolRLPretrainFactory":
        """ Create a `MolRLPretrainFactory` from a YAML file. """
        config_id, config = yaml_to_config_dict(file_path)
        return MolRLPretrainFactory(config_id, config)

    def __init__(self, id: str, config: dict):
        self._id = id
        self._config = config
        self._pretrain_config = self._config.get("Pretrain", {})
        # Pretrain might have its own seed/device settings
        self._common_config = instance_from_dict(CommonConfig, self._pretrain_config)

    def create_pretrain(self) -> Pretrain:
        self._pretrain_setup()

        try:
            # Dataset path must be in Pretrain config
            dataset_path = self._pretrain_config.get("dataset_path")
            if not dataset_path:
                 raise ConfigParsingError("Missing 'dataset_path' in Pretrain config.")
            dataset = SelfiesDataset.from_txt(dataset_path)
        except FileNotFoundError:
             raise ConfigParsingError(f"Dataset file not found: {dataset_path}")
        except Exception as e:
            raise ConfigParsingError(f"Error creating dataset: {e}")

        try:
            # Network uses vocab size from the loaded dataset
            net = train_net.SelfiesPretrainedNet(vocab_size=dataset.tokenizer.vocab_size)
        except Exception as e:
            raise ConfigParsingError(f"Error creating pretrain network: {e}")

        try:
            # Combine pretrain config with necessary created objects
            pretrain_args = {
                "id": self._id,
                "net": net,
                "dataset": dataset,
                **self._pretrain_config, # Pass all other pretrain settings
            }
            # Filter None args if Pretrain constructor expects specific types
            pretrain_args_filtered = {k: v for k, v in pretrain_args.items() if v is not None}
            pretrain = instance_from_dict(Pretrain, pretrain_args_filtered)

        except TypeError as e:
            raise ConfigParsingError(f"Invalid Pretrain config. Missing arguments or wrong type: {e}")
        except KeyError as e:
             raise ConfigParsingError(f"Missing required key in Pretrain config: {e}")

        return pretrain

    def _pretrain_setup(self):
        util.logger.enable(self._id, enable_log_file=False) # Disable file logging initially
        util.try_create_dir(util.logger.dir())
        config_to_save = {self._id: self._config}
        util.save_yaml(f"{util.logger.dir()}/config.yaml", config_to_save)
        # Store log dir if needed later, e.g., for saving vocab
        self._log_dir = util.logger.dir()
        util.logger.disable() # Disable logger after setup

        if "seed" in self._pretrain_config: # Use seed from Pretrain config if present
            util.seed(self._pretrain_config["seed"])
