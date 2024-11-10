import os
import json
import time
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
import torch
import wandb
from pathlib import Path

class TrainingLogger(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainingLogger, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        
        # Create directories
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.model_path = self.save_path / "models"
        self.log_path = self.save_path / "logs"
        self.model_path.mkdir(exist_ok=True)
        self.log_path.mkdir(exist_ok=True)
        
        # Initialize metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_start = time.time()

    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0:
            self.episode_rewards.append(self.model.ep_info_buffer[-1]["r"])
            self.episode_lengths.append(self.model.ep_info_buffer[-1]["l"])
        
        if self.n_calls % self.check_freq == 0:
            # Calculate metrics
            mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            mean_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
            
            # Log to wandb
            wandb.log({
                "timesteps": self.num_timesteps,
                "mean_reward": mean_reward,
                "mean_episode_length": mean_length,
                "total_episodes": len(self.episode_rewards)
            })
            
            # Save if we have a new best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.model_path / f"best_model_{self.num_timesteps}")
                
                # Save training stats
                stats = {
                    "timesteps": self.num_timesteps,
                    "best_mean_reward": float(self.best_mean_reward),
                    "total_episodes": len(self.episode_rewards),
                    "training_time": time.time() - self.training_start
                }
                with open(self.log_path / "training_stats.json", "w") as f:
                    json.dump(stats, f, indent=4)
            
            # Regular checkpoint save
            self.model.save(self.model_path / f"checkpoint_{self.num_timesteps}")
        
        return True

def make_env(game, rank):
    def _init():
        env = AngryBirdsEnv(game)
        # env = Monitor(env, f"./logs/env_{rank}")
        return env
    return _init

def train_agent(
    game,
    total_timesteps=1_000_000,
    eval_freq=10000,
    save_freq=50000,
    log_freq=1000,
    n_envs=4,
    load_path=None
):
    """
    Train the Angry Birds agent with comprehensive logging and saving
    
    Args:
        game: Game instance
        total_timesteps: Total training timesteps
        eval_freq: How often to evaluate
        save_freq: How often to save checkpoints
        log_freq: How often to log metrics
        n_envs: Number of parallel environments
        load_path: Path to load existing model (optional)
    """
    # Initialize wandb
    run = wandb.init(
        project="angry-birds-rl",
        config={
            "total_timesteps": total_timesteps,
            "n_envs": n_envs,
            "algorithm": "PPO"
        }
    )
    
    # Create save directories
    save_path = Path(f"training_runs/run_{int(time.time())}")
    
    # Create vectorized environment
    env = DummyVecEnv([make_env(game, i) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Initialize or load model
    if load_path is not None:
        print(f"Loading model from {load_path}")
        model = PPO.load(
            load_path,
            env=env,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        env = VecNormalize.load(f"{load_path}_env", env)
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,
            clip_range=0.2,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            tensorboard_log=str(save_path / "tensorboard")
        )
    
    # Setup callbacks
    callbacks = [
        TrainingLogger(
            check_freq=log_freq,
            save_path=save_path,
            verbose=1
        ),
        CheckpointCallback(
            save_freq=save_freq,
            save_path=str(save_path / "checkpoints"),
            name_prefix="ppo_angry_birds"
        ),
        EvalCallback(
            env,
            eval_freq=eval_freq,
            n_eval_episodes=10,
            log_path=str(save_path / "eval"),
            best_model_save_path=str(save_path / "best_model")
        )
    ]
    
    try:
        # Train the agent
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model and environment
        final_model_path = save_path / "final_model"
        model.save(final_model_path)
        env.save(f"{final_model_path}_env")
        
        # Save final training stats
        final_stats = {
            "total_timesteps": total_timesteps,
            "total_episodes": len(model.ep_info_buffer),
            "final_mean_reward": np.mean([ep["r"] for ep in model.ep_info_buffer]),
            "training_time": time.time() - callbacks[0].training_start
        }
        with open(save_path / "final_stats.json", "w") as f:
            json.dump(final_stats, f, indent=4)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving checkpoint...")
        model.save(save_path / "interrupted_model")
        env.save(f"{save_path}/interrupted_model_env")
    
    finally:
        # Close environments
        env.close()
        wandb.finish()
        
    return model, env

def evaluate_agent(model, env, n_episodes=100):
    """
    Evaluate a trained agent
    
    Args:
        model: Trained PPO model
        env: Environment
        n_episodes: Number of episodes to evaluate
    """
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if done:
                if info[0].get('success', False):  # Assuming vectorized env
                    success_count += 1
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                break
    
    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "success_rate": success_count / n_episodes
    }
    
    return results

if __name__ == "__main__":
    from main_game import Game
    from game_env import AngryBirdsEnv
    
    # Initialize game and training
    game = Game()
    
    # Train agent
    model, env = train_agent(
        game,
        total_timesteps=1_000_000,
        n_envs=4,
        load_path=None  # Specify path to load existing model
    )
    
    # Evaluate trained agent
    results = evaluate_agent(model, env, n_episodes=100)
    print("\nEvaluation Results:")
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_length']:.2f}")
    print(f"Success Rate: {results['success_rate']:.2%}")