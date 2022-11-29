# Initialization and import and tools and environment.
import gym
from stable_baselines3 import DQN       # Agent
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv      # Visualization tools
import highway_env

# Filter annoying future warning due to version.
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

# Define training environment
def train_env():
    env = gym.make('highway-fast-v0')       # Create a highway environment
    env.configure({
        "observation": {
            "type": "GrayscaleObservation",     # Set up observation type as Grayscale
            "observation_shape": (128, 64),     # Set up img shape as 128*64
            "stack_size": 4,                    # Stack 4 img
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,                    # Set zoom ratio
        },
    })
    env.reset()                      # The environment must be reset() for the change of configuration to be effective.
    return env


# Define test environment
def test_env():
    env = train_env()
    env.configure({"policy_frequency": 15, "duration": 20 * 15})        # Set up configuration
    env.reset()
    return env


if __name__ == '__main__':
    # Train
    model = DQN('CnnPolicy', DummyVecEnv([train_env]),          # Create the model with specific features
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=0.7,
                verbose=1,
                tensorboard_log="highway_cnn/")
    model.learn(total_timesteps=int(1e5))               # Set up loop times
    model.save("highway_cnn/model")                     # Save the model parameters and results

    # Record video
    model = DQN.load("highway_cnn/model")

    env = DummyVecEnv([test_env])                       # Fuse several test environment
    video_length = 2 * env.envs[0].config["duration"]       # Set up video attributes
    env = VecVideoRecorder(env, "highway_cnn/videos/",
                           record_video_trigger=lambda x: x == 0, video_length=video_length,
                           name_prefix="dqn-agent")

    obs = env.reset()
    for _ in range(video_length + 1):
        # Predict
        action, _ = model.predict(obs)
        # Take action
        obs, _, _, _ = env.step(action)
        # Render
        env.render()
    env.close()
