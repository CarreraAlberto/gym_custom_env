import sys
import gymnasium as gym
import numpy as np

from gymnasium_env.grid_world_cpp import GridWorldCPPEnv

# env
gym.register(
    id="GridWorldCPP-v0",
    entry_point="gymnasium_env.grid_world_cpp:GridWorldCPPEnv",
)

# config
GRID_SIZE    = 5
OBS_QUANTITY = 3   # n obstacles
MAX_STEPS    = 200

MODE = sys.argv[1] if len(sys.argv) > 1 else "default"

if MODE == "render":
    N_EPISODES   = 3
    RENDER_MODE  = "human"
elif MODE == "stats":
    N_EPISODES   = 100
    RENDER_MODE  = None
else:
    N_EPISODES   = 10
    RENDER_MODE  = None

env = gym.make(
    "GridWorldCPP-v0",
    render_mode=RENDER_MODE,
    size=GRID_SIZE,
    obs_quantity=OBS_QUANTITY,
    max_steps=MAX_STEPS,
)

episode_coverages = []
episode_rewards   = []
episode_steps     = []

print(f"\n{'='*60}")
print(f"  Coverage Path Planning — Random Agent")
print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}  |  Obstacles: {OBS_QUANTITY}  |  Episodes: {N_EPISODES}")
print(f"{'='*60}\n")

for ep in range(N_EPISODES):
    obs, info = env.reset(seed=ep)
    total_reward = 0.0
    done = False

    while not done:
        action = env.action_space.sample()   # purely random policy
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    coverage_pct = info["coverage_ratio"] * 100
    episode_coverages.append(coverage_pct)
    episode_rewards.append(total_reward)
    episode_steps.append(info["steps"])

    status = "COMPLETE" if terminated else "truncated"
    print(
        f"  Episode {ep+1:3d}  [{status:9s}]  "
        f"coverage={coverage_pct:5.1f}%  "
        f"steps={info['steps']:3d}  "
        f"reward={total_reward:7.2f}"
    )

env.close()

# resultados
if N_EPISODES > 1:
    print(f"\n{'='*60}")
    print(f"  Summary over {N_EPISODES} episodes")
    print(f"{'='*60}")
    print(f"  Coverage   — mean: {np.mean(episode_coverages):.1f}%  "
          f"std: {np.std(episode_coverages):.1f}%  "
          f"max: {np.max(episode_coverages):.1f}%")
    print(f"  Steps      — mean: {np.mean(episode_steps):.1f}  "
          f"std: {np.std(episode_steps):.1f}")
    print(f"  Reward     — mean: {np.mean(episode_rewards):.2f}  "
          f"std: {np.std(episode_rewards):.2f}")
    complete = sum(c >= 99.9 for c in episode_coverages)
    print(f"  Full coverage achieved: {complete}/{N_EPISODES} episodes "
          f"({100*complete/N_EPISODES:.0f}%)")
    print()
