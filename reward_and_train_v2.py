"""
reward_and_train_v2.py  —  SmartDrive improved reward + training loop

HOW TO USE:
  1. In main.py, replace the reward block inside `step()` with
     `compute_reward_v2()` below.
  2. Replace the `train()` function with `train_v2()` below.
  3. Import from new modules:
       from encoder_v2 import EncodeStateV2
       from ppo_v2 import PPOAgentV2
       from parameters_v2 import *

REWARD IMPROVEMENTS:
  - heading_factor: penalises misalignment with lane direction
  - progress_bonus: small bonus per new waypoint reached (stops agent idling)
  - smooth_steer_penalty: penalises large steer changes (smoother driving)
  - graded collision penalty: proportional to impulse (not fixed -10)
  - speed band reward: tighter band around target_speed

TRAINING LOOP IMPROVEMENTS:
  - learn() every LEARN_EVERY=5 episodes (was 10)
  - save() every SAVE_EVERY=25 episodes (was 50)
  - σ_noise decay schedule wired in
  - VAE KL loss logged to TensorBoard separately
  - Scene class distribution logged per episode
"""

import os, sys, time, random, pickle
import numpy as np
import tensorflow as tf
from datetime import datetime
from parameters_v2 import *


# ── Improved reward function ─────────────────────────────────────────────────
# Paste this into CarlaEnvironment.step() replacing the existing reward block.

def compute_reward_v2(velocity, min_speed, target_speed, max_speed,
                      distance_from_center, max_distance_from_center,
                      angle, collision_history, collision_impulse_total,
                      current_waypoint_index, prev_waypoint_index,
                      previous_steer, steer):
    """
    Returns: (reward: float, done: bool)

    velocity              : current speed km/h
    angle                 : heading error vs lane direction (radians)
    collision_impulse_total: sum of impulse magnitudes this step
    current_waypoint_index: for progress bonus
    prev_waypoint_index   : last waypoint index
    previous_steer, steer : for smoothness penalty
    """

    done = False

    # ── Terminal conditions ──────────────────────────────────────────────────
    if len(collision_history) != 0:
        done = True
        # Graded collision penalty: worse crash = bigger penalty (was fixed -10)
        severity = min(collision_impulse_total / 500.0, 1.0)   # normalise, cap at 1
        return -5.0 - 5.0 * severity, done

    if distance_from_center > max_distance_from_center:
        done = True
        return -10.0, done

    if velocity < 1.0:
        # Very slow — but give grace period (handled externally by episode timer)
        return -1.0, done

    if velocity > max_speed:
        done = True
        return -10.0, done

    # ── Shaping factors ──────────────────────────────────────────────────────
    centering_factor = max(1.0 - distance_from_center / max_distance_from_center, 0.0)
    # Heading factor: how well aligned is the vehicle with the lane?
    heading_factor   = max(1.0 - abs(angle) / np.deg2rad(20), 0.0)
    # Speed reward: incentivise staying in [min_speed, target_speed]
    if velocity < min_speed:
        speed_factor = velocity / min_speed
    elif velocity <= target_speed:
        speed_factor = 1.0
    else:
        speed_factor = max(1.0 - (velocity - target_speed) / (max_speed - target_speed), 0.0)

    # ── Base reward ──────────────────────────────────────────────────────────
    reward = speed_factor * centering_factor * heading_factor

    # ── Progress bonus: reward forward waypoint advancement ──────────────────
    # Stops agent from driving slowly in circles to avoid collision penalty
    waypoints_advanced = max(current_waypoint_index - prev_waypoint_index, 0)
    reward += 0.1 * waypoints_advanced

    # ── Steering smoothness penalty ──────────────────────────────────────────
    steer_delta = abs(steer - previous_steer)
    reward -= 0.05 * steer_delta   # small penalty for jerky steering

    return reward, done


# ── Improved training loop ───────────────────────────────────────────────────
# This is a template — integrate the imports at the top of your main.py.

def train_v2():
    """
    Improved train() — same structure as original but with:
    - EncodeStateV2 (β-VAE + scene parsing)
    - PPOAgentV2 (bug fixes + σ_noise schedule)
    - Faster learn/save cadence
    - Full TensorBoard logging including KL loss and scene distribution
    """

    # Deferred import to avoid circular imports when used as patch
    from encoder_v2 import EncodeStateV2
    from ppo_v2    import PPOAgentV2

    timestep        = 0
    episode         = 0
    cumulative_score= 0
    episodic_length = []
    scores          = []
    deviation_from_center = 0
    distance_covered      = 0

    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    # ── Connect to CARLA (same as original) ──────────────────────────────────
    try:
        from main import ClientConnection, CarlaEnvironment   # reuse existing classes
        client, world = ClientConnection().setup()
        print("Connection established.")
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit()

    os.makedirs(LOG_PATH_TRAIN, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(LOG_PATH_TRAIN)

    env     = CarlaEnvironment(client, world, TOWN)
    encoder = EncodeStateV2()          # replaces EncodeState

    if CHECKPOINT_LOAD and os.path.exists(os.path.join(CHECKPOINT_PATH, 'checkpoint.pickle')):
        print("Loading from checkpoint...")
        agent = PPOAgentV2()
        episode, timestep, cumulative_score = agent.chkpt_load()
        agent.load()
        agent.prn()
    else:
        agent = PPOAgentV2()

    # ── Training loop ─────────────────────────────────────────────────────────
    while timestep < TRAIN_TIMESTEPS:

        observation = env.reset()

        # Log scene at episode start
        scene_info = encoder.get_scene_description(observation)
        observation = encoder.process(observation)

        current_ep_reward    = 0
        prev_waypoint_index  = env.current_waypoint_index   # for progress bonus
        t1 = datetime.now()

        for t in range(EPISODE_LENGTH):

            obs_np = observation.numpy()
            action, _ = agent(obs_np, train=True)

            observation, reward, done, info = env.step(action)

            if observation is None:
                break

            observation = encoder.process(observation)

            agent.memory.rewards.append(reward)
            agent.memory.dones.append(done)

            timestep        += 1
            current_ep_reward += reward

            # σ_noise decay check (every 300 episodes)
            agent.decay_sigma_noise(episode)

            if done:
                episode += 1
                t2 = datetime.now()
                episodic_length.append(abs((t2 - t1).total_seconds()))
                break

        deviation_from_center += info[1]
        distance_covered      += info[0]
        scores.append(current_ep_reward)

        if CHECKPOINT_LOAD:
            cumulative_score = ((cumulative_score * (episode - 1)) + current_ep_reward) / episode
        else:
            cumulative_score = np.mean(scores)

        print(f"Ep: {episode}  ts: {timestep}  R: {current_ep_reward:.2f}"
              f"  AvgR: {cumulative_score:.2f}  Dist: {info[0]}"
              f"  σ: {agent.sigma_noise:.3f}")
        print(f"  Scene: road={scene_info.get('road',0):.2f}  "
              f"vehicle={scene_info.get('vehicle',0):.2f}  "
              f"ped={scene_info.get('pedestrian',0):.2f}")

        # ── PPO update ───────────────────────────────────────────────────────
        if episode % LEARN_EVERY == 0:
            actor_loss, critic_loss = agent.learn()
            with summary_writer.as_default():
                tf.summary.scalar("Loss/actor",  actor_loss,  step=episode)
                tf.summary.scalar("Loss/critic", critic_loss, step=episode)
                tf.summary.scalar("VAE/KL_loss", float(encoder.encoder.kl_loss), step=episode)

        # ── TensorBoard ──────────────────────────────────────────────────────
        if episode % 5 == 0:
            with summary_writer.as_default():
                tf.summary.scalar("Reward/episodic",    scores[-1],              step=episode)
                tf.summary.scalar("Reward/cumulative",  cumulative_score,        step=episode)
                tf.summary.scalar("Reward/avg5",        np.mean(scores[-5:]),    step=episode)
                tf.summary.scalar("Sigma/noise",        agent.sigma_noise,       step=episode)
                tf.summary.scalar("Scene/road",         scene_info.get('road',0),step=episode)
                tf.summary.scalar("Scene/vehicle",      scene_info.get('vehicle',0),step=episode)
                tf.summary.scalar("Scene/pedestrian",   scene_info.get('pedestrian',0),step=episode)
                tf.summary.scalar("Distance/covered",   distance_covered / 5,   step=episode)
                tf.summary.scalar("Deviation/center",   deviation_from_center/5, step=episode)
                tf.summary.scalar("Episode/length_s",   np.mean(episodic_length),step=episode)
                summary_writer.flush()
            episodic_length       = []
            deviation_from_center = 0
            distance_covered      = 0

        # ── Save ─────────────────────────────────────────────────────────────
        if episode % SAVE_EVERY == 0:
            agent.save()
            agent.chkpt_save(episode, timestep, cumulative_score)

    sys.exit()
