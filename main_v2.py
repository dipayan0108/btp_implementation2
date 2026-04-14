"""
main_v2.py  —  SmartDrive improved training/testing pipeline

Changes from original main.py:
  - Imports from parameters_v2, encoder_v2, ppo_v2
  - EncodeState  → EncodeStateV2 (β-VAE + semantic scene parsing)
  - PPOAgent     → PPOAgentV2   (bug fixes + σ_noise schedule + grad clipping)
  - Improved reward function (heading factor, progress bonus, steer smoothness)
  - learn() every LEARN_EVERY=5 ep  (was 10)
  - save()  every SAVE_EVERY=25 ep  (was 50)
  - Full CSV logging: per-episode row + per-step row in capture_data()
  - Clean, consistent print statements throughout
  - TensorBoard: VAE KL loss, scene distribution, σ_noise all logged
"""

import os
import sys
import glob
import math
import weakref
import pygame
import time
import random
import csv
import cv2
import pickle
import numpy as np
import pandas as pd
import logging
from datetime import datetime

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
layers = tf.keras.layers

# ── Import from v2 modules ────────────────────────────────────────────────────
from parameters_v2 import *
from encoder_v2    import EncodeStateV2, parse_semantic_scene
from ppo_v2        import PPOAgentV2

# ── GPU check ────────────────────────────────────────────────────────────────
gpu_available = tf.config.list_physical_devices('GPU')
if gpu_available:
    print("=" * 60)
    print("  GPU DETECTED — TensorFlow running on GPU")
    for gpu in gpu_available:
        print(f"  {gpu}")
    print("=" * 60)
else:
    print("=" * 60)
    print("  No GPU — TensorFlow running on CPU")
    print("=" * 60)
print()

# ── CARLA egg import ──────────────────────────────────────────────────────────
try:
    sys.path.append(glob.glob('./carla/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("[WARNING] Could not import CARLA egg. Make sure the path is correct.")

import carla


# ═════════════════════════════════════════════════════════════════════════════
#  CLIENT CONNECTION
# ═════════════════════════════════════════════════════════════════════════════

class ClientConnection:

    def __init__(self):
        self.host    = "localhost"
        self.town    = TOWN
        self.client  = None
        self.port    = 2000
        self.timeout = 20.0

    def setup(self):
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.world  = self.client.load_world(self.town)
            self.world.set_weather(carla.WeatherParameters.CloudyNoon)
            print(f"[CARLA] Connected to server — town: {self.town}")
            return self.client, self.world
        except Exception as e:
            print(f"[CARLA] Connection failed: {e}")
            if self.client and (self.client.get_client_version() !=
                                self.client.get_server_version()):
                print("[CARLA] Version mismatch between client and server!")
            raise


# ═════════════════════════════════════════════════════════════════════════════
#  CARLA ENVIRONMENT
# ═════════════════════════════════════════════════════════════════════════════

class CarlaEnvironment:

    def __init__(self, client, world, town, checkpoint_frequency=100,
                 continuous_action=True):

        self.client               = client
        self.world                = world
        self.blueprint_library    = self.world.get_blueprint_library()
        self.map                  = self.world.get_map()
        self.action_space         = self.get_discrete_action_space()
        self.continous_action_space = True
        self.display_on           = VISUAL_DISPLAY
        self.vehicle              = None
        self.settings             = None
        self.current_waypoint_index  = 0
        self.checkpoint_waypoint_index = 0
        self.fresh_start          = True
        self.checkpoint_frequency = checkpoint_frequency
        self.route_waypoints      = None
        self.town                 = town

        self.camera_obj      = None
        self.env_camera_obj  = None
        self.collision_obj   = None
        self.lane_invasion_obj = None

        self.sensor_list  = list()
        self.actor_list   = list()
        self.walker_list  = list()
        self.create_pedestrians()

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self):
        try:
            if len(self.actor_list) != 0 or len(self.sensor_list) != 0:
                self.client.apply_batch(
                    [carla.command.DestroyActor(x) for x in self.sensor_list])
                self.client.apply_batch(
                    [carla.command.DestroyActor(x) for x in self.actor_list])
                self.sensor_list.clear()
                self.actor_list.clear()

            self.remove_sensors()

            vehicle_bp = self.get_vehicle(CAR_NAME)

            if self.town == "Town07":
                transform        = self.map.get_spawn_points()[20]
                self.total_distance = 750
            elif self.town == "Town02":
                transform        = self.map.get_spawn_points()[30]
                self.total_distance = 500
            else:
                transform        = self.map.get_spawn_points()[12]
                self.total_distance = 500

            self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
            self.actor_list.append(self.vehicle)

            # Camera sensor (semantic segmentation)
            self.camera_obj = CameraSensor(self.vehicle)
            while len(self.camera_obj.front_camera) == 0:
                time.sleep(0.0001)
            self.image_obs = self.camera_obj.front_camera.pop(-1)
            self.sensor_list.append(self.camera_obj.sensor)

            if self.display_on:
                self.env_camera_obj = CameraSensorEnv(self.vehicle)
                self.sensor_list.append(self.env_camera_obj.sensor)

            # Collision sensor
            self.collision_obj     = CollisionSensor(self.vehicle)
            self.collision_history = self.collision_obj.collision_data
            self.sensor_list.append(self.collision_obj.sensor)

            # State initialisation
            self.timesteps              = 0
            self.rotation               = self.vehicle.get_transform().rotation.yaw
            self.previous_location      = self.vehicle.get_location()
            self.distance_traveled      = 0.0
            self.center_lane_deviation  = 0.0
            self.target_speed           = 22.0   # km/h
            self.max_speed              = 35.0
            self.min_speed              = 15.0
            self.max_distance_from_center = 3.0
            self.throttle               = 0.0
            self.previous_steer         = 0.0
            self.velocity               = 0.0
            self.distance_from_center   = 0.0
            self.angle                  = 0.0
            self.distance_covered       = 0.0

            if self.fresh_start:
                self.current_waypoint_index = 0
                self.route_waypoints        = list()
                self.waypoint = self.map.get_waypoint(
                    self.vehicle.get_location(),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving)
                current_waypoint = self.waypoint
                self.route_waypoints.append(current_waypoint)

                for x in range(self.total_distance):
                    if self.town == "Town07":
                        next_wp = (current_waypoint.next(1.0)[0]
                                   if x < 650 else current_waypoint.next(1.0)[-1])
                    elif self.town == "Town02":
                        next_wp = (current_waypoint.next(1.0)[0]
                                   if x <= 100 else current_waypoint.next(1.0)[-1])
                    else:
                        next_wp = (current_waypoint.next(1.0)[-1]
                                   if x < 300 else current_waypoint.next(1.0)[0])
                    self.route_waypoints.append(next_wp)
                    current_waypoint = next_wp
            else:
                waypoint  = self.route_waypoints[
                    self.checkpoint_waypoint_index % len(self.route_waypoints)]
                transform = waypoint.transform
                self.vehicle.set_transform(transform)
                self.current_waypoint_index = self.checkpoint_waypoint_index

            self.navigation_obs = np.array([
                self.throttle, self.velocity,
                self.previous_steer,
                self.distance_from_center,
                self.angle
            ])

            time.sleep(0.5)
            self.collision_history.clear()
            self.episode_start_time = time.time()

            return [self.image_obs, self.navigation_obs]

        except Exception as e:
            print(f"[ENV] Error during reset: {e}")
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, action_idx):
        try:
            self.timesteps  += 1
            self.fresh_start = False

            velocity         = self.vehicle.get_velocity()
            self.velocity    = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6

            if self.continous_action_space:
                steer    = float(np.clip(action_idx[0], -1.0,  1.0))
                throttle = float(np.clip((action_idx[1] + 1.0) / 2.0, 0.0, 1.0))

                smooth_steer    = self.previous_steer * 0.9 + steer    * 0.1
                smooth_throttle = self.throttle       * 0.9 + throttle * 0.1

                self.vehicle.apply_control(carla.VehicleControl(
                    steer=smooth_steer, throttle=smooth_throttle))

                self.previous_steer = steer
                self.throttle       = throttle

            # Traffic light override
            if self.vehicle.is_at_traffic_light():
                tl = self.vehicle.get_traffic_light()
                if tl.get_state() == carla.TrafficLightState.Red:
                    tl.set_state(carla.TrafficLightState.Green)

            self.collision_history = self.collision_obj.collision_data
            self.rotation          = self.vehicle.get_transform().rotation.yaw
            self.location          = self.vehicle.get_location()

            # Waypoint tracking
            waypoint_index = self.current_waypoint_index
            for _ in range(len(self.route_waypoints)):
                next_idx = waypoint_index + 1
                wp  = self.route_waypoints[next_idx % len(self.route_waypoints)]
                dot = np.dot(
                    self.vector(wp.transform.get_forward_vector())[:2],
                    self.vector(self.location - wp.transform.location)[:2])
                if dot > 0.0:
                    waypoint_index += 1
                else:
                    break

            prev_waypoint_index         = self.current_waypoint_index
            self.current_waypoint_index = waypoint_index
            self.current_waypoint = self.route_waypoints[
                self.current_waypoint_index % len(self.route_waypoints)]
            self.next_waypoint = self.route_waypoints[
                (self.current_waypoint_index + 1) % len(self.route_waypoints)]

            self.distance_from_center = self.distance_to_line(
                self.vector(self.current_waypoint.transform.location),
                self.vector(self.next_waypoint.transform.location),
                self.vector(self.location))
            self.center_lane_deviation += self.distance_from_center

            fwd    = self.vector(self.vehicle.get_velocity())
            wp_fwd = self.vector(
                self.current_waypoint.transform.rotation.get_forward_vector())
            self.angle = self.angle_diff(fwd, wp_fwd)

            # ── Improved reward ───────────────────────────────────────────────
            done   = False
            reward = 0.0

            collision_impulse_total = sum(self.collision_history)

            if len(self.collision_history) != 0:
                done     = True
                severity = min(collision_impulse_total / 500.0, 1.0)
                reward   = -5.0 - 5.0 * severity

            elif self.distance_from_center > self.max_distance_from_center:
                done   = True
                reward = -10.0

            elif (self.episode_start_time + 10 < time.time()
                  and self.velocity < 1.0):
                done   = True
                reward = -1.0

            elif self.velocity > self.max_speed:
                done   = True
                reward = -10.0

            else:
                # Shaping factors
                centering_factor = max(
                    1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
                heading_factor   = max(
                    1.0 - abs(self.angle) / np.deg2rad(20), 0.0)

                if self.velocity < self.min_speed:
                    speed_factor = self.velocity / self.min_speed
                elif self.velocity <= self.target_speed:
                    speed_factor = 1.0
                else:
                    speed_factor = max(
                        1.0 - (self.velocity - self.target_speed) /
                              (self.max_speed - self.target_speed), 0.0)

                reward = speed_factor * centering_factor * heading_factor

                # Progress bonus — each new waypoint advanced
                waypoints_advanced = max(
                    self.current_waypoint_index - prev_waypoint_index, 0)
                reward += 0.1 * waypoints_advanced

                # Steering smoothness penalty
                steer_delta = abs(action_idx[0] - self.previous_steer)
                reward     -= 0.05 * steer_delta

            # Terminal: timestep limit
            if self.timesteps >= 2e6:
                done = True

            elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
                print("[ENV] Destination reached — resetting to fresh start.")
                done             = True
                self.fresh_start = True
                if self.checkpoint_frequency is not None:
                    if self.checkpoint_frequency < self.total_distance // 2:
                        self.checkpoint_frequency += 2
                    else:
                        self.checkpoint_frequency        = None
                        self.checkpoint_waypoint_index   = 0

            # Next observation
            while len(self.camera_obj.front_camera) == 0:
                time.sleep(0.0001)
            self.image_obs = self.camera_obj.front_camera.pop(-1)

            normalized_velocity           = self.velocity / self.target_speed
            normalized_distance_from_center = (self.distance_from_center /
                                                self.max_distance_from_center)
            normalized_angle              = abs(self.angle / np.deg2rad(20))
            self.navigation_obs = np.array([
                self.throttle,
                self.velocity,
                normalized_velocity,
                normalized_distance_from_center,
                normalized_angle
            ])

            if done:
                self.center_lane_deviation = (self.center_lane_deviation /
                                              max(self.timesteps, 1))
                self.distance_covered = abs(
                    self.current_waypoint_index - self.checkpoint_waypoint_index)
                for sensor in self.sensor_list:
                    sensor.destroy()
                self.remove_sensors()
                for actor in self.actor_list:
                    actor.destroy()

            return ([self.image_obs, self.navigation_obs],
                    reward, done,
                    [self.distance_covered, self.center_lane_deviation])

        except Exception as e:
            print(f"[ENV] Error during step: {e}")
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()

    # ── Pedestrians ───────────────────────────────────────────────────────────
    def create_pedestrians(self):
        try:
            walker_spawn_points = []
            for _ in range(NUMBER_OF_PEDESTRIAN):
                sp  = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if loc is not None:
                    sp.location = loc
                    walker_spawn_points.append(sp)

            for sp in walker_spawn_points:
                walker_bp = random.choice(
                    self.blueprint_library.filter('walker.pedestrian.*'))
                controller_bp = self.blueprint_library.find('controller.ai.walker')
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                if walker_bp.has_attribute('speed'):
                    walker_bp.set_attribute(
                        'speed',
                        walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    walker_bp.set_attribute('speed', 0.0)
                walker = self.world.try_spawn_actor(walker_bp, sp)
                if walker is not None:
                    ctrl = self.world.spawn_actor(
                        controller_bp, carla.Transform(), walker)
                    self.walker_list.append(ctrl.id)
                    self.walker_list.append(walker.id)

            all_actors = self.world.get_actors(self.walker_list)
            for i in range(0, len(self.walker_list), 2):
                all_actors[i].start()
                all_actors[i].go_to_location(
                    self.world.get_random_location_from_navigation())

        except Exception:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.walker_list])

    # ── NPC vehicles ─────────────────────────────────────────────────────────
    def set_other_vehicles(self):
        try:
            for _ in range(NUMBER_OF_VEHICLES):
                sp = random.choice(self.map.get_spawn_points())
                bp = random.choice(self.blueprint_library.filter('vehicle'))
                v  = self.world.try_spawn_actor(bp, sp)
                if v is not None:
                    v.set_autopilot(True)
                    self.actor_list.append(v)
            print(f"[ENV] {NUMBER_OF_VEHICLES} NPC vehicles spawned in autopilot mode.")
        except Exception:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.actor_list])

    # ── Helpers ───────────────────────────────────────────────────────────────
    def change_town(self, new_town):
        self.world = self.client.load_world(new_town)

    def get_world(self):
        return self.world

    def get_blueprint_library(self):
        return self.world.get_blueprint_library()

    def angle_diff(self, v0, v1):
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        if   angle >  np.pi: angle -= 2 * np.pi
        elif angle <= -np.pi: angle += 2 * np.pi
        return angle

    def distance_to_line(self, A, B, p):
        num   = np.linalg.norm(np.cross(B - A, A - p))
        denom = np.linalg.norm(B - A)
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom

    def vector(self, v):
        if isinstance(v, (carla.Location, carla.Vector3D)):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])

    def get_discrete_action_space(self):
        return np.array([-0.50, -0.30, -0.10, 0.0, 0.10, 0.30, 0.50])

    def get_vehicle(self, vehicle_name):
        blueprint = self.blueprint_library.filter(vehicle_name)[0]
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        return blueprint

    def set_vehicle(self, vehicle_bp, spawn_points):
        sp = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, sp)

    def remove_sensors(self):
        self.camera_obj        = None
        self.collision_obj     = None
        self.lane_invasion_obj = None
        self.env_camera_obj    = None
        self.front_camera      = None
        self.collision_history = None
        self.wrong_maneuver    = None


# ═════════════════════════════════════════════════════════════════════════════
#  SENSORS
# ═════════════════════════════════════════════════════════════════════════════

class CameraSensor:

    def __init__(self, vehicle):
        self.sensor_name  = 'sensor.camera.semantic_segmentation'
        self.parent       = vehicle
        self.front_camera = list()
        world             = self.parent.get_world()
        self.sensor       = self._set_camera_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: CameraSensor._get_front_camera_data(weak_self, image))

    def _set_camera_sensor(self, world):
        bp = world.get_blueprint_library().find(self.sensor_name)
        bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        bp.set_attribute('fov', '125')
        return world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=2.4, z=1.5),
                            carla.Rotation(pitch=-10)),
            attach_to=self.parent)

    @staticmethod
    def _get_front_camera_data(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(carla.ColorConverter.CityScapesPalette)
        placeholder = np.frombuffer(image.raw_data, dtype=np.uint8)
        placeholder = placeholder.reshape((image.width, image.height, 4))
        self.front_camera.append(placeholder[:, :, :3])


class CameraSensorEnv:

    def __init__(self, vehicle):
        pygame.init()
        self.display     = pygame.display.set_mode(
            (600, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.sensor_name = 'sensor.camera.rgb'
        self.parent      = vehicle
        self.surface     = None
        world            = self.parent.get_world()
        self.sensor      = self._set_camera_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: CameraSensorEnv._get_third_person_camera(weak_self, image))

    def _set_camera_sensor(self, world):
        bp = world.get_blueprint_library().find(self.sensor_name)
        bp.set_attribute('image_size_x', '600')
        bp.set_attribute('image_size_y', '600')
        return world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=-4.0, z=2.0),
                            carla.Rotation(pitch=-12.0)),
            attach_to=self.parent)

    @staticmethod
    def _get_third_person_camera(weak_self, image):
        self = weak_self()
        if not self:
            return
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.width, image.height, 4))[:, :, :3]
        arr = arr[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        self.display.blit(self.surface, (0, 0))
        pygame.display.flip()


class CollisionSensor:

    def __init__(self, vehicle):
        self.sensor_name    = 'sensor.other.collision'
        self.parent         = vehicle
        self.collision_data = list()
        world               = self.parent.get_world()
        self.sensor         = self._set_collision_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    def _set_collision_sensor(self, world):
        bp = world.get_blueprint_library().find(self.sensor_name)
        return world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=1.3, z=0.5)),
            attach_to=self.parent)

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        self.collision_data.append(
            math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2))


# ═════════════════════════════════════════════════════════════════════════════
#  CSV LOGGER HELPER
# ═════════════════════════════════════════════════════════════════════════════

class CSVLogger:
    """
    Handles all CSV logging for training and testing.
    Creates two CSVs:
      1. episode_log.csv  — one row per episode
      2. step_log.csv     — one row per step (only written in capture_data mode)
    """

    EPISODE_FIELDS = [
        'episode', 'timestep', 'time_taken_s',
        'reward', 'cumulative_reward',
        'distance_m', 'lane_deviation_avg',
        'speed_avg_kmh', 'sigma_noise',
        'scene_road', 'scene_vehicle', 'scene_pedestrian', 'scene_sidewalk',
        'actor_loss', 'critic_loss', 'kl_loss',
        'done_reason'
    ]

    STEP_FIELDS = [
        'episode', 'step',
        'throttle', 'velocity_kmh', 'norm_velocity',
        'distance_from_center', 'angle_rad',
        'steer_action', 'throttle_action',
        'reward', 'exec_time_ms',
        'scene_road', 'scene_vehicle', 'scene_pedestrian', 'scene_sidewalk'
    ]

    def __init__(self, results_path, mode='train'):
        os.makedirs(results_path, exist_ok=True)
        self.ep_path   = os.path.join(results_path, f'{mode}_episode_log.csv')
        self.step_path = os.path.join(results_path, f'{mode}_step_log.csv')
        self._init_file(self.ep_path,   self.EPISODE_FIELDS)
        self._init_file(self.step_path, self.STEP_FIELDS)

    def _init_file(self, path, fields):
        if not os.path.exists(path):
            with open(path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def log_episode(self, row: dict):
        with open(self.ep_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=self.EPISODE_FIELDS,
                               extrasaction='ignore')
            w.writerow(row)

    def log_step(self, row: dict):
        with open(self.step_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=self.STEP_FIELDS,
                               extrasaction='ignore')
            w.writerow(row)


# ═════════════════════════════════════════════════════════════════════════════
#  TRAIN
# ═════════════════════════════════════════════════════════════════════════════

def train():

    timestep         = 0
    episode          = 0
    cumulative_score = 0.0
    scores           = []
    episodic_lengths = []
    deviation_acc    = 0.0
    distance_acc     = 0.0
    actor_loss_last  = 0.0
    critic_loss_last = 0.0

    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    print()
    print("=" * 60)
    print("  SMARTDRIVE — TRAINING  (v2 pipeline)")
    print(f"  Town        : {TOWN}")
    print(f"  Obs dim     : {OBSERVATION_DIM}  (z={LATENT_DIM} + nav={NAV_DIM} + scene={SCENE_DIM})")
    print(f"  Learn every : {LEARN_EVERY} episodes")
    print(f"  Save every  : {SAVE_EVERY}  episodes")
    print("=" * 60)
    print()

    try:
        client, world = ClientConnection().setup()
    except Exception:
        print("[TRAIN] Aborting — could not connect to CARLA.")
        sys.exit()

    os.makedirs(LOG_PATH_TRAIN, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(LOG_PATH_TRAIN)
    csv_logger     = CSVLogger(RESULTS_PATH, mode='train')

    env     = CarlaEnvironment(client, world, TOWN)
    encoder = EncodeStateV2()

    if CHECKPOINT_LOAD and os.path.exists(
            os.path.join(CHECKPOINT_PATH, 'checkpoint.pickle')):
        print("[TRAIN] Loading from checkpoint...")
        agent = PPOAgentV2()
        episode, timestep, cumulative_score = agent.chkpt_load()
        agent.load()
        agent.prn()
    else:
        print("[TRAIN] Starting fresh.")
        agent = PPOAgentV2()

    print()
    print(f"  Starting at episode={episode}  timestep={timestep}")
    print()

    # ── Training loop ─────────────────────────────────────────────────────────
    while timestep < int(TRAIN_TIMESTEPS):

        observation = env.reset()
        if observation is None:
            print("[TRAIN] reset() returned None — skipping episode.")
            continue

        # Grab scene info at episode start for logging
        scene_info  = encoder.get_scene_description(observation)
        observation = encoder.process(observation)

        current_ep_reward    = 0.0
        speed_acc            = 0.0
        step_count           = 0
        done_reason          = 'timeout'
        t1                   = datetime.now()

        for t in range(EPISODE_LENGTH):

            obs_np        = observation.numpy()
            s_time        = time.time()
            action, mean  = agent(obs_np, train=True)
            e_time        = time.time()

            observation, reward, done, info = env.step(action)

            if observation is None:
                done_reason = 'env_error'
                break

            observation = encoder.process(observation)

            agent.memory.rewards.append(reward)
            agent.memory.dones.append(done)

            timestep          += 1
            step_count        += 1
            current_ep_reward += reward
            speed_acc         += env.velocity

            if done:
                episode    += 1
                done_reason = ('collision'     if len(env.collision_obj.collision_data if env.collision_obj else []) > 0
                               else 'lane_dev' if env.distance_from_center > env.max_distance_from_center
                               else 'destination')
                t2 = datetime.now()
                episodic_lengths.append(abs((t2 - t1).total_seconds()))
                break

        # ── σ_noise decay ─────────────────────────────────────────────────────
        agent.decay_sigma_noise(episode)

        deviation_acc += info[1]
        distance_acc  += info[0]
        scores.append(current_ep_reward)

        if CHECKPOINT_LOAD and episode > 1:
            cumulative_score = ((cumulative_score * (episode - 1)) +
                                current_ep_reward) / episode
        else:
            cumulative_score = float(np.mean(scores))

        ep_time    = abs((datetime.now() - t1).total_seconds())
        speed_avg  = speed_acc / max(step_count, 1)
        kl_loss    = float(encoder.encoder.kl_loss.numpy()) if hasattr(encoder.encoder, 'kl_loss') else 0.0

        # ── Console print ──────────────────────────────────────────────────────
        print(f"[EP {episode:04d}] "
              f"ts={timestep:07d} | "
              f"R={current_ep_reward:+8.2f} | "
              f"AvgR={cumulative_score:+8.2f} | "
              f"Dist={info[0]:5.0f}m | "
              f"Dev={info[1]:.3f} | "
              f"Speed={speed_avg:.1f}km/h | "
              f"σ={agent.sigma_noise:.3f} | "
              f"t={ep_time:.1f}s | "
              f"[{done_reason}]")

        print(f"         Scene → "
              f"road={scene_info.get('road', 0):.2f}  "
              f"vehicle={scene_info.get('vehicle', 0):.2f}  "
              f"ped={scene_info.get('pedestrian', 0):.2f}  "
              f"sidewalk={scene_info.get('sidewalk', 0):.2f}")

        # ── PPO update ─────────────────────────────────────────────────────────
        if episode % LEARN_EVERY == 0 and len(agent.memory.rewards) > 0:
            print(f"[LEARN] Updating policy at episode {episode}...")
            actor_loss_last, critic_loss_last = agent.learn()
            with summary_writer.as_default():
                tf.summary.scalar("Loss/actor",   actor_loss_last,  step=episode)
                tf.summary.scalar("Loss/critic",  critic_loss_last, step=episode)
                tf.summary.scalar("VAE/KL_loss",  kl_loss,          step=episode)

        # ── TensorBoard ───────────────────────────────────────────────────────
        if episode % 5 == 0:
            with summary_writer.as_default():
                tf.summary.scalar("Reward/episodic",   scores[-1],            step=episode)
                tf.summary.scalar("Reward/cumulative", cumulative_score,       step=episode)
                tf.summary.scalar("Reward/avg5",       np.mean(scores[-5:]),  step=episode)
                tf.summary.scalar("Sigma/noise",       agent.sigma_noise,     step=episode)
                tf.summary.scalar("Distance/covered",  distance_acc / 5,      step=episode)
                tf.summary.scalar("Deviation/center",  deviation_acc / 5,     step=episode)
                tf.summary.scalar("Episode/length_s",  np.mean(episodic_lengths), step=episode)
                tf.summary.scalar("Scene/road",        scene_info.get('road', 0),       step=episode)
                tf.summary.scalar("Scene/vehicle",     scene_info.get('vehicle', 0),    step=episode)
                tf.summary.scalar("Scene/pedestrian",  scene_info.get('pedestrian', 0), step=episode)
                tf.summary.scalar("Speed/avg_kmh",     speed_avg,             step=episode)
                summary_writer.flush()
            episodic_lengths = []
            deviation_acc    = 0.0
            distance_acc     = 0.0

        # ── CSV episode log ────────────────────────────────────────────────────
        csv_logger.log_episode({
            'episode':           episode,
            'timestep':          timestep,
            'time_taken_s':      round(ep_time, 2),
            'reward':            round(current_ep_reward, 4),
            'cumulative_reward': round(cumulative_score, 4),
            'distance_m':        info[0],
            'lane_deviation_avg':round(info[1], 4),
            'speed_avg_kmh':     round(speed_avg, 2),
            'sigma_noise':       round(agent.sigma_noise, 4),
            'scene_road':        round(scene_info.get('road', 0), 4),
            'scene_vehicle':     round(scene_info.get('vehicle', 0), 4),
            'scene_pedestrian':  round(scene_info.get('pedestrian', 0), 4),
            'scene_sidewalk':    round(scene_info.get('sidewalk', 0), 4),
            'actor_loss':        round(actor_loss_last, 6),
            'critic_loss':       round(critic_loss_last, 6),
            'kl_loss':           round(kl_loss, 6),
            'done_reason':       done_reason,
        })

        # ── Save ──────────────────────────────────────────────────────────────
        if episode % SAVE_EVERY == 0:
            agent.save()
            agent.chkpt_save(episode, timestep, cumulative_score)

    print()
    print("[TRAIN] Training complete.")
    sys.exit()


# ═════════════════════════════════════════════════════════════════════════════
#  TEST
# ═════════════════════════════════════════════════════════════════════════════

def test():

    timestep = 0
    episode  = 0

    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    print()
    print("=" * 60)
    print("  SMARTDRIVE — TESTING  (v2 pipeline)")
    print(f"  Town : {TOWN}  |  Episodes : {TEST_EPISODES}")
    print("=" * 60)
    print()

    try:
        client, world = ClientConnection().setup()
    except Exception:
        print("[TEST] Aborting — could not connect to CARLA.")
        sys.exit()

    os.makedirs(LOG_PATH_TEST, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(LOG_PATH_TEST)
    csv_logger     = CSVLogger(RESULTS_PATH, mode='test')

    env     = CarlaEnvironment(client, world, TOWN)
    encoder = EncodeStateV2()
    agent   = PPOAgentV2()
    agent.load()
    agent.prn()

    rewards_all   = []
    distances_all = []
    deviations_all= []
    speeds_all    = []
    times_all     = []

    while episode < TEST_EPISODES + 1:

        observation = env.reset()
        if observation is None:
            continue

        scene_info  = encoder.get_scene_description(observation)
        observation = encoder.process(observation)

        current_ep_reward = 0.0
        speed_acc         = 0.0
        step_count        = 0
        t1                = datetime.now()

        for t in range(EPISODE_LENGTH):

            obs_np       = observation.numpy()
            action, mean = agent(obs_np, train=False)
            observation, reward, done, info = env.step(action)

            if observation is None:
                break

            observation = encoder.process(observation)

            timestep          += 1
            step_count        += 1
            current_ep_reward += reward
            speed_acc         += env.velocity

            if done:
                episode += 1
                break

        ep_time   = abs((datetime.now() - t1).total_seconds())
        speed_avg = speed_acc / max(step_count, 1)

        rewards_all.append(current_ep_reward)
        distances_all.append(info[0])
        deviations_all.append(info[1])
        speeds_all.append(speed_avg)
        times_all.append(ep_time)

        print(f"[TEST EP {episode:03d}] "
              f"R={current_ep_reward:+8.2f} | "
              f"Dist={info[0]:5.0f}m | "
              f"Dev={info[1]:.3f} | "
              f"Speed={speed_avg:.1f}km/h | "
              f"t={ep_time:.1f}s")

        print(f"            Scene → "
              f"road={scene_info.get('road',0):.2f}  "
              f"vehicle={scene_info.get('vehicle',0):.2f}  "
              f"ped={scene_info.get('pedestrian',0):.2f}")

        with summary_writer.as_default():
            tf.summary.scalar("Test/Reward",   current_ep_reward, step=episode)
            tf.summary.scalar("Test/Distance", info[0],           step=episode)
            tf.summary.scalar("Test/Deviation",info[1],           step=episode)
            tf.summary.scalar("Test/Speed",    speed_avg,         step=episode)
            summary_writer.flush()

        csv_logger.log_episode({
            'episode':           episode,
            'timestep':          timestep,
            'time_taken_s':      round(ep_time, 2),
            'reward':            round(current_ep_reward, 4),
            'cumulative_reward': round(float(np.mean(rewards_all)), 4),
            'distance_m':        info[0],
            'lane_deviation_avg':round(info[1], 4),
            'speed_avg_kmh':     round(speed_avg, 2),
            'sigma_noise':       round(agent.sigma_noise, 4),
            'scene_road':        round(scene_info.get('road', 0), 4),
            'scene_vehicle':     round(scene_info.get('vehicle', 0), 4),
            'scene_pedestrian':  round(scene_info.get('pedestrian', 0), 4),
            'scene_sidewalk':    round(scene_info.get('sidewalk', 0), 4),
            'actor_loss':        0.0,
            'critic_loss':       0.0,
            'kl_loss':           0.0,
            'done_reason':       'test',
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  TEST SUMMARY")
    print(f"  Episodes       : {TEST_EPISODES}")
    print(f"  Avg Reward     : {np.mean(rewards_all):+.2f} ± {np.std(rewards_all):.2f}")
    print(f"  Avg Distance   : {np.mean(distances_all):.1f} m")
    print(f"  Avg Deviation  : {np.mean(deviations_all):.4f}")
    print(f"  Avg Speed      : {np.mean(speeds_all):.1f} km/h")
    print(f"  Avg Time/ep    : {np.mean(times_all):.1f} s")
    print("=" * 60)
    print()

    sys.exit()


# ═════════════════════════════════════════════════════════════════════════════
#  CAPTURE DATA  (per-step CSV logging for edge evaluation dataset)
# ═════════════════════════════════════════════════════════════════════════════

def capture_data():

    timestep = 0
    episode  = 0

    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    print()
    print("=" * 60)
    print("  SMARTDRIVE — CAPTURE DATA  (v2 pipeline)")
    print(f"  Saving {NO_OF_TEST_EPISODES} episodes to {TEST_IMAGES}")
    print("=" * 60)
    print()

    try:
        client, world = ClientConnection().setup()
    except Exception:
        print("[CAPTURE] Aborting — could not connect to CARLA.")
        sys.exit()

    os.makedirs(TEST_IMAGES, exist_ok=True)
    csv_logger = CSVLogger(RESULTS_PATH, mode='capture')

    env     = CarlaEnvironment(client, world, TOWN)
    encoder = EncodeStateV2()
    agent   = PPOAgentV2()
    agent.load()

    folder_count = 1

    while folder_count <= NO_OF_TEST_EPISODES:

        save_images_dir = os.path.join(TEST_IMAGES, f'Episode_{folder_count:02d}_images')
        os.makedirs(save_images_dir, exist_ok=True)

        print(f"\n[CAPTURE] Episode {folder_count}/{NO_OF_TEST_EPISODES} "
              f"→ saving images to {save_images_dir}")

        observation = env.reset()
        if observation is None:
            print("[CAPTURE] reset() returned None — skipping.")
            folder_count += 1
            continue

        scene_info        = encoder.get_scene_description(observation)
        current_ep_reward = 0.0
        frame_count       = 0
        t1                = datetime.now()

        for t in range(EPISODE_LENGTH):

            # Save raw SS image
            image_array = np.array(observation[0], dtype=np.uint8)
            save_path   = os.path.join(save_images_dir, f"frame_{frame_count:05d}.png")
            cv2.imwrite(save_path, image_array)

            # Get scene for this frame
            scene_info = encoder.get_scene_description(observation)

            # Time the full inference
            s_time      = time.time()
            obs_tensor  = encoder.process(observation)
            obs_np      = obs_tensor.numpy()
            action, mean = agent(obs_np, train=False)
            e_time      = time.time()
            exec_ms     = (e_time - s_time) * 1000.0

            observation, reward, done, info = env.step(action)

            # ── Per-step CSV row ──────────────────────────────────────────────
            csv_logger.log_step({
                'episode':            folder_count,
                'step':               frame_count,
                'throttle':           round(float(env.throttle), 4),
                'velocity_kmh':       round(float(env.velocity), 4),
                'norm_velocity':      round(float(env.velocity / env.target_speed), 4),
                'distance_from_center': round(float(env.distance_from_center), 4),
                'angle_rad':          round(float(env.angle), 4),
                'steer_action':       round(float(action[0]), 4),
                'throttle_action':    round(float(action[1]), 4),
                'reward':             round(float(reward), 4),
                'exec_time_ms':       round(exec_ms, 3),
                'scene_road':         round(scene_info.get('road', 0), 4),
                'scene_vehicle':      round(scene_info.get('vehicle', 0), 4),
                'scene_pedestrian':   round(scene_info.get('pedestrian', 0), 4),
                'scene_sidewalk':     round(scene_info.get('sidewalk', 0), 4),
            })

            if observation is None:
                print(f"[CAPTURE] Env error at step {t} — ending episode.")
                break

            frame_count       += 1
            current_ep_reward += reward

            # Progress print every 50 frames
            if frame_count % 50 == 0:
                print(f"  frame={frame_count:04d} | "
                      f"v={env.velocity:.1f}km/h | "
                      f"R={current_ep_reward:+.2f} | "
                      f"dist={env.distance_from_center:.2f} | "
                      f"t={exec_ms:.1f}ms | "
                      f"road={scene_info.get('road',0):.2f}")

            if done:
                episode += 1
                break

        ep_time = abs((datetime.now() - t1).total_seconds())

        print(f"[CAPTURE] Episode {folder_count} done — "
              f"frames={frame_count} | "
              f"R={current_ep_reward:+.2f} | "
              f"Dist={info[0]:.0f}m | "
              f"t={ep_time:.1f}s")

        csv_logger.log_episode({
            'episode':           folder_count,
            'timestep':          timestep + frame_count,
            'time_taken_s':      round(ep_time, 2),
            'reward':            round(current_ep_reward, 4),
            'cumulative_reward': 0.0,
            'distance_m':        info[0],
            'lane_deviation_avg':round(info[1], 4),
            'speed_avg_kmh':     0.0,
            'sigma_noise':       round(agent.sigma_noise, 4),
            'scene_road':        round(scene_info.get('road', 0), 4),
            'scene_vehicle':     round(scene_info.get('vehicle', 0), 4),
            'scene_pedestrian':  round(scene_info.get('pedestrian', 0), 4),
            'scene_sidewalk':    round(scene_info.get('sidewalk', 0), 4),
            'actor_loss':        0.0,
            'critic_loss':       0.0,
            'kl_loss':           0.0,
            'done_reason':       'capture',
        })

        folder_count += 1

    print()
    print(f"[CAPTURE] All {NO_OF_TEST_EPISODES} episodes saved.")
    print(f"  Images  → {TEST_IMAGES}")
    print(f"  Step CSV → {csv_logger.step_path}")
    print(f"  Ep CSV   → {csv_logger.ep_path}")
    sys.exit()


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        train()
        # test()
        # capture_data()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit()

    finally:
        print("[INFO] Terminating.")
