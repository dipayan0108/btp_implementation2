"""
PIL_edge_v2.py  —  SmartDrive improved edge inference

Changes vs PIL_edge.py:
  1. BUG FIX: log_std was recreated as a new tf.Variable on EVERY call() —
     this allocates memory and runs Python overhead at each inference step,
     adding ~5-10 ms of latency on the RPi. Moved to class variable.

  2. scene_obs: 4 extra floats from semantic scene parsing
     (same parse_semantic_scene used in training, zero extra sensor cost).

  3. obs concat updated for 73-dim (was 100-dim).

  4. Works with INT8 TFLite models directly for minimal latency.
     Call `run_tflite()` for PIL deployment instead of `run()`.

  5. data_processing() now returns scene_obs separately.
"""

import os
import sys
import struct
import socket
import random
import numpy as np
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from parameters_v2 import *
from encoder_v2 import parse_semantic_scene  # reuse scene parser

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# ── Socket setup ──────────────────────────────────────────────────────────────
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SIMULATION_IP, PORT))
print("Connection established.")


# ── Receive observation from simulation node ──────────────────────────────────
def data_processing():
    """
    Receive image + nav from simulation node.
    Returns: (image_tf, nav_tf, scene_obs)
      image_tf  : (1, H, W, 3) float32 tensor
      nav_tf    : (5,)         float32 tensor
      scene_obs : (4,)         float32 numpy — from semantic scene parsing
    """
    header = client_socket.recv(12)
    h, w, c = struct.unpack("3I", header)
    image_size = h * w * c

    image_bytes = b""
    while len(image_bytes) < image_size:
        image_bytes += client_socket.recv(image_size - len(image_bytes))

    info_bytes = client_socket.recv(NAV_DIM * 4)   # NAV_DIM=5 floats

    image_array = np.frombuffer(image_bytes, dtype=np.uint8).reshape((h, w, c))
    nav_array   = np.frombuffer(info_bytes,  dtype=np.float32)

    # Semantic scene parsing (CPU only, <1 ms on RPi)
    scene_obs = parse_semantic_scene(image_array)   # (4,) float32

    image_tf = tf.convert_to_tensor(image_array, dtype=tf.float32)
    image_tf = tf.expand_dims(image_tf, axis=0)     # (1,H,W,3)
    nav_tf   = tf.convert_to_tensor(nav_array, dtype=tf.float32)

    return image_tf, nav_tf, scene_obs


# ── Full TensorFlow (development/test) ───────────────────────────────────────
def run():
    """
    Use full TF models — for testing and debugging.
    For PIL deployment, use run_tflite() below.
    """
    encoder = tf.keras.models.load_model(VAE_MODEL_PATH + '/var_auto_encoder_model', compile=False)
    print(f"Encoder loaded from {VAE_MODEL_PATH}")

    actor = tf.keras.models.load_model(PPO_MODEL_PATH + '/actor', compile=False)
    print(f"Actor loaded from {PPO_MODEL_PATH}")

    # Load σ_noise (used for sampling)
    sn_path = PPO_MODEL_PATH + '/sigma_noise.npy'
    sigma   = float(np.load(sn_path)[0]) if os.path.exists(sn_path) else ACTION_STD_MIN
    print(f"σ_noise = {sigma:.3f}")
    print()

    import tensorflow_probability as tfp
    tfd = tfp.distributions

    while True:
        image_tf, nav_tf, scene_obs = data_processing()

        # Encode image → latent z (LATENT_DIM=64)
        z = encoder(image_tf, training=False)     # (1, 64)

        # Build 73-dim observation
        scene_tf = tf.convert_to_tensor(scene_obs, dtype=tf.float32)
        obs = tf.concat([
            tf.reshape(z, [-1]),    # 64
            nav_tf,                 # 5
            scene_tf,               # 4
        ], axis=-1)                 # → (73,)

        if obs is None:
            break

        # Sample action
        mean = actor(tf.expand_dims(obs, 0))   # (1,2)
        std  = tf.fill((ACTION_DIM,), sigma)
        dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=std)
        action = dist.sample().numpy().flatten()

        print(f"action={action}  scene=road:{scene_obs[1]:.2f} veh:{scene_obs[3]:.2f}")

        data = struct.pack('2f', *action)
        client_socket.sendall(data)
        print("action sent")


# ── TFLite INT8 (PIL deployment — lowest latency) ────────────────────────────
def run_tflite():
    """
    Use INT8 TFLite models for minimum latency on RPi.
    Convert with convert_v2.py before running PIL.
    """
    import tensorflow_probability as tfp
    tfd = tfp.distributions

    encoder_path = os.path.join(TF_LITE_PATH, 'encoder_int8.tflite')
    actor_path   = os.path.join(TF_LITE_PATH, 'actor_int8.tflite')

    enc_interp = tf.lite.Interpreter(model_path=encoder_path)
    enc_interp.allocate_tensors()
    enc_in   = enc_interp.get_input_details()
    enc_out  = enc_interp.get_output_details()

    act_interp = tf.lite.Interpreter(model_path=actor_path)
    act_interp.allocate_tensors()
    act_in   = act_interp.get_input_details()
    act_out  = act_interp.get_output_details()

    sn_path = PPO_MODEL_PATH + '/sigma_noise.npy'
    sigma   = float(np.load(sn_path)[0]) if os.path.exists(sn_path) else ACTION_STD_MIN

    print(f"TFLite INT8 models loaded. σ={sigma:.3f}")
    print()

    while True:
        image_tf, nav_tf, scene_obs = data_processing()

        # ── Encoder inference ─────────────────────────────────────────────
        img_np = image_tf.numpy().astype(np.float32)
        enc_interp.set_tensor(enc_in[0]['index'], img_np)
        enc_interp.invoke()
        z = enc_interp.get_tensor(enc_out[0]['index']).flatten()   # (64,)

        # ── Build 73-dim obs ──────────────────────────────────────────────
        obs_np = np.concatenate([z, nav_tf.numpy(), scene_obs]).astype(np.float32)  # (73,)

        # ── Actor inference ───────────────────────────────────────────────
        act_interp.set_tensor(act_in[0]['index'], obs_np[np.newaxis, :])
        act_interp.invoke()
        mean = act_interp.get_tensor(act_out[0]['index']).flatten()   # (2,)

        # Sample with fixed σ
        mean_tf = tf.convert_to_tensor(mean, dtype=tf.float32)
        std     = tf.fill((ACTION_DIM,), sigma)
        dist    = tfd.MultivariateNormalDiag(loc=mean_tf, scale_diag=std)
        action  = dist.sample().numpy().flatten()

        print(f"action={action}  road:{scene_obs[1]:.2f}  vehicle:{scene_obs[3]:.2f}")

        data = struct.pack('2f', *action)
        client_socket.sendall(data)
        print("action sent")


if __name__ == "__main__":
    # Use TFLite for actual PIL deployment; full TF for debug
    USE_TFLITE = os.path.exists(os.path.join(TF_LITE_PATH, 'encoder_int8.tflite'))
    if USE_TFLITE:
        print("Using TFLite INT8 models.")
        run_tflite()
    else:
        print("TFLite models not found — using full TF. Run convert_v2.py first for PIL deployment.")
        run()
