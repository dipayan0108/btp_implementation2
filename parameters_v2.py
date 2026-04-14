# parameters_v2.py  —  SmartDrive improved pipeline

# ── VAE ──────────────────────────────────────────────────────────────────────
IM_WIDTH        = 160
IM_HEIGHT       = 80
LATENT_DIM      = 64          # was 95 — smaller latent = faster edge inference
BETA_VAE        = 4.0         # β > 1 forces disentangled, more compact latent
EWC_LAMBDA      = 400.0       # elastic weight consolidation strength (fine-tune)

# ── SEMANTIC SCENE ────────────────────────────────────────────────────────────
# CityScapes tag IDs present in CARLA semantic camera
# road=7, sidewalk=8, vehicle=10, pedestrian=4 → 4-dim scene vector
SEMANTIC_CLASSES  = [4, 7, 8, 10]   # pedestrian, road, sidewalk, vehicle
SCENE_DIM         = len(SEMANTIC_CLASSES)   # 4

# ── ACTOR-CRITIC ──────────────────────────────────────────────────────────────
# obs = 64 (VAE z) + 5 (nav) + 4 (scene) = 73
NAV_DIM          = 5
OBSERVATION_DIM  = LATENT_DIM + NAV_DIM + SCENE_DIM   # 73
ACTION_DIM       = 2

# ── HYPERPARAMETERS ───────────────────────────────────────────────────────────
ACTION_STD_INIT   = 0.4        # start higher, decay to 0.1
ACTION_STD_MIN    = 0.1        # floor for exploration noise
ACTION_STD_DECAY  = 0.05       # reduction per 300 episodes
LEARNING_RATE     = 1e-4
BATCH_SIZE        = 1
POLICY_CLIP       = 0.2
GAMMA             = 0.99
LAMBDA            = 0.95
NO_OF_ITERATIONS  = 15
ENTROPY_COEFF     = 0.01       # initial entropy bonus
ENTROPY_DECAY     = 0.995      # per-update decay
GRAD_CLIP_NORM    = 0.5        # gradient clipping
SEED              = 0

# ── TRAINING ──────────────────────────────────────────────────────────────────
TRAIN_TIMESTEPS   = 5e6
EPISODE_LENGTH    = 75000
LEARN_EVERY       = 5          # was 10 — more frequent updates = faster convergence
SAVE_EVERY        = 25         # was 50

# ── TEST ──────────────────────────────────────────────────────────────────────
TEST_EPISODES        = 30
NO_OF_TEST_EPISODES  = 10

# ── SIMULATION ────────────────────────────────────────────────────────────────
TOWN                = 'Town02'
CAR_NAME            = 'model3'
NUMBER_OF_VEHICLES  = 30
NUMBER_OF_PEDESTRIAN= 10
CONTINUOUS_ACTION   = True
VISUAL_DISPLAY      = True

# ── PIL ───────────────────────────────────────────────────────────────────────
SIMULATION_IP = '10.171.4.221'
EDGE_IP       = '0.0.0.0'
PORT          = 5000

# ── PATHS ─────────────────────────────────────────────────────────────────────
VAR_AUTO_MODEL_PATH = 'VAE'
RESULTS_PATH        = 'Results_v2'
PPO_MODEL_PATH      = f'{RESULTS_PATH}/ppo_model'
VAE_MODEL_PATH      = f'{RESULTS_PATH}/vae_model'
CHECKPOINT_PATH     = f'{RESULTS_PATH}/checkpoints'
LOG_PATH_TRAIN      = f'{RESULTS_PATH}/runs/train'
LOG_PATH_TEST       = f'{RESULTS_PATH}/runs/test'
LOG_PATH_PI         = f'{RESULTS_PATH}/runs/pi_test'
TEST_IMAGES         = f'{RESULTS_PATH}/test_images'
TF_LITE_PATH        = f'{RESULTS_PATH}/tf_lite_models'

CHECKPOINT_LOAD = True
