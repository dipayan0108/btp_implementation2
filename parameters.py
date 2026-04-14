# parameters.py  (DrQ version — VAE removed)

# ENCODER PARAMETERS  (replaces VAE)
IM_WIDTH  = 160
IM_HEIGHT = 80
LATENT_DIM = 64          # deterministic conv encoder output dim (64 or 128)

# ACTOR-CRITIC NETWORK PARAMETERS
# obs = latent (64) + nav features (5: throttle, velocity, steer, dist_center, angle)
OBSERVATION_DIM = LATENT_DIM + 5
ACTION_DIM = 2

# HYPERPARAMETERS
ACTION_STD_INIT = 0.2
LEARNING_RATE   = 1e-4
BATCH_SIZE      = 1
POLICY_CLIP     = 0.2
GAMMA           = 0.99
LAMBDA          = 0.95
NO_OF_ITERATIONS = 15
SEED            = 0

# DrQ AUGMENTATION
AUGMENTATION_PAD = 4      # pixels to pad before random crop (DrQ default)

# TRAINING PARAMETERS
TRAIN_TIMESTEPS  = 5e6
EPISODE_LENGTH   = 75000
TEST_EPISODES    = 30
NO_OF_TEST_EPISODES = 10

# SIMULATION PARAMETERS
TOWN              = 'Town02'
CAR_NAME          = 'model3'
NUMBER_OF_VEHICLES   = 30
NUMBER_OF_PEDESTRIAN = 10
CONTINUOUS_ACTION = True
VISUAL_DISPLAY    = True

# PIL / EDGE
SIMULATION_IP = '10.171.4.221'
EDGE_IP = '0.0.0.0'
PORT    = 5000

# PATHS  (VAE paths removed; single encoder saved alongside actor/critic)
RESULTS_PATH    = 'Results_DrQ'
PPO_MODEL_PATH  = f'{RESULTS_PATH}/ppo_model'
ENCODER_PATH    = f'{RESULTS_PATH}/encoder'        # new: conv encoder weights
CHECKPOINT_PATH = f'{RESULTS_PATH}/checkpoints'
LOG_PATH_TRAIN  = f'{RESULTS_PATH}/runs/train'
LOG_PATH_TEST   = f'{RESULTS_PATH}/runs/test'
LOG_PATH_PI     = f'{RESULTS_PATH}/runs/pi_test'
TEST_IMAGES     = f'{RESULTS_PATH}/test_images'
TF_LITE_PATH    = f'{RESULTS_PATH}/tf_lite_models'

CHECKPOINT_LOAD = False
