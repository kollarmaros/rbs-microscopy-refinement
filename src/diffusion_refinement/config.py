from argparse import Namespace

# define config
cfg = Namespace(
    # Data module params
    image_size=512,

    # Training params
    learning_rate=0.0001,
    timesteps=1000,
    codebook_size=512,
    latent_channels=4,

    # Generate images params ( Note: also set image_size in data module params, num of timesteps of trained model and codebook_size)
    seed=42,
)
