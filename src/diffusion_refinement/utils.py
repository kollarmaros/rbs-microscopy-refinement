
import torch

from model import VQVAE512ADJ2


# class for storing statistics of training
class Statistics:

    def __init__(self):
        self.values = dict()

    def step(self, key, value):
        sum, count = 0.0, 0.0
        if key in self.values:
            sum, count = self.values[key]
        sum += value
        count += 1.0
        self.values[key] = (sum, count)

    def get(self):
        result = dict()
        for k, (sum, count) in self.values.items():
            result[k] = float(sum/count)
        return result


def model_init(model_pathname, model):
    try:
        model = load_model(model_pathname, model)
        print("Model loading successful!!\n")
    except Exception as e:
        print("Model was not loaded!!\n")
        print(f"Error: An unexpected error occurred: {e}\n")

    return model


def load_model(model_pathname, model):
    # Load model
    checkpoint = torch.load(str(model_pathname))

    # Remove "module." prefix if it exists
    checkpoint_state_dict = checkpoint['model']  # adjust if necessary based on how the checkpoint is saved
    # Initialize a new state dict
    new_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        # Remove 'module.' prefix
        new_key = key[7:] if key.startswith('module.') else key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)

    return model


def load_model_vqvae(model_pathname, codebook_size):
    vae = VQVAE512ADJ2(image_channels=1, codebook_size=codebook_size)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        vae = torch.nn.DataParallel(vae)

    # Remove "module." prefix if it exists
    checkpoint_state_dict = torch.load(model_pathname)  # adjust if necessary based on how the checkpoint is saved
    new_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        new_key = key[7:] if key.startswith('module.') else key
        new_state_dict[new_key] = value

    vae.load_state_dict(new_state_dict, strict=False)
    return vae
