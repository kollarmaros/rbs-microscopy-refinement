import torch
from tqdm import tqdm
import torch.nn.functional as F

from torchvision.utils import save_image
import numpy as np
import random


def set_seed(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, cfg, model, model_vae=None, starting_epoch=0):
        self.datamodule = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        set_seed(cfg.seed, self.device)
        self.model = model.to(self.device)
        print("Device: ", self.device)
        self.cfg = cfg

        # pretrained vae model
        if model_vae is not None:
            self.model_vae = model_vae
            self.model_vae.to(self.device)
            self.model_vae.eval()
            self.image_size = int(self.cfg.image_size / self.model_vae.compress_ratio)

            for param in model_vae.parameters():
                param.requires_grad = False
        else:
            self.model_vae = None
            self.image_size = self.cfg.image_size

        self.latent_channels = cfg.latent_channels

        # optimizer
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.cfg.learning_rate)
        # self.scheduler = StepLR(self.optimizer, step_size=2000, gamma=0.1)

        # get betas from beta scheduler
        self.betas = self._cosine_beta_schedule(timesteps=self.cfg.timesteps)

        # precalculated values
        # define alphas
        self.alphas = 1. - self.betas
        # define alphas cumulative product
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        # storing best value of loss
        self.loss_best_value = 100000

        self.starting_epoch = starting_epoch

    def _extract_t(self, vals, x_shape, t):
        # extract specified values according to timesteps t from precomputed vals
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())

        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def _forward_pass_diffusion_t(self, x_start, t):
        # create noisy version of input images and return them with added noise
        noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract_t(self.sqrt_alphas_cumprod,
                                                x_start.shape,
                                                t,
                                                )
        sqrt_one_minus_alphas_cumprod_t = self._extract_t(self.sqrt_one_minus_alphas_cumprod,
                                                          x_start.shape,
                                                          t,
                                                          )
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

        return x_noisy, noise

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        x = torch.linspace(0, timesteps, timesteps + 1)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

        return torch.clip(betas, 0.0001, 0.02)

    def load_checkpoint(self, file_name):
        checkpoint = torch.load(file_name, map_location=torch.device("cpu"))

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["opt"])
        epochs_checkout = checkpoint['epoch']

        # move to device
        self.model = self.model.to(self.device)
        self.model.eval()

        return epochs_checkout

    def sample_and_save_image(self, index, model, out_path="./samples/", batch_size=1, model_vae=None, channels=1):
        # sample and save image
        img_sample = self._sample_images(model, image_size=self.image_size, batch_size=batch_size, channels=channels)
        #
        if model_vae is not None:
            img_sample = model_vae.decoder(img_sample)

        # put into range -1,1
        img_sample = torch.clamp(img_sample, -1., 1.).detach().cpu()
        # rescale from (-1,1) to (0,1)
        img_sample = (img_sample + 1) * 0.5

        save_image(img_sample, str(out_path + self.cfg.name + f'_sample-{index}.png'), nrow=batch_size)

    def sample_image(self, model, model_vae, mask_image, start_timestep=50):
        # sample and save image
        t = torch.full((1,), start_timestep, device=self.device).long()
        mask_image.to(self.device)

        if model_vae is not None:
            mask_image, _ = model_vae.encoder(mask_image)

        x_noisy, noise = self._forward_pass_diffusion_t(mask_image, t)

        img_sample = self._sample_images_mask(model, x_noisy, start_timestep)

        if model_vae is not None:
            img_sample = model_vae.decoder(img_sample)

        # rescale from (-1,1) to (0,1)
        img_sample = (img_sample + 1) * 0.5
        
        return img_sample

    @torch.no_grad()
    def _sample_images(self, model, image_size, batch_size, channels):
        # generate images
        device = next(model.parameters()).device

        shape = (batch_size, channels, image_size, image_size)

        # start from pure noise (for each example in the batch)
        imgs = torch.randn(shape, device=device)
        # iterate over all timesteps starting from t to 0
        for i in tqdm(reversed(range(0, self.cfg.timesteps)), desc='Sampling loop time step', total=self.cfg.timesteps):
            imgs = self._sample_t(model, imgs, torch.full((batch_size,), i, device=device, dtype=torch.long), i)

        return imgs

    @torch.no_grad()
    def _sample_images_mask(self, model, images, start_timestep):
        # generate images
        device = next(model.parameters()).device

        batch_size = images.shape[0]
        # start from pure noise (for each example in the batch)
        imgs = images
        # imgs = torch.randn(images.shape, device=device)

        # iterate over all timesteps starting from t to 0
        for i in tqdm(reversed(range(0, start_timestep - 1)), desc='Sampling loop time step', total=start_timestep):
            imgs = self._sample_t(model, imgs, torch.full((batch_size,), i, device=device, dtype=torch.long), i)
        return imgs

    @torch.no_grad()
    def _sample_t(self, model, x, t, t_index, condition=None):
        # sample image for a specified timestep
        betas_t = self._extract_t(self.betas, x.shape, t)
        sqrt_one_minus_alphas_cumprod_t = self._extract_t(self.sqrt_one_minus_alphas_cumprod, x.shape, t)
        sqrt_recip_alphas_t = self._extract_t(self.sqrt_recip_alphas, x.shape, t)

        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract_t(self.posterior_variance, x.shape, t)
            noise = torch.randn_like(x)
            if condition is not None:
                # noise = 0.3*condition + 0.7*torch.randn_like(x)
                return model_mean + torch.sqrt(posterior_variance_t) * noise + 0.1*condition

                # noise = torch.cat([torch.randn_like(x), condition], dim=1)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
