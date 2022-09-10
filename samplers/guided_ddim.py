import json
import logging

import cv2
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from datasets import get_dataloader
from datasets.utils import *
from models import get_sigmas
from models.ncsnv2 import NCSNv2Deepest
from utils import get_all_files

__all__ = ['guided_DDIM']


class guided_DDIM:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = config.device
        self.files = get_all_files(config.data_dir, pattern='*.h5')
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.ssim_scores = []
        self.psnr_scores = []

        os.makedirs(args.log_path, exist_ok=True)

    @torch.no_grad()
    def sample(self):

        # load model
        score = NCSNv2Deepest(self.config).to(self.device)
        score = torch.nn.DataParallel(score)
        states = torch.load(self.config.model_path)
        score.load_state_dict(states[0], strict=True)

        # get dataset
        dataloader = get_dataloader(self.files, self.config, self.args)

        # configure diffusion
        diffusion_timesteps = self.config.model.num_classes
        skip = diffusion_timesteps // self.args.steps
        timesteps = np.arange(0, diffusion_timesteps, skip)
        timesteps_next = list(timesteps)[1:] + [diffusion_timesteps]
        sigmas_torch = get_sigmas(self.config).to(self.device)
        sigmas = torch.cat([sigmas_torch, torch.zeros(1, device=self.device)], dim=0).to(self.device)

        # guided DDIM
        log_interval = 1 if len(timesteps) < 5 else len(timesteps) // 5
        logging.info(f'Total batches {len(dataloader)}')
        for idx, X in enumerate(dataloader):

            ref, mvue, maps, mask = X['ground_truth'], X['mvue'], X['maps'], X['mask']
            ref = ref.to(self.device).type(torch.complex128)
            mvue = mvue.to(self.device)
            maps = maps.to(self.device)
            mask = mask.to(self.device)
            estimated_mvue = torch.tensor(get_mvue(ref.cpu().numpy(), maps.cpu().numpy()), device=self.device)
            forward_operator = lambda x: MulticoilForwardMRI(self.args.orientation)(
                torch.complex(x[:, 0], x[:, 1]),
                maps, mask)

            xt = torch.randn((self.config.batch_size, 2, 384, 384), device=self.device)

            for step, (i, j) in enumerate(zip(timesteps, timesteps_next)):
                if step % log_interval == 0:
                    logging.info(f'Batch: {idx} - Step: {step}')

                t = (torch.ones(self.config.batch_size) * i).to(self.device)
                sigma_t = sigmas[i]
                sigma_t_next = sigmas[j]

                p_grad = score(xt, t.long())

                meas = forward_operator(normalize(xt, estimated_mvue))
                meas_grad = torch.view_as_real(
                    torch.sum(ifft(meas - ref) * torch.conj(maps), axis=1)
                ).permute(0, 3, 1, 2)
                meas_grad = unnormalize(meas_grad, estimated_mvue)
                meas_grad = meas_grad.type(torch.cuda.FloatTensor)
                meas_grad /= torch.norm(meas_grad)
                meas_grad *= torch.norm(p_grad)
                meas_grad *= self.config.sampling.mse

                s = p_grad - meas_grad
                coeff = sigma_t / (1 + sigma_t ** 2).sqrt()
                xt = (xt + coeff * (sigma_t - sigma_t_next) * s).type(torch.cuda.FloatTensor)

            # denoising step
            t_last = (torch.ones(self.config.batch_size) * (diffusion_timesteps - 1)).to(self.device)
            p_grad = score(xt, t_last.long())
            meas = forward_operator(normalize(xt, estimated_mvue))
            meas_grad = torch.view_as_real(
                torch.sum(ifft(meas - ref) * torch.conj(maps), axis=1)
            ).permute(0, 3, 1, 2)
            meas_grad = unnormalize(meas_grad, estimated_mvue).type(torch.cuda.FloatTensor)
            meas_grad /= torch.norm(meas_grad)
            meas_grad *= torch.norm(p_grad)
            meas_grad *= self.config.sampling.mse

            s = p_grad - meas_grad
            H = xt + sigmas[-2] ** 2 * s

            H_norm = normalize(H, estimated_mvue)
            to_display = torch.view_as_complex(
                H_norm.permute(0, 2, 3, 1).reshape(-1, 384, 384, 2).contiguous()
            ).abs().flip(-2)

            for i in range(self.config.batch_size):

                slice_idx = X["slice_idx"][i].item()
                file_name = os.path.join(self.args.log_path, f'{self.config.anatomy}_{slice_idx}.jpg')

                recon_img = to_display[i].unsqueeze(dim=0)
                orig_img = mvue[i].abs().flip(-2)

                orig_th, recon_th, orig_np, recon_np = self.edit(self.config, orig_img, recon_img)
                ssim_score = ssim(orig_np, recon_np)
                psnr_score = psnr(orig_np, recon_np)
                self.ssim_scores.append(ssim_score)
                self.psnr_scores.append(psnr_score)

                if self.args.save_images:

                    file_name = os.path.join(self.args.log_path, f'{self.config.anatomy}_{slice_idx}_or.jpg')
                    save_images(orig_th, file_name, normalize=True)

                    recon_np = Image.fromarray(recon_np)
                    draw = ImageDraw.Draw(recon_np)
                    font = ImageFont.truetype(
                        '/content/image_processing_with_python/09_drawing_text/Gidole-Regular.ttf', 16
                        )
                    draw.text((175, 360), "SSIM: {:0.2f}".format(ssim_score),(255), font=font )
                    draw.text((265, 360), "PSNR: {:0.2f}(db)".format(psnr_score),(255), font=font )
                    file_name = os.path.join(self.args.log_path, f'{self.config.anatomy}_{slice_idx}.jpg')
                    recon_np.save(file_name)

        stats_dict = {'ssim': self.ssim_scores, 'psnr': self.psnr_scores}
        stats_file = os.path.join(self.args.log_path, 'stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats_dict, f, indent=2)

    def edit(self, config, orig_img, recon_img):

        if config.denoise_005:
            recon_img[recon_img <= 0.05 * torch.max(orig_img)] = 0
            orig_img[orig_img <= 0.05 * torch.max(orig_img)] = 0

        orig_np = orig_img.squeeze().cpu().numpy()
        orig_np *= 255.0 / orig_np.max()
        orig_np = orig_np.astype(np.uint8)
        recon_np = recon_img.squeeze().cpu().numpy()
        recon_np *= 255.0 / recon_np.max()
        recon_np = recon_np.astype(np.uint8)

        if config.circle_mask:
            mask = np.zeros(recon_np.shape, dtype=np.uint8)
            cv2.circle(mask, (192, 192), 165, 255, -1)
            recon_np = cv2.bitwise_and(recon_np, recon_np, mask=mask)

        return orig_img, recon_img, orig_np, recon_np
