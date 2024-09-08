import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        z_map, z_indices, _ = self.vqgan.encode(x)
        z_indices = z_indices.view(z_map.shape[0], -1)
        return z_map, z_indices
        #raise Exception('TODO2 step1-1!')
        #return None
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r**2
        else:
            raise NotImplementedError

##TODO2 step1-3:     
# x: (batch, channel, h, w)       
    def forward(self, x):

        z_map, z_indices = self.encode_to_z(x)
        device=z_indices.device
        m = torch.bernoulli(torch.ones(z_indices.shape, device=device) *0.5).bool()

        m_token = torch.ones(z_indices.shape, device=device).long() * self.mask_token_id 
        new_z_indices = m * m_token + (~m) * z_indices
        logits = self.transformer(new_z_indices)

        return logits, z_indices
        # z_indices ground truth
        # logits transformer predict the probability of tokens
        #raise Exception('TODO2 step1-3!')
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self,mask_func , ratio , mask_num , z_indices, mask ):
        #raise Exception('TODO3 step1-1!')
        z_indices_mask = self.mask_token_id * mask + (~mask) * z_indices
        logit = self.transformer(z_indices_mask)
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        proba = torch.softmax(logit, dim=-1)

        z_indices_predict = self.sample_non_mask_token(logit)
        z_indices_predict =  z_indices_predict * mask + (~mask) * z_indices
        z_indices_predict_prob = proba.gather(-1, z_indices_predict.unsqueeze(-1)).squeeze(-1)
        z_indices_predict_prob = torch.where(mask, z_indices_predict_prob, torch.zeros_like(z_indices_predict_prob) + torch.inf)

        mask_ratio = self.gamma_func(mask_func)(ratio)
        mask_length = torch.floor(mask_num * mask_ratio).long()

        confidence = self.calculate_confidence(z_indices_predict_prob, mask_ratio)
        sorted_confidence = torch.sort(confidence, dim=-1)[0]
        cutoff_threshold = sorted_confidence[:, mask_length].unsqueeze(-1)

        mask_bc = confidence < cutoff_threshold

        return z_indices_predict, mask_bc
    
    def sample_non_mask_token(self, logit):
        """從logit中取樣，確保預測的token不是遮罩token"""
        z_indices_predict = torch.distributions.categorical.Categorical(logits=logit).sample()
        while torch.any(z_indices_predict == self.mask_token_id):
            z_indices_predict = torch.distributions.categorical.Categorical(logits=logit).sample()
        return z_indices_predict
    
    def calculate_confidence(self, z_indices_predict_prob, mask_ratio):
        """計算confidence並加入Gumbel噪聲"""
        gumbel_noise = torch.distributions.gumbel.Gumbel(0, 1).sample(z_indices_predict_prob.shape).to(z_indices_predict_prob.device)
        temperature_adjusted = self.choice_temperature * (1 - mask_ratio)
        confidence = z_indices_predict_prob + temperature_adjusted * gumbel_noise
        return confidence
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    
