from diffusers import StableDiffusionPipeline
import torch
from torch import autocast
import time
import datetime 

from imagejenerator.core.image_generator import ImageGenerator
from imagejenerator.models.sd_schedulers import schedulers


class StableDiffusion_1_5(ImageGenerator):

    def __init__(self, config):
        super().__init__(config)
        self.pipe = None
        self.images = None
        self.prompts = config["prompts"] * config["images_to_generate"]
        self.dtype = torch.float16


    def create_pipeline(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.config["model_path"],
            torch_dtype=self.dtype,
            safety_checker=None
        ).to(self.config["device"])

        if self.config["enable_attention_slicing"]:
            self.pipe.enable_attention_slicing()

        if self.config["scheduler"]:
            scheduler = schedulers[self.config["scheduler"]]
            self.pipe.scheduler = scheduler.from_config(self.pipe.scheduler.config)


    def generate_image_impl(self):
        with autocast(self.config["device"]):    
            self.images = self.pipe(
                self.prompts, 
                height = self.config["height"], 
                width = self.config["width"],
                num_inference_steps = self.config["num_inference_steps"],
                guidance_scale = self.config["guidance_scale"],
            ).images

    def complete_image_generation_record_impl(self):
        pass

