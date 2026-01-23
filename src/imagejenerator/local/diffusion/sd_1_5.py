from diffusers import StableDiffusionPipeline
import torch
from torch import autocast
import time
import datetime 

from basejenerator.generator_output import GeneratorOutput
from imagejenerator.registry import register_model
from imagejenerator.local.diffusion.base_diffusers_generator import BaseDiffusersGenerator
from basejenerator.generator_output import GeneratorOutput

@register_model("stable-diffusion-v1-5")
class StableDiffusion_1_5(BaseDiffusersGenerator):
    """
    Concrete implementation of ImageGenerator for Stable Diffusion v1.5.

    This class handles the loading and inference of the Stable Diffusion v1.5 model
    using the Hugging Face Diffusers library. It supports custom schedulers,
    attention slicing for memory optimization, and mixed-precision inference.
    """


    def __init__(self, config):
        """
        Initializes the Stable Diffusion 1.5 generator.

        This method expands the 'prompts' list in the config to match the total
        number of images to generate (prompts * images_per_prompt) to facilitate
        batch processing.

        Args:
            config (dict): Configuration dictionary. Must include standard ImageGenerator
                           keys plus model-specific keys.
        """
        super().__init__(config)
        self.ModelClass = StableDiffusionPipeline

    def load(self):
        self.model = self.ModelClass.from_pretrained(
            self.config["model_path"],
            torch_dtype=self.dtype,
            safety_checker=None,
        ).to(self.device)

        self.configure_attention_slicing()
        self.configure_scheduler()
        self.configure_vae_tiling()
        self.prepare()


    def generate_impl(self):
        """
        Executes the diffusion inference.

        Runs the pipeline within a `torch.autocast` context to ensure the correct
        precision (e.g., bfloat16) is used on the target device.

        The resulting images are stored in `self.images`.
        """
        with autocast(self.device):    
            images = self.model(
                self.prompts, 
                height = self.config["height"], 
                width = self.config["width"],
                num_inference_steps = self.config["num_inference_steps"],
                guidance_scale = self.config["guidance_scale"],
                generator=self.generators,
            ).images

        item_extras = [{"seed": seed} for seed in self.seeds]
        artifacts = self._quick_wrap(images, item_extras)

        return GeneratorOutput(artifacts)


    def run_pipeline_impl(self):
        """
        Executes the Stable Diffusion inference.

        Runs the pipeline within a `torch.autocast` context to ensure the correct precision (e.g., bfloat16) is used on the target device.

        The resulting images are stored in `self.images`.
        """
        with autocast(self.device):    
            images = self.model(
                self.prompts, 
                height = self.config["height"], 
                width = self.config["width"],
                num_inference_steps = self.config["num_inference_steps"],
                guidance_scale = self.config["guidance_scale"],
                generator=self.generators,
            ).images


    def complete_image_generation_record_impl(self):
        """
        Implementation hook for recording extra stats.

        Currently a no-op for SD 1.5 as the base class records all necessary metadata.
        """
        pass

