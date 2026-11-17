from diffusers import StableDiffusionXLPipeline
import torch
from torch import autocast
import time
import datetime 

from imagejenerator.registry import register_model
from imagejenerator.local.diffusion.base_diffusers_generator import BaseDiffusersGenerator
from basejenerator.generator_output import GeneratorOutput
from basejenerator.artifacts.pil_artifact import PILArtifact

@register_model("sdxl")
class SDXL(BaseDiffusersGenerator):
    """
    Concrete implementation of ImageGenerator for SDXL, which handles the loading of the model.
    """

    def __init__(self, config):
        """
        Imports the SDXL diffusers class and stores it in self.ModelClass, so that it can be used to call `from_pretrained()`.

        Args:
            config (dict): Configuration dictionary.
        """
        super().__init__(config)
        self.ModelClass = StableDiffusionXLPipeline


    def load(self):
        print("loading model,", self.dtype)
        self.model = self.ModelClass.from_pretrained(
            self.config["model_path"],
            torch_dtype=self.dtype,
            variant="fp16",
        )

        self.model = self.model.to(self.device)


    def generate_impl(self):
        """
        Executes the Stable Diffusion inference.

        Runs the pipeline within a `torch.autocast` context to ensure the correct
        precision (e.g., bfloat16) is used on the target device.

        The resulting images are stored in `self.images`.
        """
        print("running pipe from SDXL Class")
          
        images = self.model(
            self.prompts, 
            height = self.config["height"], 
            width = self.config["width"],
            num_inference_steps = self.config["num_inference_steps"],
            guidance_scale = self.config["guidance_scale"],
            generator=self.generators,
        ).images
        
        item_extras = [{"seed": seed} for seed in self.seeds]
        artifacts = self._quick_wrap(images, item_extras, PILArtifact)

        return GeneratorOutput(artifacts)