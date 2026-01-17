import torch
from torch import autocast
import time
import datetime 

from imagejenerator.core.image_generator import ImageGenerator
from imagejenerator.models.sd_schedulers import schedulers


class BaseDiffusersGenerator(ImageGenerator):
    """
    Base class with common attributes and method needed to run diffusion models locally.

    This class handles configuration, loading and inference using the Hugging Face 
    Diffusers library. It supports custom schedulers, attention slicing for memory 
    optimization, and mixed-precision inference.
    """

    def __init__(self, config):
        """
        Initializes the generator.

        This method expands the 'prompts' list in the config to match the total
        number of images to generate (prompts * images_per_prompt) to facilitate
        batch processing.

        Args:
            config (dict): Configuration dictionary. Must include standard ImageGenerator
                           keys plus model-specific keys.
        """
        super().__init__(config)
        self.model_class = None
        self.pipe = None
        self.images = None
        self.prompts = config["prompts"] * config["images_to_generate"]


    def configure_attention_slicing(self):
        print("configuring attention slicing")
        if self.config.get("enable_attention_slicing", False):
            self.pipe.enable_attention_slicing()


    def configure_scheduler(self):
        print("configuring scheduler")
        if self.config.get("scheduler", False):
            scheduler = schedulers[self.config["scheduler"]]
            self.pipe.scheduler = scheduler.from_config(self.pipe.scheduler.config)


    def configure_vae_tiling(self):
        print("configuring vae tiling")
        if self.config.get("enable_vae_tiling", False):
            self.pipe.enable_vae_tiling()


    def load_model(self):
        self.pipe = self.model_class.from_pretrained(
            self.config["model_path"],
            torch_dtype=self.dtype,
            safety_checker=None,
        ).to(self.device)


    def create_pipeline(self):
        """
        Loads the diffusion pipeline and applies config.

        Steps taken:
        1. Loads the pipeline using `self.model_class.from_pretrained` (self.model_class
           would be defined in the child class, or manually in the program into which 
           imagejenerator has been installed).
        2. Moves the pipeline to the specific device (CPU/CUDA).
        3. Enables attention slicing, applied a scheduler, and applies vae tiling if the
           config says to do so

        Raises:
            KeyError: If specific config keys (like 'model_path') are missing.
        """

        self.load_model()
        self.configure_attention_slicing()
        self.configure_scheduler()
        self.configure_vae_tiling()


    def run_pipeline_impl(self):
        """
        Executes the diffusion inference.

        Runs the pipeline within a `torch.autocast` context to ensure the correct
        precision (e.g., bfloat16) is used on the target device.

        The resulting images are stored in `self.images`.
        """
        with autocast(self.device):    
            self.images = self.pipe(
                self.prompts, 
                height = self.config["height"], 
                width = self.config["width"],
                num_inference_steps = self.config["num_inference_steps"],
                guidance_scale = self.config["guidance_scale"],
                generator=self.generators,
            ).images


    def complete_image_generation_record_impl(self):
        """
        Implementation hook for recording benchmarking data. This is deprecated: see
        the "benchmarker" package in the "jenerationutils" library.
        """
        pass


    def get_runtime_params(self) -> set[str]:
        """
        Returns parameters in the model that, if changed, DO NOT require a teardown and 
        reload of the model.

        Returns:
            Set[str]: A set containing the names of the parameters.     
        """
        return (
            "prompt",
            "negative_prompt",
            "height",
            "width",
            "num_inference_steps",
            "guidance_scale",
            "generator",
        )
