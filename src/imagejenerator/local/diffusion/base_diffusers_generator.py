import torch
import time
import datetime
import random
from abc import ABC, abstractmethod
import gc

from basejenerator.base_generator import BaseGenerator

from imagejenerator.local.diffusion.sd_schedulers import schedulers


class BaseDiffusersGenerator(BaseGenerator):
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
        self.config = config

        self.model = None
        self.DTYPES_MAP = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.dtype = None
        self.device = None
        self.seeds = config["seeds"]
        self.generators = []
        self.batch = self.config["prompts"] * self.config["images_to_generate"]
        self.batch_size = len(self.config["prompts"]) * self.config["images_to_generate"]
        self.prompts = config["prompts"] * config["images_to_generate"]
        self.detect_device_and_dtype()  


    def set_device(self):
        """
        Sets the computation device based on CUDA availability.

        Sets `self.device` to 'cuda' if available, otherwise defaults to 'cpu'.
        """
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"


    def set_dtype(self):
        """
        Sets the torch data type based on the device and configuration.

        If config['dtype'] is "detect":
            - Sets to torch.bfloat16 if device is 'cuda'.
            - Sets to torch.float32 otherwise.
        Otherwise, maps the string config to the actual torch.dtype object in self.DTYPES_MAP.
        """
        if self.config["dtype"] == "detect":
            if self.device == "cuda":
                self.dtype = torch.bfloat16
                self.config["dtype"] = "bfloat16"
            else:
                self.dtype = torch.float32
                self.config["dtype"] = "float32"
            return
        
        self.dtype = self.DTYPES_MAP[self.config["dtype"]]


    def detect_device_and_dtype(self):
        """
        If 'device' or 'dtype' in config are set to "detect", this method attempts
        to choose the optimal settings based on hardware availability (e.g., CUDA).
        """
        if self.config["device"] == "detect":
            self.set_device()
        else:
            self.device = self.config["device"]

        self.set_dtype()


    def create_generators(self):
        """
        Initializes random seeds and PyTorch Generators.

        If seeds are not provided in the config, random seeds are generated
        for the total batch size (number of prompts * images per prompt).
        Populates `self.generators` with `torch.Generator` objects.
        """
        if not self.seeds:
            self.seeds = [self.create_random_seed() for i in range(self.batch_size)]
                
        self.generators = [
            torch.Generator(device=self.device).manual_seed(seed)
            for seed in self.seeds
        ]


    @staticmethod
    def create_random_seed(size: int = 32) -> int:
        """
        Generates a random integer to serve as a seed.

        Args:
            size (int, optional): The bit-size for the random range. Defaults to 32.

        Returns:
            int: A random integer in the range [0, 2**size - 1].
        """
        seed = random.randint(0, (2**size) - 1)
        return seed


    def configure_attention_slicing(self):
        print("configuring attention slicing")
        if self.config.get("enable_attention_slicing", False):
            self.model.enable_attention_slicing()


    def configure_scheduler(self):
        print("configuring scheduler")
        if self.config.get("scheduler", False):
            scheduler = schedulers[self.config["scheduler"]]
            self.model.scheduler = scheduler.from_config(self.model.scheduler.config)


    def configure_vae_tiling(self):
        print("configuring vae tiling")
        if self.config.get("enable_vae_tiling", False):
            self.model.enable_vae_tiling()


    def prepare(self):
        """
        Lifecycle tasks to set up the pipeline for use. Can be used to reset without
        tearing down the pipeline (e.g., reset torch generators)
        """
        self.create_generators()


    @abstractmethod
    def load(self):
        """
        Subclasses must implement their own model loading
        """

    @abstractmethod
    def generate_impl(self):
        """
        Subclasses must implement their own execution.

        Runs the pipeline within a `torch.autocast` context to ensure the correct
        precision (e.g., bfloat16) is used on the target device.

        The resulting images are stored in `self.images`.
        """
        pass


    def teardown(self):
        """
        Deletes the pipeline, empties the torch cache, and forces Python's garbage collector to run. Clears the slate to create another pipeline.
        """

        if self.model is None:
            print("No pipeline found. You cannot teardown that which was not created.")
            return

        del self.model
        self.pipeline = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.seeds = None
        self.generators = []

        gc.collect()


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
