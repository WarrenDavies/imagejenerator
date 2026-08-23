import torch
import time
import datetime
import random
from abc import ABC, abstractmethod
import gc
import hashlib
import re

from pydantic import BaseModel
from PIL import Image

from basejenerator.base_generator import BaseGenerator
from basejenerator.artifacts.pil_artifact import PILArtifact
from basejenerator.generator_output import GeneratorOutput

from imagejenerator.registry import register_model


@register_model("dummy")
class DummyGenerator(BaseGenerator):
    """
    Dummy image generator class for testing
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
        Sets `self.device` to 'detected' indicating config requested auto-detect.
        """
        self.device = "detected"
        return None


    def set_dtype(self):
        """
        No dtype needed, self.dtype set to None.
        """
        self.dtype = None
        return None


    def detect_device_and_dtype(self):
        """
        If 'device' or 'dtype' in config are set to "detect", this method attempts
        to choose the optimal settings based on hardware availability (e.g., CUDA).
        """
        if self.config["device"] == "detect":
            self.set_device()
        else:
            self.device = None

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
                
        self.generators = []


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


    def load(self):
        """
        Nothing to load
        """
        return None


    def warmup(self):
        return None


    def get_colour_hex_if_exists(self, prompt: str) -> str | None:
        prompt = prompt.strip()

        if re.fullmatch(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})", prompt):
            return prompt

        if re.fullmatch(r"(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})", prompt):
            return f"#{prompt}"

        return None


    def get_rbg_colours_from_prompt(self, prompt: str) -> tuple[int, int, int]:
        prompt_digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        return (prompt_digest[0], prompt_digest[1], prompt_digest[2])


    def prompt_to_colour(self, prompt: str):
        if hex_colour_code := self.get_colour_hex_if_exists(prompt):
            return hex_colour_code

        return self.get_rbg_colours_from_prompt(prompt)


    def generate_impl(self):
        """
        Subclasses must implement their own execution.

        Runs the pipeline within a `torch.autocast` context to ensure the correct
        precision (e.g., bfloat16) is used on the target device.

        The resulting images are stored in `self.images`.
        """
        images = []
        for prompt in self.config["prompts"]:
            colour = self.prompt_to_colour(prompt)
            images.append(Image.new("RGB", (self.config["width"], self.config["height"]), colour))

        item_extras = [{"seed": seed} for seed in self.seeds]
        artifacts = self._quick_wrap(images, item_extras, PILArtifact)
        return GeneratorOutput(artifacts)
        

    def teardown(self):
        """
        Deletes the pipeline, empties the torch cache, and forces Python's garbage collector to run. Clears the slate to create another pipeline.
        """

        if self.model is None:
            print("No pipeline found. You cannot teardown that which was not created.")
            return

        del self.model
        self.model = None

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


    def get_params_schema(self):
        class ParamsSchema(BaseModel):
            dtype: str = ""
            seed: int = 0
            height: int = 0
            width: int = 0
            num_inference_steps: int = 0
            guidance_scale: float = 0
            enable_attention_slicing: bool = True
            scheduler: str = ""

        return ParamsSchema
        