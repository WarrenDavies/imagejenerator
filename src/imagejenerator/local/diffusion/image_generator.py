from abc import ABC, abstractmethod
import datetime
import time
import os
import random
import gc

import torch

from imagejenerator.config import config


class ImageGenerator(ABC):
    """
    Abstract base class for image generation pipelines using PyTorch.

    This class handles the boilerplate for device detection, data type management,
    random seed generation, image saving, and performance logging. Subclasses
    must implement the actual pipeline creation and execution logic.

    Attributes:
        config (dict): Configuration dictionary containing model parameters, paths, and settings.
        device (str): The device being used ('cuda' or 'cpu').
        dtype (torch.dtype): The tensor data type (e.g., torch.float16, torch.bfloat16).
        seeds (list[int]): List of random seeds used for generation.
        generators (list[torch.Generator]): List of PyTorch generators initialized with seeds.
        images (list): List of generated image objects (usually PIL.Image).
    """

    def __init__(self, config = config):
        """
        Initializes the ImageGenerator with a configuration.

        Args:
            config (dict): A dictionary containing configuration parameters.
                Expected keys include:
                - 'device': 'detect', 'cuda', or 'cpu'.
                - 'dtype': 'detect', 'bfloat16', 'float16', or 'float32'.
                - 'seeds': List of integers or None.
                - 'prompts': List of prompt strings.
                - 'images_to_generate': Int, number of images per prompt.
                - 'image_save_folder': Path to save output images.
        """
        super().__init__(config)
        self.config = config





    @abstractmethod
    def create_pipeline(self):
        """
        Abstract method to initialize the model pipeline.
        
        Subclasses must implement this to load the specific model (e.g., Stable Diffusion)
        and assign it to `self.pipe`.
        """
        pass


    def run_pipeline(self):
        """
        Executes the pipeline implementation.
        """
        self.run_pipeline_impl()
        return [{"artifact": image, "seed": seed} for image, seed in zip(self.images, self.seeds)]


    @abstractmethod
    def run_pipeline_impl(self):
        """
        Abstract method containing the core generation logic.

        Subclasses must implement this to call the model pipeline and populate `self.images`.
        """
        pass
    

    def generate_image(self):
        """
        Main workflow method to generate and save images.

        Steps:
            1. Creates the pipeline.
            2. Runs the pipeline implementation.
            3. Saves the resulting images to disk.
        """
        self.create_pipeline()
        self.run_pipeline()
        self.save_image()


    def save(self):
        """
        Saves generated images to the configured directory.

        Images are saved with a timestamped filename. Updates `self.save_timestamp`.
        """
        self.save_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        for i, image in enumerate(self.images):
            file_name = f"{self.save_timestamp}_no{i}.jpg"
            self.filenames.append(file_name)
            save_path = os.path.join(self.config["image_save_folder"], file_name)
            image.save(save_path)


    def get_metadata(self):

        metadata = []
        for i in range(self.batch_size):
            metadata.append({
                "filename": self.filenames[i],
                "save_path": self.config["image_save_folder"],
                "timestamp": datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
                "model": self.config["model"],
                "device": self.device,
                "dtype": self.config["dtype"],
                "prompt": self.prompts[i],
                "seed": self.seeds[i],
                "height": self.config["height"],
                "width": self.config["width"],
                "inf_steps": self.config["num_inference_steps"],
                "guidance_scale": self.config["guidance_scale"],
            })
        
        return metadata

    
    def get_images(self):
        """
        Returns the images stored in the images attribute, along with their seeds, or an empty list if none have been generated.
        """
        return zip(self.images, self.seeds)


    def teardown(self):
        """
        Deletes the pipeline, empties the torch cache, and forces Python's garbage collector to run. Clears the slate to create
        another pipeline.
        """

        if self.pipe is None:
            print("No pipeline found. You cannot teardown that which was not created.")
            return

        del self.pipe
        self.pipeline = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

