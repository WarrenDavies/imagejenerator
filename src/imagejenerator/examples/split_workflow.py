from imagejenerator.models import registry
from imagejenerator.examples.config import config

image_generator = registry.get_model_class(config)

image_generator.create_pipeline()
image_generator.run_pipeline()
image_generator.save_image()