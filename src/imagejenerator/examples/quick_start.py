import datetime

from imagejenerator.local.diffusion import registry
from imagejenerator.examples.config import config

image_generator = registry.get_model_class(config)
image_generator.load()
image_generator.prepare()
output = image_generator.generate()
ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
output.batch[0].data.save(f"images/{ts}.png")

