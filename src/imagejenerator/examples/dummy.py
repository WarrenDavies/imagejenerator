import datetime
import uuid

from imagejenerator import registry

config = {
    "model": "dummy",
    "model_path": "",

    "device": "cuda",
    "enable_attention_slicing": True,
    "scheduler": "EulerDiscreteScheduler",

    "height": 512,
    "width": 512,
    "num_inference_steps": 30,
    "guidance_scale": 10,
    "images_to_generate": 1,
    "seeds": [], # leave empty for random
    "dtype": "bfloat16",

    "prompts": [
        "A rockstar playing a guitar solo on stage",
        "#3db5ee",
        "e18aab",
    ]
}

image_generator = registry.get_model_class(config)
image_generator.load()
image_generator.prepare()
output = image_generator.generate()
ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
for image in output.batch:
    image.data.save(f"images/{ts}_{str(uuid.uuid4())[:8]}.png")
