import datetime

from imagejenerator import registry

config = {
    "model": "stable-diffusion-v1-5",
    "model_path": "runwayml/stable-diffusion-v1-5",

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
        "A rockstar playing a guitar solo on stage"
    ]
}

image_generator = registry.get_model_class(config)
image_generator.load()
image_generator.prepare()
output = image_generator.generate()
ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
output.batch[0].data.save(f"images/{ts}.png")
