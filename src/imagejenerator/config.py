config = {
    "model": "stable-diffusion-v1-5",
    "model_path": "runwayml/stable-diffusion-v1-5",

    "device": "cuda",
    "enable_attention_slicing": True,
    "scheduler": "EulerDiscreteScheduler",

    "height": 512,
    "width": 512,
    "num_inference_steps": 5,
    "guidance_scale": 10,
    "images_to_generate": 1,
    "seed": 100,

    "image_save_folder": "./images/",

    "save_image_gen_stats": True,
    "image_gen_data_path": "./stats.csv",

    "prompts": [
        "A corgi wearing sunglasses"
    ]
}