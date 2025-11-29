MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model_class(config):
    ModelClass = MODEL_REGISTRY[config["model"]]
    image_generator = ModelClass(config)

    return image_generator