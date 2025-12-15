MODEL_REGISTRY = {}

def register_model(name: str):
    def wrapper(cls):
        if name in MODEL_REGISTRY:
            raise KeyError(f"Model {name} already registered")
        MODEL_REGISTRY[name] = cls
        return cls
    return wrapper


