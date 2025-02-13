from .network import *
from .common import *

MODELS = {"ReconstructionXception": ReconstructionXception, "Differ": Differ, "FakeDiffer": FakeDiffer}


def load_model(name="FakeDiffer"):
    assert name in MODELS.keys(), f"Model name can only be one of {MODELS.keys()}."
    print(f"Using model: '{name}'")
    return MODELS[name]
