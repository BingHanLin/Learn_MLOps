from ColaModel import ColaModel
from DataModule import DataModule
from ColaPredictor import ColaPredictor


cola_predictor = ColaPredictor(
    "logs/cola/version_1/checkpoints/epoch=0-step=267.ckpt")

print(cola_predictor.predict("Our friends"))
