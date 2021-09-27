from ColaModel import ColaModel
from DataModule import DataModule
from ColaPredictor import ColaPredictor


cola_predictor = ColaPredictor(
    "logs/cola/version_1/checkpoints/epoch=0-step=267.ckpt")

print(cola_predictor.predict("This is a book."))
print(cola_predictor.predict(
    "Our friends won't buy this analysis, let alone the next one we propose."))
