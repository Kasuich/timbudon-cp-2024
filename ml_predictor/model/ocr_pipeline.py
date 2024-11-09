import difflib
import logging
import os
import re
from typing import List, Tuple

import cv2
import easyocr
import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import typer
from loguru import logger
from matplotlib.patches import Rectangle
from model.base_model import BaseModel
from paddleocr import PaddleOCR
from PIL import Image
from sklearn.metrics import accuracy_score, pairwise_distances
from tqdm import tqdm

logging.basicConfig(level=logging.ERROR)


class OcrPipeline(BaseModel):

    def __init__(self) -> None:

        df = pd.read_excel("./model/static/orders.xlsx")
        df["ДетальАртикул"] = df["ДетальАртикул"].map(lambda x: x[1:-1])
        df["ДетальАртикул"] = df["ДетальАртикул"].map(
            lambda x: x if "ТС" not in x else x.split()[0]
        )

    # TODO: Do the rest
