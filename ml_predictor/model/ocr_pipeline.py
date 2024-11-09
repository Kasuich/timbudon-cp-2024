from model.base_model import BaseModel
from typing import List, Tuple
import torch
from matplotlib.patches import Rectangle
from PIL import Image
import easyocr
import pandas as pd
from typing import List
import torch
from PIL import Image
import numpy as np
import cv2
import typer
from loguru import logger
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from paddleocr import PaddleOCR
import difflib
import Levenshtein
from sklearn.metrics import pairwise_distances
import logging
import matplotlib.pyplot as plt
import re

logging.basicConfig(level=logging.ERROR)


class OcrPipeline(BaseModel):
    
