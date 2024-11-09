import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import Levenshtein
from ultralytics import YOLO
from typing import Dict, List
from sklearn.neighbors import NearestNeighbors
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import typer
from loguru import logger
from matplotlib.patches import Rectangle

# from model.base_model import BaseModel
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import abc
from paddleocr import PaddleOCR
from PIL import Image
from sklearn.metrics import accuracy_score, pairwise_distances
from tqdm import tqdm
import pickle
import yaml
from model.base_model import BaseModel


class Segmentation:
    def __init__(self, weights_yolo_path: str):
        self.model = YOLO(weights_yolo_path)
        self.data = {}

    def get_segmentation(self) -> None:
        result = self.model(self.image, conf=0.7)
        if len(result[0]):
            object_masks = np.array(result[0].masks.xy, dtype=object)
            self.data["masks"] = object_masks
        else:
            self.data["masks"] = []


class OCR(Segmentation):
    def __init__(self, weights_yolo_path: str, image: Image.Image):
        super().__init__(weights_yolo_path)
        self.ocr = PaddleOCR(use_gpu=True, lang="en")
        self.image = image
        self.get_segmentation()
        self.crop_one_img()
        self.ocr_one_img()

    def get_mask(self) -> np.array:
        mask = np.zeros((self.image.size[1], self.image.size[0]), dtype=np.uint8)
        for object in self.data["masks"]:
            points = np.array([[x, y] for x, y in object], dtype=np.int32)
            mask = cv2.fillPoly(mask, [points], color=255)

        return mask

    def crop_one_img(self) -> None:
        mask = np.array(self.get_mask()) > 0
        mask = np.expand_dims(mask, axis=-1)
        image = self.image * mask
        if len(self.data["masks"]):
            x = np.array([x for obj in self.data["masks"] for x, y in obj])
            y = np.array([y for obj in self.data["masks"] for x, y in obj])
            x_min, x_max = int(min(x)), int(max(x))
            y_min, y_max = int(min(y)), int(max(y))
            self.data["crop_img"] = image[y_min:y_max, x_min:x_max, :]
        else:
            self.data["crop_img"] = image

    def ocr_one_img(self) -> None:
        crop_image = np.array(self.data["crop_img"])
        orig_image = np.array(self.image)

        result = self.ocr.ocr(crop_image, rec=True)
        if result[0]:
            self.data["rec_crop"] = [line[1][0] for line in result[0]]
        else:
            self.data["rec_crop"] = ["None"]

        result = self.ocr.ocr(orig_image, rec=True)
        if result[0]:
            self.data["rec_orig"] = [line[1][0] for line in result[0]]
        else:
            self.data["rec_orig"] = ["None"]

    def get_text(self) -> Dict[str, List[str]]:
        dict_text = {
            "text_orig_img": self.data["rec_orig"],
            "text_crop_img": self.data["rec_crop"],
        }
        return dict_text


class OcrBD:

    def __init__(self) -> None:
        self.model = SentenceTransformer("clip-ViT-B-16")
        self.emb_output_folder = "embeddings_vit"
        self.test_images_folder = "test/images"
        self.train_labels_folder = "train/labels"
        self.train_labels_with_text_folder = "train/labels_with_text"
        self.config_path = "config.yaml"
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        logger.info("Loaded configuration from {}", self.config_path)

    def load_embeddings_from_folder(
        self, folder: str
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        for filename in os.listdir(folder):
            emb_path = os.path.join(folder, filename)

            if os.path.isfile(emb_path):
                with open(emb_path, "rb") as f:
                    embedding = pickle.load(f)

                    if "test" in filename.lower():
                        test_embeddings = embedding
                    elif "train" in filename.lower():
                        train_embeddings = embedding

        return train_embeddings

    def vectorize_img(self, image: Image.Image) -> np.ndarray:
        return [self.model.encode(image)]

    def load_image_filenames(self, images_folder: str) -> List[str]:
        image_filenames = []
        for filename in sorted(os.listdir(images_folder)):
            if filename.lower().endswith(
                ("png", "jpg", "jpeg", "bmp", "gif", "bbox", "txt")
            ):
                image_filenames.append(filename)
        return image_filenames

    def find_nearest_neighbors(
        self,
        test_embeddings: List[np.ndarray],
        train_embeddings: List[np.ndarray],
        n_neighbors: int,
        threshold: float,
    ) -> List[List[int]]:
        test_embeddings = np.array(test_embeddings)
        train_embeddings = np.array(train_embeddings)
        nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree")
        nn.fit(train_embeddings)

        neighbors_indices = []
        for test_emb in test_embeddings:
            distances, indices = nn.kneighbors([test_emb])
            valid_indices = [
                idx
                for dist, idx in zip(distances[0], indices[0])
                if dist > 0 and dist < threshold
            ]

            if valid_indices:
                neighbors_indices.append(valid_indices[0])
            else:
                neighbors_indices.append(None)

        return neighbors_indices

    def load_labels(
        self, labels_folder: str, file_extension: str, train_filenames: List[str]
    ) -> List[str]:
        labels = []
        train_filenames_base = [filename.split(".")[0] for filename in train_filenames]

        for filename in sorted(os.listdir(labels_folder)):
            if (
                filename.split(".")[-1] == file_extension.lstrip(".")
                and filename.split(".")[0] in train_filenames_base
            ):
                with open(os.path.join(labels_folder, filename), "r") as file:
                    # Читаем строки и добавляем `\n`, если его нет
                    content = "".join(
                        line if line.endswith("\n") else line + "\n"
                        for line in file.readlines()
                    )
                    labels.append(content)

        return labels

    def predict(
        self, image: Image.Image, search_in_data: bool, dist_threshold: float
    ) -> PredictResult:
        config = self.config

        train_embeddings = self.load_embeddings_from_folder(config["emb_output_folder"])
        test_embedings = self.vectorize_img(image)

        logger.info("Embeddings were read")

        test_filenames = self.load_image_filenames(config["test_images_folder"])
        train_filenames = self.load_image_filenames(config["train_images_folder"])

        train_labels = self.load_labels(
            config["train_labels_folder"], ".txt", train_filenames
        )
        train_labels_with_text = self.load_labels(
            config["train_labels_with_text_folder"], ".bbox", train_filenames
        )
        logger.info("train_labels and train_labels_with_text were read")

        logger.info("Test image filenames were read")

        n_neighbors = config["n_neighbors"]
        threshold = config["threshold"]
        nearest_neighbors = self.find_nearest_neighbors(
            test_embedings, train_embeddings, n_neighbors, threshold
        )
        logger.info(f"Neighbours were found - {nearest_neighbors}")
        results = []
        for test_idx, neighbors in enumerate(nearest_neighbors):
            if neighbors:
                neighbor_idx = neighbors
                results.append(
                    [
                        test_idx,
                        train_labels[neighbor_idx],
                        train_labels_with_text[neighbor_idx],
                        train_filenames[neighbor_idx],
                    ]
                )
            else:
                results.append([test_filenames[test_idx], None, None, None])

        df = pd.DataFrame(
            results, columns=["Test_Embedding", "Label", "Label_With_Text", "Neighbour"]
        )
        df["Label_With_Text"] = df["Label_With_Text"].map(lambda x: x[:-1])
        df.to_excel(config["output_excel"], index=False)
        logger.info("Saved results to Excel: {}", config["output_excel"])
        return df


class OcrPipeline(BaseModel):

    def __init__(self) -> None:
        self.weights = "../../sergey/runs/segment/train3/weights/best.pt"

    def predict(
        self, image: Image.Image, search_in_data: bool, dist_threshold: float
    ) -> PredictResult:
        ocr = OCR(self.weights, image)
        dict_text = ocr.get_text()
        model_neighbour = OcrBD()
        result = model_neighbour.predict(
            image, search_in_data=False, dist_threshold=10.5
        )
        res = PredictResult(raw_text=result["Label_With_Text"])
        return res
