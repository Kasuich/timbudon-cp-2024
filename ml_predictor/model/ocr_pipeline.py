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

from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import io
import abc
from paddleocr import PaddleOCR
from PIL import Image
from sklearn.metrics import accuracy_score, pairwise_distances
from tqdm import tqdm
import pickle
import yaml
from model.base_model import BaseModel, PredictResult


class Segmentation:
    def __init__(self, weights_yolo_seg_path: str):
        self.model_seg = YOLO(weights_yolo_seg_path)
        
    def get_segmentation(self) -> None:
        result = self.model_seg(self.image, conf=0.7)
        if len(result[0]):
            object_masks = np.array(result[0].masks.xy, dtype=object)
            self.data["segment_points"] = object_masks 
        else:
            self.data["segment_points"] = []

    def make_filter_detect(self):
        image = np.array(self.image)
        mask = self.data["mask"]
        mask_bin = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 0, 255), thickness=5)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mask = 0.5 * (mask > 0) + 0.5
        new_image = (image * mask).astype("int32")
        cv2.imwrite('output_image.jpg', new_image)
    
class Detection:
    def __init__(self, weights_yolo_det_path: str):
        self.model_det = YOLO(weights_yolo_det_path)
        
    def get_detection(self) -> None:
        result = self.model_det(self.image)
        if len(result[0]):
            object_box = np.array(result[0].boxes.xywhn.to("cpu").detach().numpy(), dtype=object)
            self.data["box_xywhn"] = ['\n'.join([f"{0} {x} {y} {w} {h}" for x, y, w, h in object_box])]
        else:
            self.data["box_xywhn"] = []


class OCR(Segmentation, Detection):
    def __init__(self, 
                 weights_yolo_seg_path: str, 
                 weights_yolo_det_path: str,
                 image: Image.Image):
        Segmentation.__init__(self, weights_yolo_seg_path)
        Detection.__init__(self, weights_yolo_det_path)
        self.ocr = PaddleOCR(use_gpu=True, lang="en")  
        self.image = image
        self.data = {}
        self.get_segmentation()
        self.get_detection()
        self.crop_one_img()
        self.ocr_one_img()
        self.make_filter_detect()

    def get_mask(self) -> np.array:
        mask = np.zeros((self.image.size[1], self.image.size[0]), dtype=np.uint8)
        for object in self.data["segment_points"]:
            points = np.array(
                [[x, y] for x, y in object], dtype=np.int32
            )
            mask = cv2.fillPoly(mask, [points], color=255)
        self.data["mask"] = mask
        
        return mask
    
    def crop_one_img(self) -> None:
        mask = (np.array(self.get_mask()) > 0)
        mask = np.expand_dims(mask, axis=-1)
        image = self.image * mask
        if len(self.data["segment_points"]):
            x = np.array([x for obj in self.data["segment_points"] for x, y in obj])
            y = np.array([y for obj in self.data["segment_points"] for x, y in obj])
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
        self.emb_output_folder = "./models_and_logs/embeddings_vit"
        self.test_images_folder = "./models_and_logs/test/images"
        self.train_labels_folder = "./models_and_logs/train/labels"
        self.train_labels_with_text_folder = "./models_and_logs/train/labels_with_text"
        self.config_path = "./models_and_logs/config.yaml"
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

        # test_filenames = self.load_image_filenames(config["test_images_folder"])
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
                results.append([None, None, None, None])

        df = pd.DataFrame(
            results, columns=["Test_Embedding", "Label", "Label_With_Text", "Neighbour"]
        )
        df["Label_With_Text"] = df["Label_With_Text"].map(lambda x: x[:-1])
        df.to_excel(config["output_excel"], index=False)
        logger.info("Saved results to Excel: {}", config["output_excel"])
        return df

def replace_words_by_similarity(label_text: str, text_list: List[str]) -> str:
    words = label_text.split()
    replaced_words = []
    for word in words:
        candidates = [text for text in text_list if len(text) == len(word)]
        if candidates:
            closest_match = min(candidates, key=lambda x: Levenshtein.distance(word, x))
            replaced_words.append(closest_match)
        else:
            replaced_words.append(word)
    return ' '.join(replaced_words)

class Excel:
    def init(self, db_path: str):
        db = pd.read_excel(db_path)
        db["ДетальАртикул"] = db["ДетальАртикул"].map(lambda x: x[1:-1].split(" ")[0] if "ТС" in x else x[1:-1])
        self.db = db

    def get_info_from_db(self, detail_text: str):
        split_idx = detail_text.find(" ")
        part1, part2 = detail_text[:split_idx], detail_text[split_idx + 1:]
        info_art = self.db[self.db["ДетальАртикул"] == part1] 
        info_num = info_art[info_art["ПорядковыйНомер"] == part2]

        if info_num.shape[0]:
            return info_num.iloc[0].to_dict()

        if info_art.shape[0]:
            return info_art.iloc[0].to_dict()
        
        return {"ДетальНаименование": "Не найдено", "ЗаказНомер": "Не найдено", "СтанцияБлок": "Не найдено"}


class OcrPipeline(BaseModel):

    def __init__(self) -> None:
        self.weights_seg = "./models_and_logs/best.pt"
        self.weights_det = "./models_and_logs/best_det.pt"
        self.db_path = "db.xlsx"

    def predict(
        self, image: Image.Image, search_in_data: bool, dist_threshold: float
    ) -> PredictResult:
        ocr = OCR(self.weights_seg, self.weights_det, image)
        dict_text = ocr.get_text()
        model_neighbour = OcrBD()
        result = model_neighbour.predict(
            image, search_in_data=False, dist_threshold=10.5
        )

        with Image.open("output_image.jpg") as img:
            byte_io = io.BytesIO()
            img.save(byte_io, format="JPEG")
            image_bytes = byte_io.getvalue() 

        neighbour_text = result["Label_With_Text"].iloc[0][1:-1]
        new_label_text = replace_words_by_similarity(neighbour_text, dict_text["text_orig_img"])
        excel = Excel(self.db_path)
        info = excel.get_info_from_db(new_label_text)
        res = PredictResult(
            raw_text=new_label_text,
            pred_img=image_bytes,
            attribute1=info["ДетальНаименование"],
            attribute2=info["ЗаказНомер"],
            attribute3=info["СтанцияБлок"]
        )
        return res
