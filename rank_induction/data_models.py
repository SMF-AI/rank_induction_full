from __future__ import annotations

import os
import queue
import glob
import numbers
import multiprocessing
from enum import Enum
from logging import Logger
from typing import List, Iterable, Tuple, Set, Callable, Literal
from dataclasses import dataclass, field
from multiprocessing import Process, Queue

import cv2
import torch
import h5py
import shapely.errors
import tqdm
import shapely
import openslide
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from skimage.filters import threshold_multiotsu
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from openslide import open_slide, OpenSlide
from openslide.deepzoom import DeepZoomGenerator

from rank_induction.misc import read_json, get_foreground_tile_address

Image.MAX_IMAGE_PIXELS = None


class Centers(Enum):
    """Medical centers
    Reference:
        https://academic.oup.com/gigascience/article/7/6/giy065/5026175

    """

    RUMC = 0
    CWZ = 1
    UMCU = 2
    RST = 3
    LPON = 4


@dataclass
class Coordinates:
    x_min: int = None
    y_min: int = None
    x_max: int = None
    y_max: int = None

    def __repr__(self) -> str:
        return f"Coordinates(x_min={self.x_min}, y_min={self.y_min}, x_max={self.x_max}, y_max={self.y_max})"

    def to_string(self) -> str:
        return f"{self.x_min}_{self.y_min}_{self.x_max}_{self.y_max}"

    def to_polygon(self) -> Polygon:
        return Polygon(
            [
                (self.x_min, self.y_min),
                (self.x_max, self.y_min),
                (self.x_max, self.y_max),
                (self.x_min, self.y_max),
            ]
        )

    def to_coco(self, size: tuple) -> List[float, float, float, float]:
        w = self.x_max - self.x_min
        h = self.y_max - self.y_min
        return [
            self.x_min / size[0],
            self.y_min / size[1],
            w / size[0],
            h / size[1],
        ]

    def to_list(self) -> List[float, float, float, float]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]


@dataclass
class Polygons:
    path: str = str()
    data: List[Polygon] = field(default_factory=list)

    """Polygons
    - polygons의 집합(=Annotation set)
    
    Attributes
        - path: annotation path
        - data: annotation polygons
        
    Example:
        >>> from seedp.data_models import Polygons
        >>> qupath_polygons = Polygons.from_qupath_json(
                "heon/datasets/seedp/thyroid/BHS/3_5_135528.json"    
            )
        >>> print(qupath_polygons)
        Polygons(path=heon/datasets/seedp/thyroid/BHS/3_5_135528.json, N polygons=20)
        
        >>> qupath_polygons.data
        [
            <POLYGON ((55487 30457, 55450 30458, 55403 30463, 55393 30467, 55357 30467, ...>,
            <POLYGON ((57594 43974, 57590 43975, 57580 43979, 57576 43980, 57566 43984, ...>,
            ...
            <POLYGON ((58657 42320, 58637 42324, 58634 42326, 58617 42342, 58612 42346, ...>,
        ]
        >>> print(qupath_polygons.data[0])
        POLYGON ((55487 30457, ... 55403 31374))
    """

    @classmethod
    def _convert_qupath_json_to_polygons(cls, path: str) -> List[Polygon]:
        """JSON path을 입력받아 ploygon으로 변환
        QuPath Annotation JSON 구조
        [
            {
                "type": "Polygon",
                "coordinates": [ # 3중
                    [
                        [x1, y1],
                        [x2, y2],
                        ...
                    ]
                ]
            },
            {
                "type": "MultiPolygon",
                "coordinates": [ # 4중
                    [
                        [
                            [x1, y1],
                            [x2, y2],
                            ...
                        ]
                    ],
                    [
                        [
                            [x1, y1],
                            [x2, y2],
                            ...
                        ]
                    ]
                ]
            }
        ]
        """
        annotations: List[dict] = read_json(path)
        if isinstance(annotations, dict):
            annotations = [annotations]

        polygons = []
        for i, geometry_info in enumerate(annotations):
            if geometry_info["type"] == "Polygon":
                coordinates = geometry_info["coordinates"]  # 3중 list
                shell = [xy for coordinate in coordinates for xy in coordinate]

            elif geometry_info["type"] == "MultiPolygon":
                multi_coordinates = geometry_info["coordinates"]  # 4중 list x N개
                shell = list()
                for coordinates in multi_coordinates:
                    shell += [xy for coordinate in coordinates for xy in coordinate]

            try:
                polygons.append(Polygon(shell))
            except TypeError as e:
                print(f"{e}, passed shell({shell})")

        return polygons

    @classmethod
    def from_qupath_json(cls, path: str) -> Polygons:
        """JSON을 이용하여 Polygons

        Args:
            path (str): QuPath로 생성된 Annotation의 JSON output

        Returns:
            Polygons

        Example:
            >>> from seedp.data_models import Polygons
            >>> qupath_polygons = Polygons.from_json(
                    "heon/datasets/seedp/thyroid/BHS/3_5_135528.json"
                )
            >>> print(qupath_polygons)
            Polygons(path=heon/datasets/seedp/thyroid/BHS/3_5_135528.json, N polygons=20)
        """
        polygons = cls._convert_qupath_json_to_polygons(path)
        return Polygons(path=path, data=polygons)

    @classmethod
    def from_polygons(cls, polygons: List[Polygon]) -> Polygons:
        return Polygons(data=polygons)

    def __repr__(self) -> str:
        n_polygon: int = len(self.data) if self.data else 0
        return f"QuPathPloygon(path={self.path}, N polygons={n_polygon})"


class BinaryLabels(Enum):
    """이진분류 라벨 상수 표현의 열거형

    Example:
        # class index를 이용하여 라벨 추출
        >>> from seedp.data_models import BinaryLabels
        >>> y_pred = torch.tensor([[0.1, 0.9]])
        >>> class_idx = torch.argmax(y_pred).item()
        >>> print(BinaryLabels(class_idx).name)
        'M'

        # class name을 이용하여, 인덱스 추출
        >>> slide_label:str = os.path.basename(slide_path)
        >>> print(slide_label)
        'N'
        >>> BinaryLabels[slide_label]
        <BinaryLabels.N: 0>
        >>> BinaryLabels[slide_label].value
        0


    """

    N = 0
    M = 1


class Labels(Enum):
    benign = "benign"
    malignant = "malignant"
    unknown = "unknown"


# TODO
@dataclass
class Patch:
    """패치의 데이터클레스

    Args:
        address (tuple): col, row
        coordinates (Coordinates): Slide level 0에서의 x1, y1, x2, y2

    Example:
        # Array로부터 패치 생성
        >>> image_array:np.ndarray = ....
        >>> patch = Patch(image_array)
        >>> patch
        Patch(image_shape=(512, 512, 3), path=None, label=Labels.unknown, coordinates=None, level=None)

        # 파일로부터 생성
        >>> patch = Patch.from_file("test_path.png", label=Labels.unknown)
        >>> patch
        Patch(image_shape=(512, 512, 3), path=None, label=Labels.unknown, coordinates=None, level=None)
        >>> patch.close() # 메모리 해제


        # 좌표와 함께 생성
        >>> patch = Patch(label=Labels.unknown, coordinates=Coordinates(100, 105, 200, 200))
        >>> patch.coordinates.x_min
        100
        >>> patch.coordinates.y_min
        105
        >>> patch.coordinates.x_max
        200

    """

    image_array: np.ndarray = None
    confidences: np.ndarray = None
    coordinates: Coordinates = None
    feature: torch.Tensor = None
    address: Tuple[int, int] = None
    label: str = None

    slide_name: str = None
    path: str = None
    level: int = None

    @classmethod
    def from_file(cls, path: str, **kwargs: dict) -> Patch:
        image_array = np.array(Image.open(path))
        return Patch(image_array, **kwargs)

    def __repr__(self) -> str:
        image_shape = (
            "None" if self.image_array is None else str(self.image_array.shape)
        )
        return (
            "Patch("
            f"image_shape={image_shape}, path={self.path}, label={self.label}, "
            f"coordinates={str(self.coordinates)}, address={self.address}, "
            f"confidences={str(self.confidences)}"
            ")"
        )

    def load(self):
        if not self.path:
            raise ValueError("path is None")

        self.image_array = np.array(Image.open(self.path))

        return

    def close(self):
        del self.image_array

        return


# TODO
@dataclass
class Patches:
    """패치의 복수의 집합"""

    data: List[Patch] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    dimension: tuple = None

    def __getitem__(self, i: int) -> Patch:
        return self.data[i]

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self):
        return f"Patches(N={len(self.data)})"

    def build_feature_cube_pad(self, size: tuple) -> torch.Tensor:
        """패딩이 필요한 feature cube 생성(패딩영역은 마지막 채널)

        Args:
            size (tuple): size (rows, cols)

        Returns:
            torch.Tensor: feature cube (1, 3, row, col)
        """
        feature_cube = torch.zeros((1, 3, *size), dtype=torch.float32)
        for patch in self.data:
            col, row = patch.address
            if patch.confidences is None:
                confidences = torch.tensor([0, 0, 0], dtype=torch.float32)
                feature_cube[0, :, row, col] = confidences
                continue

            feature_cube[0, :, row, col] = torch.concat(
                [
                    torch.from_numpy(patch.confidences).float(),
                    torch.tensor([0], dtype=torch.float32),
                ],
            )

        return feature_cube

    def build_feature_cube(self, size: tuple) -> torch.Tensor:
        """패딩이 필요없는 feature cube 생성(패딩영역은 마지막 채널)

        Args:
            size (tuple): size (rows, cols)

        Returns:
            np.ndarray: feature cube
        """
        feature_cube = torch.zeros((1, 3, *size), dtype=torch.float32)
        for patch in self.data:
            col, row = patch.address
            if patch.confidences is None:
                confidences = torch.tensor([0, 0, 0], dtype=torch.float32)
                feature_cube[0, :, row, col] = confidences
                continue

            feature_cube[0, :, row, col] = torch.from_numpy(patch.confidences).float()

        return feature_cube

    @classmethod
    def from_queue(cls, queue: queue.Queue, drop_empty_patch: bool = True) -> Patches:
        """병렬처리시 Queue로부터 patches을 생성함

        Args:
            queue (Queue): 병렬처리할 때 사용된 shared memory queue
            drop_empty_patch (bool): 패치이미지중 필터된 이미지의 삭제 여부
                (True: 해당 패치 데이터클레스 삭제)

        Returns:
            patches (Patches): 패치의 복수의 집합

        Example:

            >>> tiler = Tiler(
                    config.tile_size,
                    config.overlap,
                    config.limit_bounds,
                    deepzoom_level=deepzoom_level,
                    n_workers=config.tile_workers,
                    patch_filter=patch_filter,
                    logger=logger,
                )
            >>> output_queue: queue.Queue = tiler.do_tile(query.slide_path)
            >>> tiler.join() # 병렬처리 종료대기
            >>> patches = Patches.from_queue(output_queue)

        """
        data = list()
        while not queue.empty():
            patch = queue.get()

            if not drop_empty_patch:
                data.append(patch)
                continue

            if patch.image_array is not None:
                data.append(patch)

        return Patches(data)

    def save(self, dir: str, format: Literal["png", "h5"]) -> None:
        """패치들을 저장

        Example:
            >>> # PNG로 저장
            >>> patches.save("slide_name", format="png")
            >>> os.listdir("slide_name")
            [
                "slide_name_15_23.png",
                "slide_name_15_24.png",
                ...
            ]

            >>> # h5로 저장
            >>> patches.save("slide_name.h5", format="h5")

            >>> # h5로부터 로드
            >>> from seedp.data_models import Patches
            >>> patches = Patches.from_h5("slide_name.h5")

        Args:
            dir (str): 목적지 경로
            format (Literal["png", "h5"], optional): 저장포맷. Defaults to "h5".
        """
        if format == "png":
            os.makedirs(dir, exist_ok=True)
            for patch in self.data:
                Image.fromarray(patch.image_array).save(
                    os.path.join(
                        dir,
                        f"{patch.slide_name}_{patch.address[0]}_{patch.address[1]}.png",
                    )
                )

        elif format == "h5":
            with h5py.File(dir, "w") as fh:
                for patch in self.data:
                    col, row = patch.address
                    key = f"{patch.address[0]}_{patch.address[1]}"
                    dataset = fh.create_dataset(
                        key, data=patch.image_array, compression="gzip"
                    )
                    dataset.attrs["col"] = col
                    dataset.attrs["row"] = row
                    dataset.attrs["x_min"] = patch.coordinates.x_min
                    dataset.attrs["y_min"] = patch.coordinates.y_min
                    dataset.attrs["x_max"] = patch.coordinates.x_max
                    dataset.attrs["y_max"] = patch.coordinates.y_max

        return

    def close(self) -> None:
        for patch in self.data:
            del patch.image_array
        return

    @classmethod
    def from_patch_h5(cls, path: str) -> Patches:
        """HDF5 파일에서 Patches 객체를 생성

        Args:
            path (str): HDF5 파일 경로

        Returns:
            Patches: 복원된 Patches 객체

        Example:
            >>> from seedp.data_models import Patches
            >>> patches = Patches.from_patch_h5("***.h5")
            Patches(length=585)
        """
        data: List[Patch] = []

        with h5py.File(path, "r") as fh:
            for key in fh.keys():
                dataset = fh[key]

                # 이미지 배열을 불러오기
                image_array = np.array(dataset)

                # address (col, row) 복원
                col = dataset.attrs.get("col")
                row = dataset.attrs.get("row")
                address = (col, row)

                # coordinates 복원
                coordinates = Coordinates(
                    x_min=dataset.attrs.get("x_min"),
                    y_min=dataset.attrs.get("y_min"),
                    x_max=dataset.attrs.get("x_max"),
                    y_max=dataset.attrs.get("y_max"),
                )
                patch = Patch(
                    image_array=image_array, address=address, coordinates=coordinates
                )
                data.append(patch)

        return cls(data=data)

    @property
    def features(self) -> torch.Tensor:
        """
        Example:
        >>> patches = Patches.from_feature_h5("here.feature.h5")
        >>> print(patches.features.shape)
        torch.Size([585, 768])
        """
        return torch.stack([patch.feature for patch in self.data], dim=0)

    @classmethod
    def from_feature_h5(cls, path: str) -> Patches:
        """HDF5 파일에서 Patches 객체를 생성

        Args:
            path (str): HDF5 파일 경로

        Returns:
            Patches: 복원된 Patches 객체
        """
        data: List[Patch] = []

        with h5py.File(path, "r") as fh:
            for key in fh.keys():
                dataset = fh[key]

                # 이미지 배열을 불러오기
                feature = torch.from_numpy(np.array(dataset))

                # address (col, row) 복원
                col = dataset.attrs.get("col")
                row = dataset.attrs.get("row")
                address = (col, row)

                # coordinates 복원
                coordinates = Coordinates(
                    x_min=dataset.attrs.get("x_min"),
                    y_min=dataset.attrs.get("y_min"),
                    x_max=dataset.attrs.get("x_max"),
                    y_max=dataset.attrs.get("y_max"),
                )
                patch = Patch(
                    image_array=None,
                    feature=feature,
                    address=address,
                    coordinates=coordinates,
                )
                data.append(patch)

        return cls(data=data)


@dataclass
class Polygons:
    path: str = str()
    data: List[Polygon] = field(default_factory=list)

    """Polygons
    - polygons의 집합(=Annotation set)
    
    Attributes
        - path: annotation path
        - data: annotation polygons
        
    Example:
        >>> from seedp.data_models import Polygons
        >>> qupath_polygons = Polygons.from_qupath_json(
                "heon/datasets/seedp/thyroid/BHS/3_5_135528.json"    
            )
        >>> print(qupath_polygons)
        Polygons(path=heon/datasets/seedp/thyroid/BHS/3_5_135528.json, N polygons=20)
        
        >>> qupath_polygons.data
        [
            <POLYGON ((55487 30457, 55450 30458, 55403 30463, 55393 30467, 55357 30467, ...>,
            <POLYGON ((57594 43974, 57590 43975, 57580 43979, 57576 43980, 57566 43984, ...>,
            ...
            <POLYGON ((58657 42320, 58637 42324, 58634 42326, 58617 42342, 58612 42346, ...>,
        ]
        >>> print(qupath_polygons.data[0])
        POLYGON ((55487 30457, ... 55403 31374))
    """

    @classmethod
    def _convert_qupath_json_to_polygons(cls, path: str) -> List[Polygon]:
        """JSON path을 입력받아 ploygon으로 변환
        QuPath Annotation JSON 구조
        [
            {
                "type": "Polygon",
                "coordinates": [ # 3중
                    [
                        [x1, y1],
                        [x2, y2],
                        ...
                    ]
                ]
            },
            {
                "type": "MultiPolygon",
                "coordinates": [ # 4중
                    [
                        [
                            [x1, y1],
                            [x2, y2],
                            ...
                        ]
                    ],
                    [
                        [
                            [x1, y1],
                            [x2, y2],
                            ...
                        ]
                    ]
                ]
            }
        ]
        """
        annotations: List[dict] = read_json(path)
        if isinstance(annotations, dict):
            annotations = [annotations]

        polygons = []
        for i, geometry_info in enumerate(annotations):
            if geometry_info["type"] == "Polygon":
                coordinates = geometry_info["coordinates"]  # 3중 list
                shell = [xy for coordinate in coordinates for xy in coordinate]

            elif geometry_info["type"] == "MultiPolygon":
                multi_coordinates = geometry_info["coordinates"]  # 4중 list x N개
                shell = list()
                for coordinates in multi_coordinates:
                    shell += [xy for coordinate in coordinates for xy in coordinate]

            try:
                polygons.append(Polygon(shell))
            except TypeError as e:
                print(f"{e}, passed shell({shell})")

        return polygons

    @classmethod
    def from_qupath_json(cls, path: str) -> Polygons:
        """JSON을 이용하여 Polygons

        Args:
            path (str): QuPath로 생성된 Annotation의 JSON output

        Returns:
            Polygons

        Example:
            >>> from seedp.data_models import Polygons
            >>> qupath_polygons = Polygons.from_json(
                    "heon/datasets/seedp/thyroid/BHS/3_5_135528.json"
                )
            >>> print(qupath_polygons)
            Polygons(path=heon/datasets/seedp/thyroid/BHS/3_5_135528.json, N polygons=20)
        """
        polygons = cls._convert_qupath_json_to_polygons(path)
        return Polygons(path=path, data=polygons)

    @classmethod
    def from_xml(cls, path: str) -> Polygons:
        root = ET.parse(path)

        polygons = list()
        for annotation in root.findall(".//Annotation"):
            polygon = list()
            coordinates = annotation.find("Coordinates")
            if coordinates is not None:
                for coordinate in coordinates.findall("Coordinate"):
                    try:
                        x = int(coordinate.get("X"))
                    except:
                        x = int(float(coordinate.get("X")))
                    try:
                        y = int(coordinate.get("Y"))
                    except:
                        y = int(float(coordinate.get("Y")))

                    polygon.append((x, y))

            try:
                polygons.append(Polygon(polygon))
            except ValueError:
                # polygon이라고 있는데, 좌표는 1개인경우
                continue

        return Polygons(path=path, data=polygons)

    @classmethod
    def from_polygons(cls, polygons: List[Polygon]) -> Polygons:
        return Polygons(data=polygons)

    def __repr__(self) -> str:
        n_polygon: int = len(self.data) if self.data else 0
        return f"QuPathPloygon(path={self.path}, N polygons={n_polygon})"


class PatchWorker(multiprocessing.Process):
    def __init__(
        self,
        slide_path: str,
        input_queue,
        tile_size,
        overlap,
        limit_bounds,
        patch_filter: callable = None,
        logger=None,
    ):
        multiprocessing.Process.__init__(self)
        self.slide_path = slide_path
        self.input_queue = input_queue
        self.tile_size = tile_size
        self.patch_filter = patch_filter
        self.overlap = overlap
        self.daemon = True  # 부모프로세스 종료시 서브프로세스 종료
        self.limit_bounds = limit_bounds
        self.logger = logger

    def run(self):
        """병렬처리의 메인 루틴"""

        slide = open_slide(self.slide_path)  # slide할당시 병목

        while True:
            packed_data = self.input_queue.get()
            if packed_data is None:
                self.input_queue.task_done()
                break

            deepzoom_level, col, row, return_queue = packed_data
            address = (col, row)
            dz_generator = DeepZoomGenerator(
                slide,
                tile_size=self.tile_size,
                overlap=self.overlap,
                limit_bounds=self.limit_bounds,
            )

            try:
                patch: Image.Image = dz_generator.get_tile(deepzoom_level, address)
            except ValueError:
                self.logger.critical(
                    f"Invalid error: (deepzoom_level({deepzoom_level}), "
                    f"address({str(address)})"
                )

            location, level, size = dz_generator.get_tile_coordinates(
                deepzoom_level, address
            )

            to_level_0_rate = slide.level_downsamples[level]
            width_size_level_0 = int(size[0] * to_level_0_rate)
            height_size_level_0 = int(size[1] * to_level_0_rate)

            patch = patch.convert("RGB")
            x_min, y_min = location
            x_max, y_max = x_min + width_size_level_0, y_min + height_size_level_0
            coordinates = Coordinates(x_min, y_min, x_max, y_max)

            if self.patch_filter and self.patch_filter(np.asarray(patch)):
                self.input_queue.task_done()
                continue

            patch = Patch(
                image_array=np.array(patch).astype(np.uint8),
                coordinates=coordinates,
                address=address,
                slide_name=os.path.basename(self.slide_path).split(".")[0],
            )
            self.input_queue.task_done()
            return_queue.put(patch)

        return


class Tiler:
    def __init__(
        self,
        tile_size,
        overlap,
        limit_bounds,
        deepzoom_level,
        n_workers: int,
        patch_filter: Callable = None,
        logger=None,
    ) -> None:
        self.tile_size = tile_size
        self.overlap = overlap
        self.limit_bounds = limit_bounds
        self.deepzoom_level = deepzoom_level
        self.n_workers = n_workers
        self.patch_filter = patch_filter
        self.logger = logger

    def do_tile(self, slide_path) -> queue.Queue:
        input_queue = multiprocessing.JoinableQueue(self.n_workers * 2)
        return_queue = multiprocessing.Manager().Queue()

        non_overlapping_tile_size = self.tile_size - (2 * self.overlap)
        self.workers = list()
        for _ in range(self.n_workers):
            worker = PatchWorker(
                slide_path,
                input_queue,
                tile_size=non_overlapping_tile_size,
                overlap=self.overlap,
                limit_bounds=self.limit_bounds,
                patch_filter=self.patch_filter,
                logger=self.logger,
            )
            worker.start()
            self.workers.append(worker)

        # A list of (tiles_x, tiles_y)
        osr = OpenSlide(slide_path)
        dz_generator = DeepZoomGenerator(
            osr=osr,
            tile_size=non_overlapping_tile_size,
            overlap=self.overlap,
            limit_bounds=self.limit_bounds,
        )
        n_cols, n_rows = dz_generator.level_tiles[self.deepzoom_level]
        if self.logger:
            self.logger.info(
                f"Start tiling (Deepzoom lv: {self.deepzoom_level})"
                f", N({n_cols}, {n_rows})"
            )

        for row in range(1, n_rows - 1):
            for col in range(1, n_cols - 1):
                input_queue.put((self.deepzoom_level, col, row, return_queue))

        for _ in range(self.n_workers):
            input_queue.put(None)

        osr.close()

        return return_queue

    def _get_binary_thumnail_otsu(
        self,
        slide_path: str,
        otsu_level: Literal["multi", "single"],
        downsample: int = 16,
    ) -> np.ndarray:
        """오츄알고리즘을 이용한 전/배경 분리(Binary image)

        Args:
            slide_path (str): 슬라이드 경로
            otsu_level (str): otsu알고리즘의 multi, single level 방법론
                - "multi": multi-level otsu
                - "single": single-level otsu
            downsample (int, optional): Thumbnial시의 다운샘플링 비율율 Defaults to 16.

        Returns:
            np.ndarray: binary image
                - shape: (H at level 0 / downsample, W at level 0 / downsample)
        """

        with OpenSlide(slide_path) as osr:
            w, h = osr.dimensions
            thumnail = np.array(
                osr.get_thumbnail((int(w / downsample), int(h / downsample)))
            )
        gray_image = cv2.cvtColor(thumnail, cv2.COLOR_RGB2GRAY)

        if otsu_level == "single":
            ret, binary_image = cv2.threshold(
                gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif otsu_level == "multi":
            thresholds = threshold_multiotsu(gray_image)  # 2개나옴옴
            digitized_image = np.digitize(gray_image, bins=thresholds)
            binary_image = np.where(digitized_image == 2, 255, 0)

        return binary_image

    def do_tile_otsu(
        self,
        slide_path: str,
        otsu_level: Literal["multi", "single"] = "multi",
        downsample: int = 16,
        foreground_threshold: float = 0.05,
    ) -> queue.Queue:

        if self.logger:
            self.logger.info("Start binarization")

        binary_image = self._get_binary_thumnail_otsu(
            slide_path, otsu_level=otsu_level, downsample=downsample
        )
        non_overlapping_tile_size = self.tile_size - (2 * self.overlap)
        osr = OpenSlide(slide_path)
        dz_generator = DeepZoomGenerator(
            osr=osr,
            tile_size=non_overlapping_tile_size,
            overlap=self.overlap,
            limit_bounds=self.limit_bounds,
        )

        if self.logger:
            self.logger.info("Start candidate address from binary image")

        # (col, row)
        candidate_addresses = get_foreground_tile_address(
            dz_generator, self.deepzoom_level, binary_image, foreground_threshold
        )

        input_queue = multiprocessing.JoinableQueue()
        return_queue = multiprocessing.Manager().Queue()
        self.workers = list()
        for _ in range(self.n_workers):
            worker = PatchWorker(
                slide_path,
                input_queue,
                tile_size=non_overlapping_tile_size,
                overlap=self.overlap,
                patch_filter=self.patch_filter,
                limit_bounds=self.limit_bounds,
                logger=self.logger,
            )
            worker.start()
            self.workers.append(worker)

        for col, row in candidate_addresses:
            input_queue.put((self.deepzoom_level, col, row, return_queue))

        for _ in range(self.n_workers):
            input_queue.put(None)

        osr.close()

        return return_queue

    def join(self) -> None:
        for worker in self.workers:
            worker.join()

        return


@dataclass
class WSI:
    """WholeSlideImage의 데이터클레스"""

    slide_path: str = None
    patches: Patches = None
    label: str = None

    def __postinit__(self):
        """객체가 초기화된 후에 호출되는 메서드로, 슬라이드 파일이 존재하는지 확인"""
        if not os.path.exists(self.slide_path):
            raise FileNotFoundError(f"{self.slide_path} not found")

    def __repr__(self) -> str:
        n_patch = len(self.patches) if self.patches is not None else 0
        return f"WSI(slide_path={self.slide_path}, N patches={n_patch}, label={self.label})"

    def do_tile(
        self,
        tile_size,
        overlap,
        limit_bounds: bool,
        deepzoom_level: int,
        n_workers: int,
        patch_filter: callable = None,
        logger=None,
    ) -> Patches:

        osr = OpenSlide(self.slide_path)
        dzg = DeepZoomGenerator(osr)
        if len(dzg.level_tiles) <= deepzoom_level:
            raise IndexError(
                f"Deepzoom level ({len(dzg.level_tiles)}) < (Passed deep zoom level({deepzoom_level}))"
            )

        tiler = Tiler(
            tile_size,
            overlap,
            limit_bounds,
            deepzoom_level,
            n_workers,
            patch_filter,
            logger=logger,
        )

        queue = tiler.do_tile(self.slide_path)
        tiler.join()

        patches = list()
        while not queue.empty():
            patches.append(queue.get())

        return Patches(
            [
                patch
                for patch in patches
                if patch is not None and patch.image_array is not None
            ]
        )

    def do_tile_otsu(
        self,
        tile_size,
        overlap,
        limit_bounds: bool,
        deepzoom_level: int,
        n_workers: int,
        patch_filter: callable = None,
        otsu_level: Literal["multi", "signle"] = "multi",
        logger=None,
    ) -> Patches:

        osr = OpenSlide(self.slide_path)
        dzg = DeepZoomGenerator(osr)
        if len(dzg.level_tiles) <= deepzoom_level:
            raise IndexError(
                f"Deepzoom level ({len(dzg.level_tiles)}) < (Passed deep zoom level({deepzoom_level}))"
            )

        tiler = Tiler(
            tile_size,
            overlap,
            limit_bounds,
            deepzoom_level,
            n_workers,
            patch_filter,
            logger=logger,
        )

        queue = tiler.do_tile_otsu(
            self.slide_path, otsu_level=otsu_level, foreground_threshold=0.05
        )
        tiler.join()
        osr.close()

        patches = list()
        while not queue.empty():
            patches.append(queue.get())

        return Patches(
            [
                patch
                for patch in patches
                if patch is not None and patch.image_array is not None
            ]
        )


@dataclass
class DigestPath:
    """DigestPath2019의 데이터 클래스 (jpg, png 이미지 전용)

    Attributes:
        image_path (str): 입력 이미지 (jpg, png 등)의 파일 경로
        patches (Patches): 타일링된 패치들의 집합 (초기에는 None)
        label (str): 라벨 정보 (필요시 지정, 기본값은 "unknown")

    Example:
        >>> dp = DigestPath(image_path="path/to/image.jpg", label="normal")
        >>> patches = dp.do_tile(tile_size=256, overlap=0)
        >>> print(patches)
        Patches(N=XX)

        # Otsu thresholding을 통한 타일 후보 선택
        >>> patches_otsu = dp.do_tile_otsu(tile_size=256, overlap=0, foreground_threshold=0.1, otsu_level="single")
        >>> print(patches_otsu)
        Patches(N=YY)
    """

    image_path: str = None
    patches: Patches = None
    label: str = None

    def __postinit__(self):
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"{self.image_path} not found")

    def __repr__(self) -> str:
        n_patch = len(self.patches) if self.patches is not None else 0
        return f"DigestPath(image_path={self.image_path}, N patches={n_patch}, label={self.label})"

    def do_tile(
        self,
        tile_size: int,
        overlap: int = 0,
        patch_filter: Callable[[np.ndarray], bool] = None,
    ) -> Patches:
        """
        단일 jpg/png 이미지에 대해 지정한 크기와 오버랩으로 타일링(패치 추출) 수행

        Args:
            tile_size (int): 타일의 정방형 크기 (예: 256이면 256x256 크기의 패치)
            overlap (int, optional): 타일 간 겹침 크기. Defaults to 0.
            patch_filter (Callable, optional): 패치에 대해 추가적으로 필터링을 진행하는 함수.
                                               인자로 numpy array (패치 이미지)를 받고, True이면 해당 패치를 제외.

        Returns:
            Patches: 생성된 패치들의 집합
        """
        image = Image.open(self.image_path).convert("RGB")
        image_array = np.array(image)
        height, width = image_array.shape[:2]
        patches = []
        # 타일 추출 시 이동 간격 (overlap 적용)
        step = tile_size - overlap
        slide_name = os.path.basename(self.image_path).split(".")[0]

        # 이미지 전체를 tile_size 크기로 슬라이싱
        for y in range(0, height - tile_size + 1, step):
            for x in range(0, width - tile_size + 1, step):
                tile = image_array[y : y + tile_size, x : x + tile_size]
                # 패치 필터가 지정되어 있으면 필터링
                if patch_filter is not None and patch_filter(tile):
                    continue
                coord = Coordinates(
                    x_min=x, y_min=y, x_max=x + tile_size, y_max=y + tile_size
                )
                patch = Patch(
                    image_array=tile,
                    coordinates=coord,
                    address=(x, y),
                    slide_name=slide_name,
                    label=self.label,
                )
                patches.append(patch)

        return Patches(
            [
                patch
                for patch in patches
                if patch is not None and patch.image_array is not None
            ]
        )

    def do_tile_otsu(
        self,
        tile_size: int,
        overlap: int = 0,
        otsu_level: Literal["multi", "single"] = "multi",
        patch_filter: Callable[[np.ndarray], bool] = None,
    ) -> Patches:
        """
        Otsu thresholding을 적용하여 전/배경 분리 후, 배경 비율이
        0.05 이상인 영역만 타일링 수행

        Args:
            tile_size (int): 타일의 정방형 크기
            overlap (int, optional): 타일 간의 겹침. Defaults to 0.
            otsu_level (Literal["multi", "single"], optional): Otsu 알고리즘 적용 방식.
                                                               Defaults to "multi".
            patch_filter (Callable, optional): 패치 추가 필터 함수

        Returns:
            Patches: 후보로 선정된 패치들의 집합
        """
        image = Image.open(self.image_path).convert("RGB")
        image_array = np.array(image)
        height, width = image_array.shape[:2]
        # 그레이스케일 변환 후 Otsu thresholding 적용
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        if otsu_level == "single":
            _, binary_image = cv2.threshold(
                gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif otsu_level == "multi":
            thresholds = threshold_multiotsu(gray_image)
            digitized = np.digitize(gray_image, bins=thresholds)
            binary_image = np.where(digitized == 2, 255, 0).astype(np.uint8)
        else:
            raise ValueError("otsu_level must be either 'multi' or 'single'")

        patches = []
        step = tile_size - overlap
        slide_name = os.path.basename(self.image_path).split(".")[0]

        for y in range(0, height - tile_size + 1, step):
            for x in range(0, width - tile_size + 1, step):
                tile = image_array[y : y + tile_size, x : x + tile_size]
                binary_tile = binary_image[y : y + tile_size, x : x + tile_size]
                # 전경 픽셀 비율 계산 (여기서는 255인 픽셀을 전경으로 간주)
                foreground_ratio = np.count_nonzero(binary_tile == 255) / (
                    tile_size * tile_size
                )
                if foreground_ratio < 0.05:  # foreground_threshold = 0.05
                    continue
                if patch_filter is not None and patch_filter(tile):
                    continue
                coord = Coordinates(
                    x_min=x, y_min=y, x_max=x + tile_size, y_max=y + tile_size
                )
                patch = Patch(
                    image_array=tile,
                    coordinates=coord,
                    address=(x, y),
                    slide_name=slide_name,
                    label=self.label,
                )
                patches.append(patch)

        return Patches(
            [
                patch
                for patch in patches
                if patch is not None and patch.image_array is not None
            ]
        )
