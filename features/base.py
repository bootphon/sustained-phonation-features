from typing import Any

import numpy as np


class AudioSample:

    def __init__(self, data: np.ndarray, rate: int, name: str):
        self.data = data
        self.rate = rate
        self.name = name


class SampleProcessor:

    def process(self, sample: AudioSample) -> Any:
        pass