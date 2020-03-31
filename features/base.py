from typing import Any

import numpy as np


class AudioSample:

    def __init__(self, data: np.ndarray, rate: float):
        self.data = data
        self.rate = rate


class SampleProcessor:

    def process(self, sample: AudioSample) -> Any:
        pass