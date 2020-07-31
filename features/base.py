from typing import Any

import numpy as np
import scipy.signal as sps


class AudioSample:

    def __init__(self, data: np.ndarray, rate: int, name: str):
        self.data = data
        self.rate = rate
        self.name = name


class SampleProcessor:
    INPUT_SAMPLING_RATE = 44100

    def process(self, sample: AudioSample) -> Any:
        pass

    def __call__(self, sample: AudioSample):
        if sample.rate != self.INPUT_SAMPLING_RATE:
            number_of_samples = round(len(sample.data) * float(self.INPUT_SAMPLING_RATE) / sample.rate)
            data = sps.resample(sample.data, number_of_samples)
            return self.process(AudioSample(data=data, rate=self.INPUT_SAMPLING_RATE, name=sample.name))
        else:
            return self.process(sample)
