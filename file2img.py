import io
import typing as T

import numpy as np
from PIL import Image
import pydub
from scipy.io import wavfile
import torch
import torchaudio
import argparse


def spectrogram_image_from_file(filename, max_volume: float = 50, power_for_image: float = 0.25) -> Image.Image:
    """
    Generate a spectrogram image from an MP3 file.
    """
    # Load MP3 file into AudioSegment object
    audio = pydub.AudioSegment.from_file(filename)

    # Convert to mono and set frame rate
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(44100)

    length_in_ms = len(audio)

    # Extract first 5 seconds of audio data
    
    
    audio = audio[:60000]
    length_in_ms = len(audio)
    print("AUDIO LENGTH IN MS:", length_in_ms)

    # Convert to WAV and save as BytesIO object
    wav_bytes = io.BytesIO()
    audio.export("clip.wav", format="wav")
    audio.export(wav_bytes, format="wav")
    wav_bytes.seek(0)

    # Generate spectrogram image from WAV file
    return spectrogram_image_from_wav(wav_bytes, max_volume=max_volume, power_for_image=power_for_image, ms_duration=length_in_ms)

def spectrogram_image_from_wav(wav_bytes: io.BytesIO, max_volume: float = 50, power_for_image: float = 0.25, ms_duration: int = 5000) -> Image.Image:
    """
    Generate a spectrogram image from a WAV file.
    """
    # Read WAV file from bytes
    sample_rate, waveform = wavfile.read(wav_bytes)

    #sample_rate = 44100  # [Hz]
    clip_duration_ms = ms_duration  # [ms]

    bins_per_image = 512
    n_mels = 1024
    mel_scale = True

    # FFT parameters
    window_duration_ms = 100  # [ms]
    padded_duration_ms = 400  # [ms]
    step_size_ms = 10  # [ms]

    # Derived parameters
    num_samples = int(512 / float(bins_per_image) * clip_duration_ms) * sample_rate
    n_fft = int(padded_duration_ms / 1000.0 * sample_rate)
    hop_length = int(step_size_ms / 1000.0 * sample_rate)
    win_length = int(window_duration_ms / 1000.0 * sample_rate)

    # Compute spectrogram from waveform
    Sxx = spectrogram_from_waveform(
        waveform=waveform,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        mel_scale=mel_scale,
        n_mels=n_mels,
    )

    # Convert spectrogram to image
    image = image_from_spectrogram(Sxx, max_volume=max_volume, power_for_image=power_for_image)

    return image

def spectrogram_from_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    mel_scale: bool = True,
    n_mels: int = 512,
) -> np.ndarray:
    """
    Compute a spectrogram from a waveform.
    """

    spectrogram_func = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        power=None,
        hop_length=hop_length,
        win_length=win_length,
    )

    waveform_tensor = torch.from_numpy(waveform.astype(np.float32)).reshape(1, -1)
    Sxx_complex = spectrogram_func(waveform_tensor).numpy()[0]

    Sxx_mag = np.abs(Sxx_complex)

    if mel_scale:
        mel_scaler = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=0,
            f_max=10000,
            n_stft=n_fft // 2 + 1,
            norm=None,
            mel_scale="htk",
        )

        Sxx_mag = mel_scaler(torch.from_numpy(Sxx_mag)).numpy()

    return Sxx_mag

def image_from_spectrogram(
        data: np.ndarray,
        max_volume: float = 50,
        power_for_image: float = 0.25
) -> Image.Image:
    data = np.power(data, power_for_image)
    data = data / (max_volume / 255)
    data = 255 - data
    data = data[::-1]
    image = Image.fromarray(data.astype(np.uint8))
    return image

parser = argparse.ArgumentParser()
parser.add_argument("-f", help="the file to process")
parser.add_argument("-o", help="the file to process")
args = parser.parse_args()

# The filename is stored in the `filename` attribute of the `args` object
filename = args.f
image = spectrogram_image_from_file(filename, max_volume=100)

image.save(args.o)