import os
import random
import math
import torchaudio
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as transforms
import torchaudio.transforms as T
from config import SAMPLING_RATE



class AugmentAudio:
    def __init__(self):
        self.transforms = [
            T.Vol(gain=0.9),
            T.TimeStretch(fixed_rate=0.8),
            T.FrequencyMasking(freq_mask_param=15),
            T.TimeMasking(time_mask_param=35)
        ]

    def __call__(self, audio):
        if random.random() > 0.5:
            for transform in self.transforms:
                audio = transform(audio)
        return audio



class AudioDataset(Dataset):
    """
    A PyTorch Dataset class for loading audio data from a folder structure where each subdirectory is a class.

    Attributes:
        root_dir (str): The root directory of the dataset.
        classes (list): List of class names (subdirectory names).
        file_list (list): List of (file_path, class_idx) tuples.
        transform (callable, optional): Optional transform to be applied on a sample.
        duration (float): The duration (in seconds) of audio samples to extract.
        sample_rate (int): The sample rate of the audio files.
    """

    def __init__(self, root_dir, duration=5, transform=None):
        """
        Initialize the dataset with the root directory.

        Parameters:
            root_dir (str): Path to the root directory containing class subdirectories.
            duration (float, optional): Duration (in seconds) of audio samples to extract. Default is 5 seconds.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.duration = duration
        self.transform = transform
        
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.classes.sort()  # Ensure consistent class ordering
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.file_list = []
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_path):
                if filename.endswith(('.wav', '.mp3', '.flac')):  # Add more audio formats if needed
                    self.file_list.append((os.path.join(class_path, filename), self.class_to_idx[class_name]))
        
        # Determine the sample rate from the first audio file
        first_audio_path = self.file_list[0][0]
        waveform, sample_rate = torchaudio.load(first_audio_path)
        self.sample_rate = sample_rate

    def get_class_name(self, class_idx):
        """
        Returns the class name corresponding to the given class index.

        Parameters:
            class_idx (int): The class index.

        Returns:
            str: The corresponding class name.
        """
        for name, idx in self.class_to_idx.items():
            if idx == class_idx:
                return name
        return None  # Return None if the class index is not found

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Generates one sample of data.

        Parameters:
            idx (int): Index of the sample to be fetched.

        Returns:
            tuple: A tuple containing the audio sample (waveform and sample rate) and the label (class ID).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the file path and label
        audio_path, label_idx = self.file_list[idx]

        # Load the audio file
        waveform, _ = torchaudio.load(audio_path)

        # Apply transform if provided
        if self.transform:
            waveform = self.transform(waveform)

        # Calculate the number of samples to extract
        samples_to_extract = int(self.duration * self.sample_rate)

        # Pad or truncate the waveform to the desired duration
        if waveform.size(1) < samples_to_extract:
            pad_length = samples_to_extract - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_length), 'constant', 0)
        elif waveform.size(1) > samples_to_extract:
            waveform = waveform[:, :samples_to_extract]

        sample = {'data': waveform, 'sample_rate': self.sample_rate}

        return sample, label_idx

class MAudioDataset(Dataset):
    """
    A PyTorch Dataset class for loading audio data from a folder structure where each subdirectory is a class.
    This version splits longer audio samples into segments of the desired duration, pads shorter segments,
    and resamples all audio to a specified sample rate.

    Attributes:
        root_dir (str): The root directory of the dataset.
        classes (list): List of class names (subdirectory names).
        file_list (list): List of (file_path, class_idx, start_time) tuples.
        transform (callable, optional): Optional transform to be applied on a sample.
        duration (float): The duration (in seconds) of audio samples to extract.
        target_sample_rate (int): The target sample rate for all audio files.
    """

    def __init__(self, root_dir, duration=5, target_sample_rate=SAMPLING_RATE, transform=None):
        """
        Initialize the dataset with the root directory.

        Parameters:
            root_dir (str): Path to the root directory containing class subdirectories.
            duration (float, optional): Duration (in seconds) of audio samples to extract. Default is 5 seconds.
            target_sample_rate (int, optional): Target sample rate for all audio. Default is 44100 Hz.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.duration = duration
        self.target_sample_rate = target_sample_rate
        self.transform = transform
        
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.classes.sort()  # Ensure consistent class ordering
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.file_list = []
        self.file_sample_rates = []
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_path):
                if filename.endswith(('.wav', '.mp3', '.flac')):  # Add more audio formats if needed
                    file_path = os.path.join(class_path, filename)
                    self._add_file_segments(file_path, self.class_to_idx[class_name])
        
        if not self.file_list:
            raise ValueError("No audio files found in the specified directory.")

    def _add_file_segments(self, file_path, class_idx):
        """
        Add file segments to the file_list based on audio duration.

        Parameters:
            file_path (str): Path to the audio file.
            class_idx (int): Class index for the audio file.
        """
        audio_metadata = torchaudio.info(file_path)
        audio_length = audio_metadata.num_frames / audio_metadata.sample_rate
        num_segments = math.ceil(audio_length / self.duration)
        
        for i in range(num_segments):
            start_time = i * self.duration
            self.file_list.append((file_path, class_idx, start_time))
            self.file_sample_rates.append(audio_metadata.sample_rate)

    def get_class_name(self, class_idx):
        """
        Returns the class name corresponding to the given class index.

        Parameters:
            class_idx (int): The class index.

        Returns:
            str: The corresponding class name.
        """
        for name, idx in self.class_to_idx.items():
            if idx == class_idx:
                return name
        return None  # Return None if the class index is not found

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Generates one sample of data.

        Parameters:
            idx (int): Index of the sample to be fetched.

        Returns:
            tuple: A tuple containing the audio sample (waveform and sample rate) and the label (class ID).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the file path, label, and start time
        audio_path, label_idx, start_time = self.file_list[idx]
        sr = self.file_sample_rates[idx]

        # Load the audio segment
        # waveform, sample_rate = torchaudio.load(audio_path, 
        #                                         frame_offset=int(start_time * sample_rate),
        #                                         num_frames=int(self.duration * sample_rate))
        
        if sr == self.target_sample_rate:
            waveform, sample_rate = torchaudio.load(audio_path, 
                                                frame_offset=int(start_time * self.target_sample_rate),
                                                num_frames=int(self.duration * self.target_sample_rate))
        else:
            waveform, sample_rate = torchaudio.load(audio_path, 
                                                frame_offset=int(start_time * sr),
                                                num_frames=int(self.duration * sr))

        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)

        # Calculate the number of samples to extract after resampling
        samples_to_extract = int(self.duration * self.target_sample_rate)

        # Pad or truncate the waveform to the desired duration
        if waveform.size(1) < samples_to_extract:
            pad_length = samples_to_extract - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_length), 'constant', 0)
        elif waveform.size(1) > samples_to_extract:
            waveform = waveform[:, :samples_to_extract]

        # Apply transform if provided
        if self.transform:
            waveform = self.transform(waveform)

        sample = {'data': waveform, 'sample_rate': self.target_sample_rate}

        return sample, label_idx


class SpectrogramDataset(MAudioDataset):
    """
    A PyTorch Dataset class for loading audio data and converting it to log mel spectrograms.
    This class inherits from MAudioDataset and adds spectrogram transformation.

    Attributes:
        n_mels (int): Number of mel filterbanks.
        n_fft (int): Size of FFT.
        hop_length (int): Number of samples between successive frames.
        power (float): Exponent for the magnitude spectrogram.
        normalize (bool): Whether to normalize the spectrograms.
    """

    def __init__(self, root_dir, duration=5, target_sample_rate=SAMPLING_RATE, n_fft=512, 
                 hop_length=256, power=2.0, normalize=True, transform=None):
        """
        Initialize the dataset with the root directory and spectrogram parameters.

        Parameters:
            root_dir (str): Path to the root directory containing class subdirectories.
            duration (float, optional): Duration (in seconds) of audio samples to extract. Default is 5 seconds.
            target_sample_rate (int, optional): Target sample rate for all audio. Default is 44100 Hz.
            n_fft (int, optional): Size of FFT. Default is 512.
            hop_length (int, optional): Number of samples between successive frames. Default is 256.
            power (float, optional): Exponent for the magnitude spectrogram. Default is 2.0.
            normalize (bool, optional): Whether to normalize the spectrograms. Default is True.
            transform (callable, optional): Optional transform to be applied on the spectrogram.
        """
        super().__init__(root_dir, duration, target_sample_rate, transform)
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.normalize = normalize
        
        self.spectrogram = transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=self.power)
        # log_spectrogram = transforms.AmplitudeToDB()(spectrogram)

        # self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        #     sample_rate=target_sample_rate,
        #     n_fft=n_fft,
        #     hop_length=hop_length,
        #     power=power
        # )

    def __getitem__(self, idx):
        """
        Generates one sample of data.

        Parameters:
            idx (int): Index of the sample to be fetched.

        Returns:
            tuple: A tuple containing the log mel spectrogram and the label (class ID).
        """
        sample, label_idx = super().__getitem__(idx)
        waveform = sample['data']

        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Compute mel spectrogram
        spectrogram = self.spectrogram(waveform)

        # Convert to decibels
        log_spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)

        # Normalize if required
        if self.normalize:
            log_spectrogram = (log_spectrogram - log_spectrogram.mean()) / log_spectrogram.std()

        # Apply additional transform if provided
        if self.transform:
            log_spectrogram = self.transform(log_spectrogram)
        
        sample = {'data': log_spectrogram,'waveform': waveform, 'sample_rate': self.target_sample_rate}

        return sample, label_idx


    
if __name__ == "__main__":
    pass
