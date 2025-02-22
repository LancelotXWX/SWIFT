# Reimport necessary libraries after state reset
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class NonStationaryTimeSeriesDataset(Dataset):
    def __init__(self, seq_length_input, seq_length_output, frequencies, amp_range, total_length, is_test=False):
        """
        Args:
            seq_length_input: Length of input sequences.
            seq_length_output: Length of output sequences.
            frequencies: List of frequency components for signal generation.
            amp_range: Amplitude range for signal generation.
            total_length: Total length of the time series.
            is_test: Whether the dataset is for testing (to introduce distinct temporal patterns).
        """
        self.seq_length_input = seq_length_input
        self.seq_length_output = seq_length_output
        self.frequencies = frequencies
        self.amp_range = amp_range
        self.total_length = total_length
        self.is_test = is_test

        # Generate one long continuous sequence
        self.sequence = self._generate_sequence()

    def _generate_sequence(self):
        # Time axis
        t = np.linspace(0, 2, self.total_length)  # Normalize time axis to [0, 2]

        # Generate signal with multiple frequency components
        amplitude = np.random.uniform(*self.amp_range)
        phase = np.random.uniform(0, 2 * np.pi, len(self.frequencies))  # Different phase for each frequency
        base_signal = sum(
            amplitude * np.sin(2 * np.pi * freq * t + phi)
            for freq, phi in zip(self.frequencies, phase)
        )

        # Generate non-stationary signal: time-dependent frequency modulation
        if self.is_test:
            non_stationary_signal = sum(
                amplitude * np.sin(2 * np.pi * freq * t**2 + phi)
                for freq, phi in zip(self.frequencies, phase)
            )
        else:
            non_stationary_signal = sum(
                amplitude * np.sin(2 * np.pi * freq * (t + 0.2 * np.sin(0.5 * t)) + phi)
                for freq, phi in zip(self.frequencies, phase)
            )

        # Combine signals with trends and random noise
        trend = np.sin(0.5 * t) + 0.2 * t  # Non-linear trend
        random_oscillations = 0.2 * np.random.normal(size=self.total_length)  # Small random noise

        # Final sequence: base + non-stationary + trend + noise
        sequence = base_signal + non_stationary_signal + trend + random_oscillations

        # Normalize the sequence
        sequence = (sequence - np.min(sequence)) / (np.max(sequence) - np.min(sequence))
        return sequence

    def __len__(self):
        return self.total_length - self.seq_length_input - self.seq_length_output + 1

    def __getitem__(self, idx):
        input_seq = self.sequence[idx:idx + self.seq_length_input]
        output_seq = self.sequence[idx + self.seq_length_input:idx + self.seq_length_input + self.seq_length_output]
        return (
            torch.tensor(input_seq, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(output_seq, dtype=torch.float32).unsqueeze(-1),
        )


if __name__ == '__main__':
    # Parameters
    num_samples = 1000
    seq_length_input = 48
    seq_length_output = 8
    freq_range = [8, 24]
    amp_range = [0.5, 1.5]

    # Create dataset and dataloader
    dataset = NonStationaryTimeSeriesDataset(seq_length_input, seq_length_output, freq_range, amp_range, num_samples)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Visualize a few samples from the dataset
    import random

    # Randomly select a few samples from the dataset
    samples = random.sample(range(len(dataset)), 2)

    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(samples, start=1):
        input_seq, output_seq = dataset[idx]
        t_input = np.arange(len(input_seq))
        t_output = np.arange(len(input_seq), len(input_seq) + len(output_seq))

        # Plot in time domain
        plt.subplot(2, 2, 2 * i - 1)
        plt.plot(t_input, input_seq.squeeze().numpy(), label="Input Sequence")
        plt.plot(t_output, output_seq.squeeze().numpy(), label="Output Sequence")
        plt.title(f"Sample {i} - Time Domain")
        plt.xlabel("Time")
        plt.ylabel("Normalized Value")
        plt.legend()
        plt.grid()

        # Plot in frequency domain
        plt.subplot(2, 2, 2 * i)
        full_seq = torch.cat([input_seq, output_seq]).squeeze().numpy()
        freq_spectrum = np.fft.fft(full_seq)
        freq = np.fft.fftfreq(len(full_seq))
        plt.plot(np.abs(freq[:len(full_seq) // 2]), np.abs(freq_spectrum[:len(full_seq) // 2]))
        plt.title(f"Sample {i} - Frequency Domain")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.grid()

    plt.tight_layout()
    plt.show()
