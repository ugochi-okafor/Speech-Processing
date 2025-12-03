"""
This code loads a multi-speaker speech data file, segments it,
calculates the mean fundamental frequency (F0) for each segment,
sorts the segments based on F0, and writes the concatenated, sorted
audio back to a new WAV file.

It uses numpy, librosa, and soundfile external libraries.
"""

import sys
import os
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, List, Optional

# --- Constants for F0 Calculation ---
F0_MIN = 70.0  # Minimum frequency for F0 tracking (Hz)
F0_MAX = 500.0 # Maximum frequency for F0 tracking (Hz)

def load_audio_data(input_path: str) -> Tuple[np.ndarray, int]:
    """
    Loads audio data from a specified file using soundfile.

    Args:
        input_path: The path to the input audio file.

    Returns:
        A tuple containing:
        - The audio data as a numpy.ndarray (monophonic, float).
        - The sampling rate (int).

    Raises:
        FileNotFoundError: If the input file does not exist.
        RuntimeError: If the file cannot be read by soundfile.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        # soundfile.read returns data and samplerate. 'always_2d=False' ensures mono is 1D.
        data, sr = sf.read(input_path, dtype='float32', always_2d=False)
        # Ensure data is a proper numpy.ndarray even if 1D for type consistency
        if not isinstance(data, np.ndarray):
            data = np.array(data)
    except Exception as e:
        # Catch soundfile/IO errors
        raise RuntimeError(f"Could not read audio file {input_path}: {e}")

    # Ensure audio is mono (if multi-channel, it needs to be mixed down)
    # The assignment implies a mono file, but this adds robustness.
    if data.ndim > 1:
        # Assuming stereo, take the mean across channels
        data = np.mean(data, axis=1)

    return data, sr

def segment_audio(data: np.ndarray, sr: int) -> List[np.ndarray]:
    """
    Splits the audio data into segments (clips) based on silent pauses.

    Args:
        data: The audio data (1D numpy.ndarray).
        sr: The sampling rate (int).

    Returns:
        A list of numpy.ndarray, where each array is an audio clip.
    """
    # Use librosa.effects.split to detect non-silent intervals
    # Frame length (frame_length) and hop length (hop_length) are default
    # The 'top_db' default (60) is usually good for speech/pause detection
    intervals = librosa.effects.split(data, top_db=60)

    # Extract clips using the start and end indices
    clips: List[np.ndarray] = [data[start:end] for start, end in intervals]
    return clips

def get_mean_f0(clip: np.ndarray, sr: int) -> Optional[float]:
    """
    Computes the mean fundamental frequency (F0) of an audio clip.

    Uses librosa.pyin for F0 tracking and numpy.nanmean to calculate the
    mean, handling NaN values. Discards clips if F0 could not be computed.

    Args:
        clip: The audio clip (1D numpy.ndarray).
        sr: The sampling rate (int).

    Returns:
        The mean F0 in Hz (float) or None if F0 could not be reliably computed.
    """
    # 1. Compute the F0 curve using librosa.pyin
    # pyin returns (F0_curve, voice_flag)
    f0_curve, voiced_flag, voiced_probs = librosa.pyin(
        clip,
        fmin=F0_MIN,
        fmax=F0_MAX,
        sr=sr
    )

    # 2. Check if the F0 curve contains only NaN values
    if np.all(np.isnan(f0_curve)):
        return None # Discard this clip

    # 3. Compute the mean F0, ignoring NaN values
    mean_f0: float = np.nanmean(f0_curve)
    return mean_f0

def process_clips_and_sort(clips: List[np.ndarray], sr: int) -> Tuple[List[np.ndarray], List[float]]:
    """
    Calculates F0 for each clip, filters out invalid clips, and sorts them.

    Args:
        clips: A list of audio clips (numpy.ndarray).
        sr: The sampling rate (int).

    Returns:
        A tuple containing:
        - A list of the sorted clips (numpy.ndarray).
        - A list of the corresponding mean F0 values (float).
    """
    # List to hold tuples of (mean_f0, clip_data)
    f0_clip_pairs: List[Tuple[float, np.ndarray]] = []

    print(f"Processing {len(clips)} potential clips...")

    for i, clip in enumerate(clips):
        # Calculate mean F0 for the clip
        mean_f0 = get_mean_f0(clip, sr)

        # Discard clips where F0 could not be computed (mean_f0 is None)
        if mean_f0 is not None:
            f0_clip_pairs.append((mean_f0, clip))
        else:
            print(f"Warning: Discarded clip {i+1} due to all NaN F0.")

    print(f"Successfully processed and kept {len(f0_clip_pairs)} clips.")

    # Sort the list of (mean_f0, clip_data) tuples by mean_f0 (lowest to highest)
    # The default key and reverse=False achieves lowest-to-highest sort by the first element (F0)
    sorted_pairs = sorted(f0_clip_pairs, key=lambda x: x[0])

    # Separate the sorted clips and F0 values
    sorted_f0s: List[float] = [f0 for f0, _ in sorted_pairs]
    sorted_clips: List[np.ndarray] = [clip for _, clip in sorted_pairs]

    return sorted_clips, sorted_f0s

def write_audio_data(output_path: str, data: np.ndarray, sr: int) -> None:
    """
    Writes the concatenated audio data to a WAV file.

    Args:
        output_path: The path to the output audio file.
        data: The concatenated audio data (1D numpy.ndarray).
        sr: The sampling rate (int).

    Raises:
        RuntimeError: If the file cannot be written by soundfile.
    """
    try:
        # Write data to WAV file
        sf.write(output_path, data, sr)
    except Exception as e:
        raise RuntimeError(f"Could not write audio file {output_path}: {e}")

def main():
    """
    Main function to orchestrate the program's execution.
    Handles command-line arguments and calls all processing steps.
    """
    # 1. Error Handling: Check for correct number of command-line arguments
    if len(sys.argv) != 3:
        print("Error: Incorrect number of arguments.")
        print(f"Usage: python3 {sys.argv[0]} <input_wav_file> <output_wav_file>")
        # Use a non-zero exit code to signal error
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        print(f"Starting process for input: {input_file}")

        # --- Step 1: Read data file ---
        data, sr = load_audio_data(input_file)
        print(f"Input file loaded. Sample Rate: {sr} Hz. Duration: {len(data)/sr:.2f} s.")

        # --- Step 2: Cut the recording into clips ---
        clips = segment_audio(data, sr)
        print(f"Audio segmented into {len(clips)} clips.")

        # --- Step 3, 4, 5: Compute F0, Filter, and Sort ---
        sorted_clips, sorted_f0s = process_clips_and_sort(clips, sr)

        # Sanity Check Output (as requested in the assignment)
        # Round F0s to nearest integer
        rounded_f0s = [round(f) for f in sorted_f0s]
        print("\n--- Sanity Check: Estimated Mean F0s (sorted, rounded) ---")
        print(rounded_f0s)
        print("----------------------------------------------------------\n")


        # --- Step 6: Concatenate the clips ---
        if not sorted_clips:
            print("Error: No valid clips remained after F0 processing. Cannot concatenate.")
            sys.exit(1)

        # Concatenate all sorted clips into a single array
        concatenated_data = np.concatenate(sorted_clips)
        print(f"Clips concatenated. New duration: {len(concatenated_data)/sr:.2f} s.")

        # --- Step 7: Write the resulting data to a wav file ---
        write_audio_data(output_file, concatenated_data, sr)
        print(f"Success! Output written to: {output_file}")

    except (FileNotFoundError, RuntimeError) as e:
        # Catch custom, informative errors
        print(f"\nFatal Error: {e}")
        sys.exit(1)
    except Exception as e:
        # Catch any unexpected, critical errors
        print(f"\nAn unexpected error occurred: {e}")
        # Optionally, re-raise the error for debugging: raise
        sys.exit(1)

if __name__ == "__main__":
    main()
