# Speech Processing: Sorting "Seven" by Fundamental Frequency (F0)

This project processes an audio file containing multiple speakers saying the word “seven” and produces a new audio file where speakers are arranged from lowest to highest mean pitch (F0).

## Pipeline

1. **Loading audio**  
   - `load_audio_data()` uses `soundfile` to read the `.wav` file into a mono NumPy array.

2. **Segmenting non-silent clips**  
   - `librosa.effects.split()` detects non-silent segments, each containing one speaker saying “seven”.

3. **Pitch estimation (F0)**  
   - `get_mean_f0()` applies `librosa.pyin()` to estimate the pitch contour for each clip.  
   - Uses `numpy.nanmean` to compute mean F0 per clip and discards segments where F0 cannot be estimated.

4. **Sorting and recombining**  
   - `process_clips_and_sort()` pairs each clip with its mean F0 and sorts clips from lowest to highest pitch.  
   - Concatenates sorted clips and writes them to a new `.wav` file with `soundfile.write()`.

5. **Command-line interface and error handling**  
   - `main()` provides a CLI interface and checks for:  
     - Missing arguments  
     - Invalid file paths  
     - Runtime errors, with clear error messages instead of raw tracebacks.

## Files

- `process.py` – main script with the full pipeline and CLI.  
- Input: e.g., `seven.wav` with multiple speakers.  
- Output: e.g., `output_sorted.wav` with speakers ordered by mean F0.

## Requirements

- Python 3.x  
- `librosa`  
- `soundfile`  
- `numpy`

Install dependencies:

```bash
pip install librosa soundfile numpy
```
## Usage
```python
python3 process.py path/to/seven.wav path/to/output_sorted.wav
```
### Example (course environment):
```python
python3 process.py /home/dsv/robe/lis020/labs/data/seven.wav ~/public_html/output.wav
```
### Note
This project was run in a virtual environment on Linux, removing a problematic clip with all-NaN F0 values and successfully generating a sorted output file. It was developed as part of a speech processing lab and graded with distinction.
