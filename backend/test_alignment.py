from alignment import extract_word_intervals, extract_word_audio_segments
from preprocess import preprocess_audio


audio, sample_rate = preprocess_audio("test.m4a")

intervals = extract_word_intervals(
    "../alignment_test/aligned_output/sample_fixed.TextGrid"
)

segments = extract_word_audio_segments(
    audio,
    sample_rate,
    intervals
)

for segment in segments:
    print(segment["word"], len(segment["audio"]))