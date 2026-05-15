from textgrid import TextGrid


def extract_word_intervals(textgrid_path):

    tg = TextGrid.fromFile(textgrid_path)

    word_intervals = []

    for tier in tg.tiers:
        if tier.name.lower() in ["words", "word"]:

            for interval in tier.intervals:
                word = interval.mark.strip()

                if word:
                    word_intervals.append({
                        "word": word,
                        "start": interval.minTime,
                        "end": interval.maxTime
                    })

    return word_intervals


def extract_word_audio_segments(audio, sample_rate, word_intervals):


    segments = []

    for item in word_intervals:
        start_sample = int(item["start"] * sample_rate)
        end_sample = int(item["end"] * sample_rate)

        word_audio = audio[start_sample:end_sample]

        # skip empty/broken segments
        if len(word_audio) == 0:
            continue

        segments.append({
            "word": item["word"],
            "audio": word_audio,
            "start": item["start"],
            "end": item["end"]
        })

    return segments