"""
Defines autosub's main functionality.
"""

#!/usr/bin/env python


import numpy as np
from scipy.fft import fft
import argparse
import audioop
import math
import multiprocessing
import os
import subprocess
import sys
import tempfile
import wave
import json
import requests
from pathlib import Path
from tqdm import tqdm

try:
    from json.decoder import JSONDecodeError
except ImportError:
    JSONDecodeError = ValueError

from googleapiclient.discovery import build

from autosub.constants import (
    LANGUAGE_CODES,
    GOOGLE_SPEECH_API_KEY,
    GOOGLE_SPEECH_API_URL,
)
from autosub.formatters import FORMATTERS

DEFAULT_SUBTITLE_FORMAT = "srt"
DEFAULT_CONCURRENCY = 10
DEFAULT_SRC_LANGUAGE = "en"
DEFAULT_DST_LANGUAGE = "en"


def percentile(arr, percent):
    """
    Calculate the given percentile of arr.
    """
    arr = sorted(arr)
    index = (len(arr) - 1) * percent
    floor = math.floor(index)
    ceil = math.ceil(index)
    if floor == ceil:
        return arr[int(index)]
    low_value = arr[int(floor)] * (ceil - index)
    high_value = arr[int(ceil)] * (index - floor)
    return low_value + high_value


class FLACConverter(object):  # pylint: disable=too-few-public-methods
    """
    Class for converting a region of an input audio or video file into a FLAC audio file
    """

    def __init__(self, source_path, include_before=0.25, include_after=0.25, verbose=False):
        self.source_path = Path(source_path)
        self.flac_path = self.source_path.parent.joinpath(self.source_path.stem + "_chunks")
        self.flac_path.mkdir(exist_ok=True)
        self.include_before = include_before
        self.include_after = include_after
        self.verbose = verbose

    def __call__(self, region):
        try:
            start, end = region
            temp_path = self.flac_path.joinpath(f"tmp_{int(start * 1000):08}_{int(end * 1000):08}.flac")
            start = max(0, start - self.include_before)
            end += self.include_after
            command = [
                "ffmpeg",
                "-ss",
                str(start),
                "-t",
                str(end - start),
                "-y",
                "-i",
                self.source_path,
                "-loglevel",
                "error",
                str(temp_path),
            ]
            use_shell = True if os.name == "nt" else False
            subprocess.check_output(command, stdin=open(os.devnull), shell=use_shell)
            with open(temp_path, "rb") as temp:
                read_data = temp.read()
            if not self.verbose:
                os.unlink(temp.name)
            return read_data

        except KeyboardInterrupt:
            return None


class SpeechRecognizer(object):  # pylint: disable=too-few-public-methods
    """
    Class for performing speech-to-text for an input FLAC file.
    """

    def __init__(self, language="en", rate=44100, retries=3, api_key=GOOGLE_SPEECH_API_KEY):
        self.language = language
        self.rate = rate
        self.api_key = api_key
        self.retries = retries

    def __call__(self, data):
        try:
            for _ in range(self.retries):
                url = GOOGLE_SPEECH_API_URL.format(lang=self.language, key=self.api_key)
                headers = {"Content-Type": "audio/x-flac; rate=%d" % self.rate}

                try:
                    resp = requests.post(url, data=data, headers=headers)
                except requests.exceptions.ConnectionError:
                    continue

                for line in resp.content.decode("utf-8").split("\n"):
                    try:
                        line = json.loads(line)
                        line = line["result"][0]["alternative"][0]["transcript"]
                        return line[:1].upper() + line[1:]
                    except IndexError:
                        # no result
                        continue
                    except JSONDecodeError:
                        continue

        except KeyboardInterrupt:
            return None


class Translator(object):  # pylint: disable=too-few-public-methods
    """
    Class for translating a sentence from a one language to another.
    """

    def __init__(self, language, api_key, src, dst):
        self.language = language
        self.api_key = api_key
        self.service = build("translate", "v2", developerKey=self.api_key)
        self.src = src
        self.dst = dst

    def __call__(self, sentence):
        try:
            if not sentence:
                return None

            result = (
                self.service.translations()
                .list(source=self.src, target=self.dst, q=[sentence])  # pylint: disable=no-member
                .execute()
            )

            if "translations" in result and result["translations"] and "translatedText" in result["translations"][0]:
                return result["translations"][0]["translatedText"]

            return None

        except KeyboardInterrupt:
            return None


def which(program):
    """
    Return the path for a given executable.
    """

    def is_exe(file_path):
        """
        Checks whether a file is executable.
        """
        return os.path.isfile(file_path) and os.access(file_path, os.X_OK)

    fpath, _ = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


def extract_audio(filename, channels=1, rate=44100):
    """
    Extract audio from an input file to a temporary WAV file.
    """
    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    os.unlink(temp.name)
    if not os.path.isfile(filename):
        print("The given file does not exist: {}".format(filename))
        raise Exception("Invalid filepath: {}".format(filename))
    if not which("ffmpeg") and not which("ffmpeg.exe"):
        print("ffmpeg: Executable not found on machine.")
        raise Exception("Dependency not found: ffmpeg")
    command = ["ffmpeg", "-y", "-i", filename, "-ac", str(channels), "-ar", str(rate), "-loglevel", "error", temp.name]
    use_shell = True if os.name == "nt" else False
    subprocess.check_output(command, stdin=open(os.devnull), shell=use_shell)
    return temp.name, rate


# def find_speech_regions(filename, frame_width=4096, min_region_size=0.5, max_region_size=6): # pylint: disable=too-many-locals
#     """
#     Perform voice activity detection on a given audio file.
#     """
#     reader = wave.open(filename)
#     sample_width = reader.getsampwidth()
#     rate = reader.getframerate()
#     n_channels = reader.getnchannels()
#     chunk_duration = float(frame_width) / rate
#
#     n_chunks = int(math.ceil(reader.getnframes()*1.0 / frame_width))
#     energies = []
#
#     for _ in range(n_chunks):
#         chunk = reader.readframes(frame_width)
#         energies.append(audioop.rms(chunk, sample_width * n_channels))
#
#     threshold = percentile(energies, 0.2)
#
#     elapsed_time = 0
#
#     regions = []
#     region_start = None
#
#     for energy in energies:
#         is_silence = energy <= threshold
#         max_exceeded = region_start and elapsed_time - region_start >= max_region_size
#
#         if (max_exceeded or is_silence) and region_start:
#             if elapsed_time - region_start >= min_region_size:
#                 regions.append((region_start, elapsed_time))
#                 region_start = None
#
#         elif (not region_start) and (not is_silence):
#             region_start = elapsed_time
#         elapsed_time += chunk_duration
#     return regions
def find_speech_regions(
    filename,
    chunk_duration=0.1,
    min_region_size=0.5,
    max_region_size=6,
    speach_freq_range=(20, 3000),
    non_speech_freq_threshold=2000,
    db_filter=0.0001,
    non_speech_filters=[(0.2, 2), (0.1, np.inf)],
    verbose=False,
):  # pylint: disable=too-many-locals
    """
    Perform voice activity detection on a given audio file.
    """
    reader = wave.open(filename)
    sample_width = reader.getsampwidth()
    rate = reader.getframerate()
    n_channels = reader.getnchannels()
    frame_width = int(rate * chunk_duration)

    n_chunks = int(math.ceil(reader.getnframes() * 1.0 / frame_width))
    audio_data_chunks = [
        np.frombuffer(reader.readframes(frame_width), getattr(np, f"int{8 * sample_width}")).reshape(-1, n_channels).T
        for _ in range(n_chunks)
    ]
    audio_data_chunks_fft = [fft(adc) for adc in audio_data_chunks]
    max_freq_idx = frame_width // 2
    freq_idx_lb = max(1, int(np.floor(speach_freq_range[0] * chunk_duration)))
    freq_idx_ub = min(max_freq_idx, int(np.ceil(speach_freq_range[1] * chunk_duration)))

    energies = [
        np.abs((adcf[:, freq_idx_lb:freq_idx_ub] * adcf[:, -freq_idx_lb:-freq_idx_ub:-1]).sum())
        for adcf in audio_data_chunks_fft
    ]

    threshold = percentile(energies, 0.2)
    max_energy = max(energies)
    energy_threshold = max_energy * db_filter

    elapsed_time = 0

    regions = []
    filtered_regions = []
    region_start = None
    start_idx = 0

    if non_speech_freq_threshold > 0 and len(non_speech_filters) > 1:
        non_speech_freq_idx = min(max_freq_idx, int(np.ceil(non_speech_freq_threshold * chunk_duration)))
        non_speech_energies = [
            np.abs((adcf[:, non_speech_freq_idx:max_freq_idx] * adcf[:, -non_speech_freq_idx:-max_freq_idx:-1]).sum())
            for adcf in audio_data_chunks_fft
        ]
    else:
        non_speech_energies = None

    def filter_ranges(range_start_idx, range_end_idx, duration):
        avg_speech_energy = sum(energies[range_start_idx:range_end_idx]) / (range_end_idx - range_start_idx)
        if avg_speech_energy < energy_threshold:
            return False, f"avg_energy ({avg_speech_energy}) < threshold ({energy_threshold})"
        if non_speech_energies is not None:
            avg_non_speech_energy = sum(non_speech_energies[range_start_idx:range_end_idx]) / (
                range_end_idx - range_start_idx
            )
            ratio = avg_non_speech_energy / avg_speech_energy
            for ratio_threshold, duration_threshold in non_speech_filters:
                if ratio < ratio_threshold and duration < duration_threshold:
                    return True, ""
            return False, f"filtered, duration={duration}, non_speech_energy={avg_non_speech_energy}, speech_energy={avg_speech_energy }, ratio={ratio}"
        return True, ""

    for idx, energy in enumerate(energies):
        is_silence = energy <= threshold
        max_exceeded = region_start and elapsed_time - region_start >= max_region_size

        if (max_exceeded or is_silence) and region_start:
            if elapsed_time - region_start >= min_region_size:
                keep, reason = filter_ranges(start_idx, idx, elapsed_time - region_start)
                if keep:
                    regions.append((region_start, elapsed_time))
                elif verbose:
                    filtered_regions.append((region_start, elapsed_time, reason))
                region_start = None
                start_idx = idx

        elif (not region_start) and (not is_silence):
            region_start = elapsed_time
        elapsed_time += chunk_duration

    if verbose:
        with open("filtered_region.txt", "w") as f:
            f.write("\n".join(f"{s:.3f} {e:.3f} {r}" for s, e, r in filtered_regions))

    return regions


def generate_subtitles(  # pylint: disable=too-many-locals,too-many-arguments
    source_path,
    output=None,
    concurrency=DEFAULT_CONCURRENCY,
    src_language=DEFAULT_SRC_LANGUAGE,
    dst_language=DEFAULT_DST_LANGUAGE,
    subtitle_file_format=DEFAULT_SUBTITLE_FORMAT,
    api_key=None,
    verbose=False,
    skip_recognize=False,
):
    """
    Given an input audio/video file, generate subtitles in the specified language and format.
    """
    print("Extract audios ...")
    audio_filename, audio_rate = extract_audio(source_path, rate=22050)
    print("Searching regions which possibly contains vocal ...")
    regions = find_speech_regions(audio_filename, verbose=verbose)

    pool = multiprocessing.Pool(concurrency)
    converter = FLACConverter(source_path=audio_filename, verbose=verbose)
    recognizer = SpeechRecognizer(language=src_language, rate=audio_rate, api_key=GOOGLE_SPEECH_API_KEY)

    transcripts = []
    if regions:
        try:
            extracted_regions = []
            with tqdm(total=len(regions), desc="Converting speech regions to FLAC files") as pbar:
                for i, extracted_region in enumerate(pool.imap(converter, regions)):
                    extracted_regions.append(extracted_region)
                    pbar.update(1)

            if skip_recognize:
                return ""

            with tqdm(total=len(extracted_regions), desc="Performing speech recognition") as pbar:
                for i, transcript in enumerate(pool.imap(recognizer, extracted_regions)):
                    transcripts.append(transcript)
                    pbar.update(1)

            if src_language.split("-")[0] != dst_language.split("-")[0]:
                if api_key:
                    google_translate_api_key = api_key
                    translator = Translator(dst_language, google_translate_api_key, dst=dst_language, src=src_language)
                    translated_transcripts = []
                    with tqdm(total=len(transcripts), desc="Translating from {0} to {1}: ".format(src_language, dst_language)):
                        for i, transcript in enumerate(pool.imap(translator, transcripts)):
                            translated_transcripts.append(transcript)
                            pbar.update(1)

                    transcripts = translated_transcripts
                else:
                    print(
                        "Error: Subtitle translation requires specified Google Translate API key. "
                        "See --help for further information."
                    )
                    return 1

        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            print("Cancelling transcription")
            raise

    timed_subtitles = [(r, t) for r, t in zip(regions, transcripts) if t]
    formatter = FORMATTERS.get(subtitle_file_format)
    formatted_subtitles = formatter(timed_subtitles)

    dest = output

    if not dest:
        base = os.path.splitext(source_path)[0]
        dest = "{base}.{format}".format(base=base, format=subtitle_file_format)

    with open(dest, "wb") as output_file:
        output_file.write(formatted_subtitles.encode("utf-8"))

    if not verbose:
        os.remove(audio_filename)

    return dest


def validate(args):
    """
    Check that the CLI arguments passed to autosub are valid.
    """
    if args.format not in FORMATTERS:
        print("Subtitle format not supported. " "Run with --list-formats to see all supported formats.")
        return False

    if args.src_language not in list(LANGUAGE_CODES.keys()):
        print("Source language not supported. " "Run with --list-languages to see all supported languages.")
        return False

    if args.dst_language not in list(LANGUAGE_CODES.keys()):
        print("Destination language not supported. " "Run with --list-languages to see all supported languages.")
        return False

    if not args.source_path:
        print("Error: You need to specify a source path.")
        return False

    return True


def main():
    """
    Run autosub as a command-line program.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("source_path", help="Path to the video or audio file to subtitle", nargs="?")
    parser.add_argument(
        "-C", "--concurrency", help="Number of concurrent API requests to make", type=int, default=DEFAULT_CONCURRENCY
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path for subtitles (by default, subtitles are saved in \
                        the same directory and name as the source path)",
    )
    parser.add_argument("-F", "--format", help="Destination subtitle format", default=DEFAULT_SUBTITLE_FORMAT)
    parser.add_argument("-S", "--src-language", help="Language spoken in source file", default=DEFAULT_SRC_LANGUAGE)
    parser.add_argument("-D", "--dst-language", help="Desired language for the subtitles", default=DEFAULT_DST_LANGUAGE)
    parser.add_argument(
        "-K",
        "--api-key",
        help="The Google Translate API key to be used. \
                        (Required for subtitle translation)",
    )
    parser.add_argument("--list-formats", help="List all available subtitle formats", action="store_true")
    parser.add_argument("--list-languages", help="List all available source/destination languages", action="store_true")
    parser.add_argument("-v", "--verbose", type=bool, help="if keep intermediate results", default=False)
    parser.add_argument("-skr", "--skip-recognize", type=bool, help="if prepare data only", default=False)

    args = parser.parse_args()

    if args.list_formats:
        print("List of formats:")
        for subtitle_format in FORMATTERS:
            print("{format}".format(format=subtitle_format))
        return 0

    if args.list_languages:
        print("List of all languages:")
        for code, language in sorted(LANGUAGE_CODES.items()):
            print("{code}\t{language}".format(code=code, language=language))
        return 0

    if not validate(args):
        return 1

    try:
        subtitle_file_path = generate_subtitles(
            source_path=args.source_path,
            concurrency=args.concurrency,
            src_language=args.src_language,
            dst_language=args.dst_language,
            api_key=args.api_key,
            subtitle_file_format=args.format,
            output=args.output,
            verbose=args.verbose,
        )
        print("Subtitles file created at {}".format(subtitle_file_path))
    except KeyboardInterrupt:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
