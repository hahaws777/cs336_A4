from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
import hashlib
from collections import Counter



_LANGUAGE_ID_MODEL = None
_NSFW_MODEL = None
_TOXIC_SPEECH_MODEL = None


def _load_fasttext_model(candidates: list[str | None], model_name: str):
    import fasttext

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return fasttext.load_model(candidate)
    raise FileNotFoundError(f"Could not find {model_name}")

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    from resiliparse.extract.html2text import extract_plain_text
    from resiliparse.parse.encoding import detect_encoding

    try:
        html = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        html = html_bytes.decode(encoding, errors="replace")
    return extract_plain_text(html)


def run_identify_language(text: str) -> tuple[Any, float]:
    global _LANGUAGE_ID_MODEL

    if _LANGUAGE_ID_MODEL is None:
        _LANGUAGE_ID_MODEL = _load_fasttext_model(
            [
                os.environ.get("FASTTEXT_LID_MODEL"),
                "/data/classifiers/lid.176.bin",
                "cs336_data/assets/lid.176.bin",
                "data/classifiers/lid.176.bin",
                "lid.176.bin",
            ],
            "lid.176.bin",
        )

    one_line_text = " ".join(text.split())
    if not one_line_text:
        return "unknown", 0.0

    labels, scores = _LANGUAGE_ID_MODEL.predict(one_line_text)
    language = labels[0].removeprefix("__label__")
    return language, float(scores[0])


def run_mask_emails(text: str) -> tuple[str, int]:
    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    return email_pattern.subn("|||EMAIL_ADDRESS|||", text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    phone_pattern = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")
    phone_pattern2 = re.compile(r"\b\d{3}\s\d{3}\s\d{4}\b")
    phone_pattern3  = re.compile(r"\b\(\d{3}\)\s\d{3}\d{4}\b")
    phone_pattern4 = re.compile(r"\b\(\d{3}\)\-\d{3}\-\d{4}\b")

    return (
        phone_pattern.subn("|||PHONE_NUMBER|||", text)[1] +
        phone_pattern2.subn("|||PHONE_NUMBER|||", text)[1] +
        phone_pattern3.subn("|||PHONE_NUMBER|||", text)[1] +
        phone_pattern4.subn("|||PHONE_NUMBER|||", text)[1]
    )


def run_mask_ips(text: str) -> tuple[str, int]:
    ip_pattern = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
    return ip_pattern.subn("|||IP_ADDRESS|||", text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    global _NSFW_MODEL

    if _NSFW_MODEL is None:
        _NSFW_MODEL = _load_fasttext_model(
            [
                os.environ.get("FASTTEXT_NSFW_MODEL"),
                "/data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin",
                "cs336_data/assets/dolma_fasttext_nsfw_jigsaw_model.bin",
                "data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin",
            ],
            "dolma_fasttext_nsfw_jigsaw_model.bin",
        )

    one_line_text = " ".join(text.split())
    labels, scores = _NSFW_MODEL.predict(one_line_text)
    return labels[0].removeprefix("__label__"), float(scores[0])


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    global _TOXIC_SPEECH_MODEL

    if _TOXIC_SPEECH_MODEL is None:
        _TOXIC_SPEECH_MODEL = _load_fasttext_model(
            [
                os.environ.get("FASTTEXT_TOXIC_SPEECH_MODEL"),
                os.environ.get("FASTTEXT_HATESPEECH_MODEL"),
                "/data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin",
                "cs336_data/assets/dolma_fasttext_hatespeech_jigsaw_model.bin",
                "data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin",
            ],
            "dolma_fasttext_hatespeech_jigsaw_model.bin",
        )

    one_line_text = " ".join(text.split())
    labels, scores = _TOXIC_SPEECH_MODEL.predict(one_line_text)
    return labels[0].removeprefix("__label__"), float(scores[0])


def run_classify_quality(text: str) -> tuple[Any, float]:
    # firstly use our trained fasttext model to classify the quality of the text

    raise NotImplementedError


def run_gopher_quality_filter(text: str) -> bool:
    # check if word count is between 50 and 100000
    word_count = len(text.split())
    if word_count < 50 or word_count > 100000:
        return False
    # check the average word length is between 3 and 10
    average_word_length = sum(len(word) for word in text.split()) / len(text.split())
    if average_word_length < 3 or average_word_length > 10:
        return False
    # check if more than 30% of the lines ending with ...
    ellipses_count = sum(1 for line in text.split("\n") if line.endswith("..."))
    if ellipses_count > len(text.split("\n")) * 0.3:
        return False
    # check if more than 80% words have more than 1 english letter
    english_letter_count = sum(1 for word in text.split() if len(re.findall(r'[a-zA-Z]', word)) > 1)
    if english_letter_count < len(text.split()) * 0.8:
        return False
    return True


def hash_line(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    corpus = Counter()
    for input_file in input_files:
        with open(input_file) as f:
            for line in f:
                corpus[line] += 1

    for input_file in input_files:
        input_path = Path(input_file)
        with open(input_path) as f:
            with open(output_path / input_path.name, "w") as f_out:
                for line in f:
                    if corpus[line] == 1:
                        f_out.write(line)
    return output_path


def _word_ngrams(text: str, ngrams: int) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    if len(words) < ngrams:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + ngrams]) for i in range(len(words) - ngrams + 1)}


def _jaccard_similarity(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    kept_documents: list[tuple[Path, str, set[tuple[str, ...]]]] = []
    for input_file in input_files:
        input_path = Path(input_file)
        text = input_path.read_text()
        shingles = _word_ngrams(text, ngrams)
        is_duplicate = any(
            _jaccard_similarity(shingles, kept_shingles) >= jaccard_threshold
            for _, _, kept_shingles in kept_documents
        )
        if not is_duplicate:
            kept_documents.append((input_path, text, shingles))

    for input_path, text, _ in kept_documents:
        (output_path / input_path.name).write_text(text)

    return output_path
