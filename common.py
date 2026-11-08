"""Shared setup: project-local caches, device selection, tokenizer, and the TinyGSM prompt format.

Import this module before anything that imports huggingface_hub, so every cache stays inside the
project directory.
"""

import contextlib
import functools
import os
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RUNS_DIR = PROJECT_DIR / "runs"
CACHE_DIR = PROJECT_DIR / ".cache"

os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE_DIR / "torchinductor")
os.environ["TRITON_CACHE_DIR"] = str(CACHE_DIR / "triton")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch  # noqa: E402  (after the cache variables are set)

# Pinned upstream revisions, so every machine sees identical data and tokenization.
TINYGSM_REPO, TINYGSM_REV = "TinyGSM/TinyGSM", "5049a7840fa546985eaae4c698d5daf6fdf7b47d"
GSM8K_REPO, GSM8K_REV = "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866"
# TinyGSM's own models use the Phi-1.5 (CodeGen) tokenizer: GPT-2 BPE plus whitespace-run tokens,
# which keeps Python indentation cheap (~152 tokens per TinyGSM program vs ~176 with GPT-2).
TOKENIZER_REPO, TOKENIZER_REV = "microsoft/phi-1_5", "77aa61eeac94fbf33d492b9f2744c98b42d5b5eb"
EOS_ID = 50256  # <|endoftext|>: starts every document; generation stops when the model emits it


# ---------------------------------------------------------------------------------------------
# Devices


def get_device(name=None):
    """CUDA if available (device 0 = the first GPU in CUDA_VISIBLE_DEVICES), else MPS, else CPU."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        print(f"CUDA_VISIBLE_DEVICES={visible} -> using cuda:0 ({torch.cuda.get_device_name(0)})")
        if visible is None and torch.cuda.device_count() > 1:
            print("WARNING: CUDA_VISIBLE_DEVICES is unset; set it to the GPU you were given.")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device):
    """bf16 autocast on CUDA; plain fp32 elsewhere (CPU/MPS are only used for tests)."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ---------------------------------------------------------------------------------------------
# Tokenizer


@functools.cache
def get_tokenizer():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    path = hf_hub_download(TOKENIZER_REPO, "tokenizer.json", revision=TOKENIZER_REV)
    tok = Tokenizer.from_file(path)
    assert tok.token_to_id("<|endoftext|>") == EOS_ID
    return tok


# ---------------------------------------------------------------------------------------------
# TinyGSM prompt format
#
# Every TinyGSM program starts with the same layout: the question split into sentences, one per
# docstring line, indented 4 spaces. The splitter below is the widely used regex sentence splitter
# ("split_into_sentences"); it reproduces the docstring of every sampled TinyGSM row whose
# docstring is a clean copy of its `question` (47,550 / 47,550 rows from all 17 shards). Quotes are
# ''' in ~88% of rows and """ in the rest (only in the first five shards).

FUNC_HEADER = "def simple_math_problem() -> int:\n"

_ALPHABETS = "([A-Za-z])"
_PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
_SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
_ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"


def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(_PREFIXES, "\\1<prd>", text)
    text = re.sub(_WEBSITES, "<prd>\\1", text)
    text = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", text)
    text = re.sub(_MULTIPLE_DOTS, lambda m: "<prd>" * len(m.group(0)) + "<stop>", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + _ALPHABETS + "[.] ", " \\1<prd> ", text)
    text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
    text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", text)
    text = re.sub(" " + _ALPHABETS + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = [s.strip() for s in text.split("<stop>")]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def format_prompt(question, quote="'''"):
    """The function header and docstring exactly as TinyGSM lays them out, ending with a newline.

    The model continues with the function body. Tokenizing this prompt gives an exact prefix of the
    tokenized full program (checked on all sampled rows), so eval prompts match training text.
    """
    lines = "".join(f"    {s}\n" for s in split_into_sentences(question))
    return f"{FUNC_HEADER}    {quote}\n{lines}    {quote}\n"


def split_prompt_solution(question, code):
    """(prompt, solution) if `code` starts with format_prompt(question), else None (~1% of rows)."""
    for quote in ("'''", '"""'):
        prompt = format_prompt(question, quote)
        if code.startswith(prompt):
            return prompt, code[len(prompt):]
    return None
