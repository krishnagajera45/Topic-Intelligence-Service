"""
Shared text-preprocessing functions.

Two cleaning presets:
  - ``clean_text_twitter``  — strips @mentions, agent signatures, etc.
  - ``clean_text_general``  — dataset-agnostic cleaning

``clean_for_dataset(text, mode)`` dispatches to the correct preset
based on the ``clean_mode`` setting in the active dataset profile.
"""
import re
import html
import unicodedata


# ── Twitter-specific cleaning ────────────────────────────────────────

def clean_text_twitter(text: str) -> str:
    """Clean tweet text: URLs, @mentions, agent signatures, emojis, etc."""
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)

    # URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # @mentions
    text = re.sub(r'@\w+', '', text)

    # Hashtags (keep word)
    text = re.sub(r'#(\w+)', r'\1', text)

    # Outbound agent signatures (e.g., ^MM, /AY)
    text = re.sub(r'(\s*[\^/][A-Za-z]{1,3})+$', '', text)

    # Mask phone-like numbers
    text = re.sub(r'\b(?:\d[\s\-\.\(\)]*){7,}\d\b', '<PHONE>', text)

    # Normalise version-strings
    text = re.sub(r'\b\d+(?:\.\d+){1,}\b', '<VERSION>', text)

    # Emojis / pictographs
    text = re.sub(
        r'[\U0001F300-\U0001F6FF\U0001F700-\U0001FAFF\U00002700-\U000027BF\U0001F1E6-\U0001F1FF]+',
        '', text,
    )

    # Normalise repeated punctuation
    text = re.sub(r'([!?\.]){2,}', r'\1', text)

    # Whitespace
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


# ── General (dataset-agnostic) cleaning ──────────────────────────────

def clean_text_general(text: str) -> str:
    """General text cleaning suitable for any dataset (news, reviews, …)."""
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)

    # URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Emojis / pictographs
    text = re.sub(
        r'[\U0001F300-\U0001F6FF\U0001F700-\U0001FAFF\U00002700-\U000027BF\U0001F1E6-\U0001F1FF]+',
        '', text,
    )

    # Normalise repeated punctuation
    text = re.sub(r'([!?\.]){2,}', r'\1', text)

    # Whitespace
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


# ── Dispatcher ────────────────────────────────────────────────────────

_CLEANERS = {
    "twitter": clean_text_twitter,
    "general": clean_text_general,
}


def clean_for_dataset(text: str, mode: str = "general") -> str:
    """Dispatch to the correct cleaning function based on *mode*."""
    fn = _CLEANERS.get(mode, clean_text_general)
    return fn(text)
