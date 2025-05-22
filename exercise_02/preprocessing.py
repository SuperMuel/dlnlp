# preprocessing.py

import re
from typing import Sequence
from nltk.stem import PorterStemmer

# --- Constants  ---

CHARACTERS_TO_REMOVE_DEFAULT: list[str] = [
    "\x96",
    "\x91",
    "\x97",
    "\xad",
    "\x84",
    "\x08",
    "\x80",
    "\x8e",
    "\x9e",
    "\x95",
    "\x9a",
    "\x10",
    "\x8d",
    "\uf0b7",
]

CHAR_MAP_DEFAULT: dict[str, str | None] = {
    "%": " ",  # -> replace it with a space (will be tokenized later)
    "´": "'",  # -> Unify apostrophes. Apostrophes are important in words like "don't"
    ")": " ",
    "}": " ",
    "-": None,  # --> will be removed or kept later, depending on the context
    "/": " ",
    "$": " ",
    "_": " ",  # in the dataset, it's used to make text in italics, so we remove it since that's only formatting
    "£": " ",
    "\\": " ",
    "'": "'",
    "“": " ",
    "~": " ",
    "¦": " ",
    "»": " ",
    "^": " ",
    "¨": " ",
    "(": " ",
    "”": " ",
    "|": " ",
    "’": "'",
    ".": " ",
    "[": " ",
    "°": " ",
    "¡": " ",
    "·": " ",
    "!": " ",
    "+": " ",
    "¤": " ",
    "¿": " ",
    ";": " ",
    "{": " ",
    '"': " ",
    "?": " ",
    "<": " ",
    ">": " ",
    "–": " ",
    "®": " ",
    "*": " ",
    "=": " ",
    "#": " ",
    "]": " ",
    "…": " ",
    ":": " ",
    ",": " ",
    "&": " ",
    "₤": " ",
    "‘": "'",
    "§": " ",
    "`": " ",
    "@": " ",
    "«": " ",
    "½": " ",
    "¢": " ",
    "©": " ",
    "″": " ",
    "，": " ",
    "、": " ",
    "★": " ",
    "▼": " ",
}

# --- Individual Preprocessing Functions ---


def clean_text_from_weird_chars(
    text: str,
    weird_chars: Sequence[str] = CHARACTERS_TO_REMOVE_DEFAULT,
    replace_with: str = " ",
) -> str:
    """Removes or replaces specified 'weird' characters from text."""
    for char in weird_chars:
        text = text.replace(char, replace_with)
    return text


def clean_html_tags(text: str) -> str:
    """
    Remove HTML tags from text while preserving content.
    Special case: preserves "<3" (heart symbol).
    """
    # Temporarily replace "<3" with a placeholder
    text_with_placeholder = text.replace("<3", "HEART_SYMBOL_PLACEHOLDER_XYZ")
    # Remove HTML tags
    cleaned_text = re.sub(r"<[^>]*>", "", text_with_placeholder)
    # Restore the heart symbols
    cleaned_text_final = cleaned_text.replace("HEART_SYMBOL_PLACEHOLDER_XYZ", "<3")
    return cleaned_text_final


# Test basic HTML tag removal
assert clean_html_tags("<p>Hello world</p>") == "Hello world"

# Test nested tags
assert clean_html_tags("<div><p>Nested content</p></div>") == "Nested content"

# Test with attributes
assert clean_html_tags('<a href="https://example.com">Link text</a>') == "Link text"

# Test with multiple tags and text
assert clean_html_tags("<h1>Title</h1><p>Paragraph</p>") == "TitleParagraph"

# Test with the special case of "<3" (heart symbol)
assert (
    clean_html_tags(
        "I LUVED IT SO MUCH <3 <br /><br />its about a women...<br /><br /> her<br /><br />"
    )
    == "I LUVED IT SO MUCH <3 its about a women... her"
)

# Test with mixed content
assert (
    clean_html_tags("Text with <b>bold</b> and <i>italic</i>")
    == "Text with bold and italic"
)

# Test with "<em>"
assert clean_html_tags("<em>This is emphasized</em>") == "This is emphasized"

# Test with </SPOILER>
assert clean_html_tags("</SPOILER>This is a spoiler</SPOILER>") == "This is a spoiler"

# Test with empty tags
assert clean_html_tags("<br><hr>Text") == "Text"

# Test with no tags
assert clean_html_tags("Plain text without tags") == "Plain text without tags"


def get_rid_of_non_alphanumeric_characters(
    text: str, char_map: dict[str, str | None] = CHAR_MAP_DEFAULT
) -> str:
    """
    Replaces non-alphanumeric characters based on char_map.
    If replacement in char_map is None, character is kept (e.g., for dashes handled separately).
    """
    for char, replacement in char_map.items():
        if replacement is None:
            continue
        text = text.replace(char, replacement)
    return text


assert (
    get_rid_of_non_alphanumeric_characters("Hello world", CHAR_MAP_DEFAULT)
    == "Hello world"
)
assert (
    get_rid_of_non_alphanumeric_characters("Hello @ world", CHAR_MAP_DEFAULT)
    == "Hello   world"
)
assert (
    get_rid_of_non_alphanumeric_characters(
        "only £300 000 and 7 weeks to write.", CHAR_MAP_DEFAULT
    )
    == "only  300 000 and 7 weeks to write "
)


def keep_or_remove_dashes(text: str) -> str:
    """
    Processes dashes in a text string based on their context.

    Keeps dashes that appear to be part of a word, specifically those
    immediately surrounded by alphanumeric characters. Replaces all
    other dashes (e.g., standalone dashes, dashes next to spaces,
    consecutive dashes not surrounded by alphanumeric characters)
    with a single space.

    Args:
        text: The input text string.

    Returns:
        The processed text string where dashes not part of words are
        replaced by spaces.

    Examples:
        >>> keep_or_remove_dash("a-composed-word")
        'a-composed-word'
        >>> keep_or_remove_dash("an hyphen in - the - middle - of a word")
        'an hyphen in   the   middle   of a word'
        >>> keep_or_remove_dash(" - this is a bullet list but this-is-a-composed-word")
        '   this is a bullet list but this-is-a-composed-word'
        >>> keep_or_remove_dash("multiple---consecutive---dashes")
        'multiple   consecutive   dashes'
    """
    result = []
    n = len(text)
    for i, char in enumerate(text):
        if char == "-":
            # Check if the dash is part of a word (surrounded by alphanumeric chars)
            is_part_of_word = (
                i > 0 and text[i - 1].isalnum() and i < n - 1 and text[i + 1].isalnum()
            )

            if is_part_of_word:
                result.append("-")
            else:
                # Replace dash with space if it's not part of a word
                result.append(" ")
        else:
            result.append(char)
    return "".join(result)


assert keep_or_remove_dashes("a-composed-word") == "a-composed-word"
assert (
    keep_or_remove_dashes("an hyphen in - the - middle - of a word")
    == "an hyphen in   the   middle   of a word"
)
assert (
    keep_or_remove_dashes(" - this is a bullet list but this-is-a-composed-word")
    == "   this is a bullet list but this-is-a-composed-word"
)


def tokenize_and_clean_tokens(text: str) -> list[str]:
    """
    Splits text into tokens by whitespace (assuming punctuation already handled)
    and cleans individual tokens (e.g., removes leading/trailing apostrophes).
    """
    tokens = text.split()
    cleaned_tokens = []
    for token in tokens:
        # Remove leading/trailing apostrophes that might remain
        cleaned_token = token.strip("'")

        # # Remove trailing `'s`` that might remain
        # cleaned_token = cleaned_token.rstrip("'s")

        if cleaned_token:  # Avoid empty strings
            cleaned_tokens.append(cleaned_token)

    return cleaned_tokens


assert tokenize_and_clean_tokens("'Hello'") == ["Hello"]
assert tokenize_and_clean_tokens("'Hello' world") == ["Hello", "world"]
# assert tokenize_and_clean_tokens("'Hello' world's") == ["Hello", "world"]
# assert tokenize_and_clean_tokens("'world's") == ["world"]


def stem_words(words: list[str], stemmer: PorterStemmer) -> list[str]:
    """Applies Porter stemming to a list of words."""
    return [stemmer.stem(word) for word in words]


# --- Main Preprocessing Pipeline Function ---


def full_preprocess_document(
    raw_text: str,
    *,
    stemmer_instance: PorterStemmer | None = None,
    tokens_to_remove: set[str] | None = None,
    custom_char_map: dict[str, str | None] = CHAR_MAP_DEFAULT,
    custom_weird_chars: Sequence[str] = CHARACTERS_TO_REMOVE_DEFAULT,
) -> list[str]:
    """
    Applies the full preprocessing pipeline to a single raw document.
    Order of operations:
    1. Clean weird characters.
    2. Clean HTML tags.
    3. Lowercase.
    4. Replace/remove punctuation and non-alphanumerics (using char_map).
    5. Handle dashes specifically.
    6. Tokenize.
    7. Remove frequent terms.
    8. Apply Porter Stemming.
    """
    if stemmer_instance is None:
        stemmer_instance = PorterStemmer()

    # 1. Clean weird characters
    text = clean_text_from_weird_chars(raw_text, weird_chars=custom_weird_chars)
    # 2. Clean HTML tags
    text = clean_html_tags(text)
    # 3. Lowercase all text
    text = text.lower()
    # 4. Replace/remove punctuation and non-alphanumerics
    text = get_rid_of_non_alphanumeric_characters(text, char_map=custom_char_map)
    # 5. Handle dashes specifically (after general punctuation, as char_map might affect dashes)
    text = keep_or_remove_dashes(text)  # Important: char_map has `'-': None`
    # so get_rid_of_non_alphanumeric_characters leaves them for this step.
    # 6. Tokenize
    #    `tokenize_and_clean_tokens` expects text where punctuation is mostly spaces.
    #    It also handles stripping leading/trailing apostrophes.
    current_tokens = tokenize_and_clean_tokens(text)

    # 7. Remove terms appearing in more than a set threshold (frequent terms)
    #    Note: `tokens_to_remove` should be based on unstemmed tokens if this step
    #    is before stemming, as per exercise.
    filtered_tokens = (
        [token for token in current_tokens if token not in tokens_to_remove]
        if tokens_to_remove
        else current_tokens
    )

    # 8. Apply Porter Stemming
    stemmed_tokens = stem_words(filtered_tokens, stemmer_instance)

    return stemmed_tokens


# --- Example Usage ---
if __name__ == "__main__":
    # Initialize stemmer
    porter_stemmer = PorterStemmer()

    # Example set of frequent tokens to remove (normally derived from training corpus)
    # These should be unstemmed if removal is before stemming.
    TOKENS_TO_REMOVE_EXAMPLE: set[str] = {"the", "a", "is", "this", "and", "to", "br"}

    example_review = """
    This is a TRULY amazing movie! <br /><br /> I loved the characters & the plot twists...
    It's probably one of the best films of 2024. GREAT stuff. Cost: $100.
    Check it out - you won't be disappointed. My score: 9/10. <3
    Some weird chars: \x96 \x91.
    A self-driving car is a well-designed piece of tech.
    """

    print(f"Original Review:\n{example_review}\n")

    processed_tokens = full_preprocess_document(
        raw_text=example_review,
        tokens_to_remove=TOKENS_TO_REMOVE_EXAMPLE,
        stemmer_instance=porter_stemmer,
    )

    print(f"Processed Tokens:\n{processed_tokens}")

    # Example 2: Focusing on dash and HTML
    example_2 = "This is <B>bold</B> and <I>italic</I> with a <a href='#'>link</a>. A heart <3. And well-being."
    processed_tokens_2 = full_preprocess_document(
        raw_text=example_2,
        tokens_to_remove=TOKENS_TO_REMOVE_EXAMPLE,
        stemmer_instance=porter_stemmer,
    )
    print(f"\nOriginal Review 2:\n{example_2}\n")
    print(f"Processed Tokens 2:\n{processed_tokens_2}")
