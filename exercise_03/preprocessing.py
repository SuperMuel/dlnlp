import re
import string
import collections


def _remove_html_tags(text: str) -> str:
    """
    Removes HTML tags from a string.

    Args:
        text: The input string, potentially containing HTML tags.

    Returns:
        The string with HTML tags removed.
    """
    # A common regex to remove HTML tags. It matches '<' followed by
    # any characters except '>', one or more times, followed by '>'.
    clean_text = re.sub(r"<[^>]+>", "", text)
    return clean_text


def _remove_non_printable(text: str) -> str:
    return "".join(c for c in text if c.isprintable())


assert _remove_non_printable("Hello\x00World") == "HelloWorld"
assert _remove_non_printable("Hello\x09World") == "HelloWorld"

# Tests for _remove_html_tags
assert _remove_html_tags("This is <b>bold</b> text.") == "This is bold text."
assert (
    _remove_html_tags("<p>A paragraph.</p> <br/> Another line.")
    == "A paragraph.  Another line."
)
assert _remove_html_tags("No HTML here.") == "No HTML here."
assert _remove_html_tags("") == ""
assert _remove_html_tags("<tag attribute='value'>Content</tag>") == "Content"
assert (
    _remove_html_tags("Text with <img src='image.png'> an image.")
    == "Text with  an image."
)


def _to_lowercase_if_needed(text: str, to_lowercase_flag: bool) -> str:
    """
    Converts the text to lowercase if the flag is set.

    Args:
        text: The input string.
        to_lowercase_flag: If True, convert text to lowercase.

    Returns:
        The processed string.
    """
    if to_lowercase_flag:
        return text.lower()
    return text


# Tests for _to_lowercase_if_needed
assert _to_lowercase_if_needed("Hello World", True) == "hello world"
assert _to_lowercase_if_needed("Hello World", False) == "Hello World"
assert _to_lowercase_if_needed("ALREADY_LOWER", True) == "already_lower"
assert _to_lowercase_if_needed("", True) == ""
assert _to_lowercase_if_needed("With Numbers 123", True) == "with numbers 123"


def _replace_numbers_if_needed(
    tokens: list[str], replacement_token: str | None
) -> list[str]:
    """
    Replaces tokens that are only numbers with a specified token if the token is provided.

    Args:
        tokens: The input tokens.
        replacement_token: The token to replace numbers with (e.g., "NUMTOKEN").
                           If None, no replacement is done.

    Returns:
        The processed tokens with numbers potentially replaced.
    """
    if replacement_token is None:
        return tokens

    pattern = r"^[+-]?(\d+(\.\d*)?|\.\d+)$"

    _tokens = []
    for token in tokens:
        if re.fullmatch(pattern, token):
            _tokens.append(replacement_token)
        else:
            _tokens.append(token)
    return _tokens


# Tests for _replace_numbers_if_needed
assert _replace_numbers_if_needed(["Review", "123", "and", "45", "."], "NUMTOKEN") == [
    "Review",
    "NUMTOKEN",
    "and",
    "NUMTOKEN",
    ".",
]
assert _replace_numbers_if_needed(["No", "numbers", "here", "."], "NUMTOKEN") == [
    "No",
    "numbers",
    "here",
    ".",
]
assert _replace_numbers_if_needed(["Review", "123", "and", "45", "."], None) == [
    "Review",
    "123",
    "and",
    "45",
    ".",
]
assert _replace_numbers_if_needed(["123", "456"], "N") == ["N", "N"]
assert _replace_numbers_if_needed(["word123word"], "NUMTOKEN") == ["word123word"]
assert _replace_numbers_if_needed([""], "NUMTOKEN") == [""]


def _tokenize_text(text: str, tokenize_on_punctuation_flag: bool) -> list[str]:
    """
    Tokenizes the text based on the specified strategy.

    Args:
        text: The input string.
        tokenize_on_punctuation_flag: If True, punctuation marks are treated as
                                      separate tokens. Otherwise, tokenization is
                                      based on whitespace, and punctuation remains
                                      attached to words.

    Returns:
        A list of tokens. Empty strings resulting from tokenization are removed.
    """
    if tokenize_on_punctuation_flag:
        # Regex to find words (alphanumeric + underscore) or any non-whitespace, non-word character (punctuation)
        tokens = re.findall(r"\w+|[^\s\w]", text)
    else:
        # Splits by any whitespace and handles multiple spaces, leading/trailing spaces.
        tokens = text.split()
    return [token for token in tokens if token]  # Ensure no empty tokens are kept


# Tests for _tokenize_text
assert _tokenize_text("Hello, world!", True) == ["Hello", ",", "world", "!"]
assert _tokenize_text("Hello, world!", False) == ["Hello,", "world!"]
assert _tokenize_text("It's a test.", True) == ["It", "'", "s", "a", "test", "."]
assert _tokenize_text("It's a test.", False) == ["It's", "a", "test."]
assert _tokenize_text("  leading spaces", True) == ["leading", "spaces"]
assert _tokenize_text("trailing spaces  ", False) == ["trailing", "spaces"]
assert _tokenize_text("word1 word2", True) == ["word1", "word2"]
assert _tokenize_text("word1 word2", False) == ["word1", "word2"]
assert _tokenize_text("", True) == []
assert _tokenize_text("", False) == []
assert _tokenize_text("!@#", True) == ["!", "@", "#"]
assert _tokenize_text("!@#", False) == ["!@#"]
assert _tokenize_text("multi   space", False) == ["multi", "space"]
assert _tokenize_text("multi   space", True) == ["multi", "space"]


def _remove_punctuation_tokens_if_needed(
    tokens: list[str], remove_punctuation_flag: bool
) -> list[str]:
    """
    Removes tokens that consist solely of punctuation characters if the flag is set.

    Args:
        tokens: A list of tokens.
        remove_punctuation_flag: If True, remove punctuation-only tokens.

    Returns:
        A list of tokens with punctuation-only tokens potentially removed.
    """
    if not remove_punctuation_flag:
        return tokens

    punctuation_chars = set(string.punctuation)
    # A token is removed if it's not empty and all its characters are punctuation characters.
    return [
        token
        for token in tokens
        if not (token and all(char in punctuation_chars for char in token))
    ]


# Tests for _remove_punctuation_tokens_if_needed
assert _remove_punctuation_tokens_if_needed(["Hello", ",", "world", "!"], True) == [
    "Hello",
    "world",
]
assert _remove_punctuation_tokens_if_needed(["Hello", ",", "world", "!"], False) == [
    "Hello",
    ",",
    "world",
    "!",
]
assert _remove_punctuation_tokens_if_needed(["word.", "another-word"], True) == [
    "word.",
    "another-word",
]
assert _remove_punctuation_tokens_if_needed(["--", "...", "!"], True) == []
assert _remove_punctuation_tokens_if_needed(["Hello", "world"], True) == [
    "Hello",
    "world",
]
assert _remove_punctuation_tokens_if_needed([], True) == []
assert _remove_punctuation_tokens_if_needed(["a", "-", "b"], True) == ["a", "b"]
assert _remove_punctuation_tokens_if_needed(["a-b"], True) == [
    "a-b"
]  # Not purely punctuation


def _get_corpus_token_frequencies(
    list_of_token_lists: list[list[str]],
) -> tuple[collections.Counter[str], int]:
    """
    Calculates the frequency of each token in the entire corpus and the total token count.

    Args:
        list_of_token_lists: A list where each inner list contains tokens of a document.

    Returns:
        A tuple containing:
            - A Counter object mapping tokens to their frequencies.
            - The total number of tokens in the corpus.
    """
    token_counts: collections.Counter[str] = collections.Counter()
    total_tokens = 0
    for tokens in list_of_token_lists:
        token_counts.update(tokens)
        total_tokens += len(tokens)
    return token_counts, total_tokens


# Tests for _get_corpus_token_frequencies
corpus1 = [["a", "b", "a"], ["b", "c"]]
counts1, total1 = _get_corpus_token_frequencies(corpus1)
assert counts1 == collections.Counter({"a": 2, "b": 2, "c": 1})
assert total1 == 5

corpus2 = [["hello", "world"], ["hello", "python"]]
counts2, total2 = _get_corpus_token_frequencies(corpus2)
assert counts2 == collections.Counter({"hello": 2, "world": 1, "python": 1})
assert total2 == 4

corpus3: list[list[str]] = []
counts3, total3 = _get_corpus_token_frequencies(corpus3)
assert counts3 == collections.Counter()
assert total3 == 0

corpus4 = [["token"]]
counts4, total4 = _get_corpus_token_frequencies(corpus4)
assert counts4 == collections.Counter({"token": 1})
assert total4 == 1


def _identify_high_freq_tokens(
    token_frequencies: collections.Counter[str], total_tokens: int, threshold: float
) -> set[str]:
    """
    Identifies tokens whose frequency (count/total_tokens) exceeds a given threshold.

    Args:
        token_frequencies: A Counter object mapping tokens to their frequencies.
        total_tokens: The total number of tokens in the corpus.
        threshold: The frequency threshold (0.0 to 1.0). Tokens with
                   frequency > threshold will be identified.

    Returns:
        A set of high-frequency tokens.
    """
    assert total_tokens > 0
    assert 0.0 <= threshold <= 1.0

    high_freq_set = set()
    for token, count in token_frequencies.items():
        if (count / total_tokens) > threshold:
            high_freq_set.add(token)
    return high_freq_set


# Tests for _identify_high_freq_tokens
freqs1 = collections.Counter({"a": 50, "b": 30, "c": 20})  # total = 100
assert _identify_high_freq_tokens(freqs1, 100, 0.4) == {"a"}
assert _identify_high_freq_tokens(freqs1, 100, 0.25) == {"a", "b"}
assert _identify_high_freq_tokens(freqs1, 100, 0.5) == set()
assert _identify_high_freq_tokens(freqs1, 100, 0.0) == {"a", "b", "c"}
assert (
    _identify_high_freq_tokens(collections.Counter({"a": 5}), 10, 0.5) == set()
)  # 5/10 is not > 0.5
assert _identify_high_freq_tokens(collections.Counter({"a": 6}), 10, 0.5) == {
    "a"
}  # 6/10 is > 0.5


def _filter_tokens_by_set(
    list_of_token_lists: list[list[str]], tokens_to_remove: set[str]
) -> list[list[str]]:
    """
    Removes specified tokens from each list of tokens in the corpus.

    Args:
        list_of_token_lists: A list where each inner list contains tokens of a document.
        tokens_to_remove: A set of tokens to be removed.

    Returns:
        A new list of token lists with the specified tokens removed.
    """
    return [
        [token for token in token_list if token not in tokens_to_remove]
        for token_list in list_of_token_lists
    ]


# Tests for _filter_tokens_by_set
corpus_to_filter = [["a", "b", "c"], ["b", "d", "a"]]
to_remove_set1 = {"a", "d"}
assert _filter_tokens_by_set(corpus_to_filter, to_remove_set1) == [["b", "c"], ["b"]]

to_remove_none: set[str] = set()
assert _filter_tokens_by_set(corpus_to_filter, to_remove_none) == corpus_to_filter

to_remove_all = {"a", "b", "c", "d"}
assert _filter_tokens_by_set(corpus_to_filter, to_remove_all) == [[], []]
assert _filter_tokens_by_set([[]], {"a"}) == [[]]


# --- Main Pre-processing Function ---


def preprocess(
    reviews: list[str],
    tokenize_on_punctuation: bool = True,
    to_lowercase: bool = False,
    remove_punctuation_tokens: bool = False,
    high_freq_term_threshold: float | None = None,
    number_replacement_token: str | None = None,
    remove_non_printable: bool = True,
) -> list[list[str]]:
    """
    Pre-processes a list of raw review strings according to specified options.
    HTML tags are always removed as a first step.

    The pipeline order for each review is:
    1. HTML tag removal.
    2. Lowercase conversion (if `to_lowercase` is True).
    3. Number replacement (if `number_replacement_token` is provided).
    4. Tokenization (based on `tokenize_on_punctuation`).
    5. Punctuation token removal (if `remove_punctuation_tokens` is True).

    After individual processing, a corpus-wide step is performed:
    6. High-frequency term removal (if `high_freq_term_threshold` is provided).

    Args:
        reviews: A list of raw review strings.
        tokenize_on_punctuation: If True, punctuation marks are treated as
                                 separate tokens. Otherwise, tokenization is
                                 based on whitespace. Default is False.
        to_lowercase: If True, convert all text to lowercase. Default is False.
        remove_punctuation_tokens: If True, remove tokens that consist purely
                                   of punctuation characters after tokenization.
        high_freq_term_threshold: If a float (0.0-1.0), remove terms whose corpus
                                  frequency (count / total tokens in corpus)
                                  exceeds this threshold. If None, no
                                  high-frequency term removal is performed.
        number_replacement_token: If a string (e.g., "NUMTOKEN"), replace all
                                  sequences of digits with this token. If None,
                                  numbers are not replaced. Default is None.
                                  Should not contain any special characters or spaces.
        remove_non_printable: If True, remove non-printable characters.

    Returns:
        A list of lists of strings, where each inner list contains the
        processed tokens for a review.

    Raises:
        ValueError: If `high_freq_term_threshold` is provided but not
                    within the range [0.0, 1.0].
    """
    if not reviews:
        return []

    if high_freq_term_threshold is not None and not (
        0.0 <= high_freq_term_threshold <= 1.0
    ):
        raise ValueError(
            "high_freq_term_threshold must be between 0.0 and 1.0, or None."
        )

    if number_replacement_token is not None:
        if any(char in number_replacement_token for char in string.punctuation):
            raise ValueError(
                "number_replacement_token should not contain any special characters or spaces."
            )
        if any(char in number_replacement_token for char in " "):
            raise ValueError("number_replacement_token should not contain any spaces.")

    processed_reviews_intermediate: list[list[str]] = []

    for review_text in reviews:
        # Remove non-printable characters
        if remove_non_printable:
            review_text = _remove_non_printable(review_text)

        # HTML Tag Removal (always on)
        current_text = _remove_html_tags(review_text)

        # Lowercase
        current_text = _to_lowercase_if_needed(current_text, to_lowercase)

        # Tokenization
        tokens = _tokenize_text(current_text, tokenize_on_punctuation)

        # Punctuation Token Removal
        tokens = _remove_punctuation_tokens_if_needed(tokens, remove_punctuation_tokens)

        # Number Replacement
        tokens = _replace_numbers_if_needed(tokens, number_replacement_token)

        processed_reviews_intermediate.append(tokens)

    if high_freq_term_threshold is None:
        return processed_reviews_intermediate

    # 6. (Corpus-wide) High-Frequency Term Removal
    token_frequencies, total_tokens = _get_corpus_token_frequencies(
        processed_reviews_intermediate
    )

    if (
        total_tokens == 0
    ):  # If corpus is empty after initial processing, return it as is
        return processed_reviews_intermediate

    tokens_to_remove = _identify_high_freq_tokens(
        token_frequencies, total_tokens, high_freq_term_threshold
    )

    final_processed_reviews = _filter_tokens_by_set(
        processed_reviews_intermediate, tokens_to_remove
    )

    return final_processed_reviews


# --- Tests for preprocess_reviews ---

# Scenario 0: HTML tag removal test
reviews0 = ["<p>Hello <b>World</b>.</p>", "Test 123! <br/> No tags here."]
# After HTML removal: ["Hello World.", "Test 123!  No tags here."]
# Default tokenization: [["Hello", "World."], ["Test", "123!", "No", "tags", "here."]]
expected0 = [["Hello", "World."], ["Test", "123!", "No", "tags", "here."]]
assert preprocess(reviews0, tokenize_on_punctuation=False) == expected0, (
    f"S0 failed: {preprocess(reviews0, tokenize_on_punctuation=False)}"
)

# Scenario 1: Basic tokenization (defaults)
reviews1 = ["Hello World.", "Test 123!"]
expected1 = [["Hello", "World."], ["Test", "123!"]]
assert preprocess(reviews1, tokenize_on_punctuation=False) == expected1, (
    f"S1 failed: {preprocess(reviews1, tokenize_on_punctuation=False)}"
)

# Scenario 2: Lowercase and number replacement
expected2 = [["hello", "world", "."], ["test", "NUMTOKEN", "!"]]
assert (
    preprocess(reviews1, to_lowercase=True, number_replacement_token="NUMTOKEN")
    == expected2
), (
    f"S2 failed: {preprocess(reviews1, to_lowercase=True, number_replacement_token='NUMTOKEN')}"
)

# Scenario 3: Tokenize on punctuation and remove punctuation tokens
reviews3 = ["It's good, really!", "Score: 10/10."]
# HTML removal: no change
# Lowercase (if on): no change for this test
# Num replace (if on): no change for this test
# Tokenize on punct:
#   "It's good, really!" -> ["It", "'", "s", "good", ",", "really", "!"]
#   "Score: 10/10." -> ["Score", ":", "10", "/", "10", "."]
# Remove punct tokens:
#   -> ["It", "s", "good", "really"]
#   -> ["Score", "10", "10"]
expected3_final_corrected = [["It", "s", "good", "really"], ["Score", "10", "10"]]
assert (
    preprocess(reviews3, tokenize_on_punctuation=True, remove_punctuation_tokens=True)
    == expected3_final_corrected
), (
    f"S3 failed: {preprocess(reviews3, tokenize_on_punctuation=True, remove_punctuation_tokens=True)}"
)

# Scenario 4: Exercise 3, PT#1 initial scenario
# - (HTML tags removed)
# - convert all the reviews to lowercase
# - remove punctuations (means tokenize on punctuation AND remove punctuation tokens)
reviews_ex3 = ["This is <b>Review #1</b>, with some punctuation!! And UPPERCASE."]
# HTML removal: "This is Review #1, with some punctuation!! And UPPERCASE."
# Lowercase: "this is review #1, with some punctuation!! and uppercase."
# Tokenize on punct: ["this", "is", "review", "#", "1", ",", "with", "some", "punctuation", "!", "!", "and", "uppercase", "."]
# Remove punct tokens: ["this", "is", "review", "1", "with", "some", "punctuation", "and", "uppercase"] (since # , ! . are punctuation)
expected_ex3 = [
    ["this", "is", "review", "1", "with", "some", "punctuation", "and", "uppercase"]
]
assert (
    preprocess(
        reviews_ex3,
        to_lowercase=True,
        tokenize_on_punctuation=True,
        remove_punctuation_tokens=True,
        number_replacement_token=None,
        high_freq_term_threshold=None,
    )
    == expected_ex3
), (
    f"S4 failed: {preprocess(reviews_ex3, to_lowercase=True, tokenize_on_punctuation=True, remove_punctuation_tokens=True)}"
)

# Scenario 5: High-frequency term removal
reviews5 = ["the cat sat", "the dog sat", "the bird flew"]
# HTML removal: no change
# Intermediate (lowercase=True, default tokenization):
# [["the", "cat", "sat"], ["the", "dog", "sat"], ["the", "bird", "flew"]]
# Frequencies: "the": 3, "cat": 1, "sat": 2, "dog": 1, "bird": 1, "flew": 1. Total tokens = 9
# "the": 3/9 = 0.333...
# "sat": 2/9 = 0.222...
# If threshold = 0.3, "the" (freq > 0.3) should be removed.
expected5 = [["cat", "sat"], ["dog", "sat"], ["bird", "flew"]]
assert (
    preprocess(reviews5, to_lowercase=True, high_freq_term_threshold=0.3) == expected5
), (
    f"S5a failed: {preprocess(reviews5, to_lowercase=True, high_freq_term_threshold=0.3)}"
)

# If threshold = 0.2, "the" (0.333 > 0.2) and "sat" (0.222 > 0.2) should be removed.
expected5_b = [["cat"], ["dog"], ["bird", "flew"]]
assert (
    preprocess(reviews5, to_lowercase=True, high_freq_term_threshold=0.2) == expected5_b
), (
    f"S5b failed: {preprocess(reviews5, to_lowercase=True, high_freq_term_threshold=0.2)}"
)

# Scenario 6: Empty input
assert preprocess([]) == [], "S6 failed"

# Scenario 7: All options combined, including HTML
reviews7 = [
    "The <b>QUICK</b> Brown Fox 123, <p>jumps over</p> the lazy dog 456!! THE END."
]
# 1. HTML removal: "The QUICK Brown Fox 123, jumps over the lazy dog 456!! THE END."
# 2. Lowercase: "the quick brown fox 123, jumps over the lazy dog 456!! the end."
# 3. Num replace "NUMTOKEN": "the quick brown fox NUMTOKEN, jumps over the lazy dog NUMTOKEN!! the end."
# 4. Tokenize on punct: ["the", "quick", "brown", "fox", "NUMTOKEN", ",", "jumps", "over", "the", "lazy", "dog", "NUMTOKEN", "!", "!", "the", "end", "."]
# 5. Remove punct tokens: ["the", "quick", "brown", "fox", "<N>", "jumps", "over", "the", "lazy", "dog", "<N>", "the", "end"]
# Intermediate list: [["the", "quick", "brown", "fox", "<N>", "jumps", "over", "the", "lazy", "dog", "<N>", "the", "end"]]
# Frequencies: "the":3, "quick":1, "brown":1, "fox":1, "<N>":2, "jumps":1, "over":1, "lazy":1, "dog":1, "end":1. Total = 13
# "the": 3/13 approx 0.23
# "<N>": 2/13 approx 0.153
# If high_freq_threshold = 0.2: "the" is removed.
expected7 = [
    [
        "quick",
        "brown",
        "fox",
        "NUMTOKEN",
        "jumps",
        "over",
        "lazy",
        "dog",
        "NUMTOKEN",
        "end",
    ]
]
result7 = preprocess(
    reviews7,
    to_lowercase=True,
    number_replacement_token="NUMTOKEN",
    tokenize_on_punctuation=True,
    remove_punctuation_tokens=True,
    high_freq_term_threshold=0.2,
)
assert result7 == expected7, f"S7 failed: {result7}"

# Scenario 8: Test edge case for high_freq_term_threshold: if corpus becomes empty before this step
reviews8 = ["<br/>!!!", "...", "<b>???</b>"]
# After HTML removal: ["!!!", "...", "???"]
# After lowercase (no change), num_replace (no change), tokenize_on_punct, remove_punct_tokens:
# intermediate = [[], [], []]
# total_tokens will be 0.
expected8: list[list[str]] = [[], [], []]
result8 = preprocess(
    reviews8,
    to_lowercase=True,
    tokenize_on_punctuation=True,
    remove_punctuation_tokens=True,
    high_freq_term_threshold=0.1,
)
assert result8 == expected8, f"S8 failed: {result8}"

# Scenario 9: Test ValueError for high_freq_term_threshold
try:
    preprocess(["a b c"], high_freq_term_threshold=1.1)
    assert False, "S9 failed: ValueError not raised for invalid threshold"
except ValueError:
    pass  # Expected

try:
    preprocess(["a b c"], high_freq_term_threshold=-0.1)
    assert False, "S9 failed: ValueError not raised for invalid threshold"
except ValueError:
    pass  # Expected
