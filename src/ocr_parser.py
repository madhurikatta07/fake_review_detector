import re


def extract_rating(text):
    """
    Extract rating like:
    - 4
    - 4.5
    - 5 stars
    - ★★★★☆
    """

    # ⭐ FIRST: star symbols (★★★★☆)
    star_match = re.search(r'(★{1,5})', text)
    if star_match:
        return float(len(star_match.group(1)))

    # 🔢 THEN: numeric ratings (4, 4.5, 5 stars)
    num_match = re.search(
        r'\b([1-5](?:\.\d)?)\b\s*(star|stars)?',
        text,
        re.IGNORECASE
    )
    if num_match:
        return float(num_match.group(1))

    return None


def extract_date(text):
    """
    Extract common date formats
    """

    patterns = [
        r'\d{1,2}\s+[A-Za-z]+\s+\d{4}',   # 12 March 2024
        r'\d{1,2}/\d{1,2}/\d{4}',         # 12/03/2024
        r'\d{4}-\d{2}-\d{2}',             # 2024-03-12
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group()

    return None


def extract_username(text):
    """
    Heuristic:
    - First non-empty line
    - Not containing rating
    - Not containing date
    """

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    for line in lines:
        # Skip rating lines
        if re.search(r'(star|★|[1-5](?:\.\d)?)', line, re.IGNORECASE):
            continue

        # Skip date lines
        if re.search(r'\d{4}', line):
            continue

        return line

    return None


# ------------------ TEST ------------------
if __name__ == "__main__":
    sample_text = """
    John_Doe
    ★★★★☆
    Reviewed on 12 March 2024
    This product is amazing and very useful
    """

    print("Rating:", extract_rating(sample_text))
    print("Date:", extract_date(sample_text))
    print("Username:", extract_username(sample_text))
