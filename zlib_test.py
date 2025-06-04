import zlib

def zlib_entropy(text: str) -> float:
    original_size = len(text.encode('utf-8'))
    compressed_size = len(zlib.compress(text.encode('utf-8')))
    if original_size == 0:
        return 0.0
    return compressed_size / original_size

# Example usage
text_1 = "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat."
text_2 = "x7vA!e@1qZ#l%9bN^mWk^r$2dXz&U3jH!aM5kP#nD7g"

entropy_1 = zlib_entropy(text_1)
entropy_2 = zlib_entropy(text_2)

print(f"Zlib entropy for repetitive text: {entropy_1:.2f}")
print(f"Zlib entropy for random text:     {entropy_2:.2f}")

