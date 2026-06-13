"""
File hashing for EXACT duplicate detection. The bytes are the only honest identity:
same content, different name = duplicate; same name, different content = NOT.
We stream in 8KB chunks so a 50MB upload doesn't sit fully in memory.
"""
import hashlib


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of_file(path: str, chunk_size: int = 8192) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()