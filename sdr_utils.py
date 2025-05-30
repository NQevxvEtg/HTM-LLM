# sdr_utils.py
import numpy as np
import re
import hashlib

_token_sdr_cache = {} # Module-level cache

def create_sdr(size, active_bits):
    sdr = np.zeros(size, dtype=int)
    if active_bits > size:
        active_bits = size
    active_indices = np.random.choice(size, active_bits, replace=False)
    sdr[active_indices] = 1
    return sdr

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def get_token_sdr(token, sdr_size, active_bits):
    global _token_sdr_cache # Explicitly declare if modifying a global from within a different scope in some contexts
                            # For module-level, it's generally fine.
    if token not in _token_sdr_cache:
        seed_val = int(hashlib.sha256(token.encode('utf-8')).hexdigest(), 16) % (2**32)
        np.random.seed(seed_val)
        _token_sdr_cache[token] = create_sdr(sdr_size, active_bits)
    return _token_sdr_cache[token].copy()

def clear_sdr_cache(): # Good utility to have
    global _token_sdr_cache
    _token_sdr_cache = {}