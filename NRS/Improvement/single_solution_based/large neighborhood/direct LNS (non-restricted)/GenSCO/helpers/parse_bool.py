def parse_bool(s: str) -> bool:
    if s.lower() in {'true', '1'}:
        return True
    elif s.lower() in {'false', '0'}:
        return False
    else:
        raise ValueError(f'Invalid bool string: {s}')
