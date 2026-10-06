"""Minimal ctypes wrapper around the system libarchive (reads RAR/RAR5/ZIP/TAR without extra Python packages)."""
import ctypes
import ctypes.util

_path = ctypes.util.find_library("archive") or "libarchive.so.13"
_la = ctypes.CDLL(_path)
_la.archive_read_new.restype = ctypes.c_void_p
_la.archive_read_support_format_all.argtypes = [ctypes.c_void_p]
_la.archive_read_support_filter_all.argtypes = [ctypes.c_void_p]
_la.archive_read_open_filename.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
_la.archive_read_open_memory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
_la.archive_read_next_header.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
_la.archive_entry_pathname_utf8.argtypes = [ctypes.c_void_p]
_la.archive_entry_pathname_utf8.restype = ctypes.c_char_p
_la.archive_entry_pathname.argtypes = [ctypes.c_void_p]
_la.archive_entry_pathname.restype = ctypes.c_char_p
_la.archive_entry_size.argtypes = [ctypes.c_void_p]
_la.archive_entry_size.restype = ctypes.c_int64
_la.archive_read_data.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
_la.archive_read_data.restype = ctypes.c_ssize_t
_la.archive_read_data_skip.argtypes = [ctypes.c_void_p]
_la.archive_error_string.argtypes = [ctypes.c_void_p]
_la.archive_error_string.restype = ctypes.c_char_p
_la.archive_read_free.argtypes = [ctypes.c_void_p]
_la.archive_entry_filetype.argtypes = [ctypes.c_void_p]
_la.archive_entry_filetype.restype = ctypes.c_uint
AE_IFDIR = 0o040000


def iter_archive(path=None, data=None, want=lambda name: True, errors=None):
    """Yield (name, bytes) for every entry accepted by `want`; other entries are skipped.
    If `errors` is a list, entries that cannot be decoded are recorded there and skipped instead of raising."""
    a = _la.archive_read_new()
    _la.archive_read_support_format_all(a)
    _la.archive_read_support_filter_all(a)
    keep = None
    if path is not None:
        r = _la.archive_read_open_filename(a, path.encode(), 1 << 20)
    else:
        keep = ctypes.create_string_buffer(data, len(data))
        r = _la.archive_read_open_memory(a, keep, len(data))
    if r != 0:
        raise IOError(_la.archive_error_string(a))
    entry = ctypes.c_void_p()
    buf = ctypes.create_string_buffer(1 << 20)
    try:
        while True:
            r = _la.archive_read_next_header(a, ctypes.byref(entry))
            if r == 1:            # ARCHIVE_EOF
                break
            if r < -10:
                if errors is None:
                    raise IOError(_la.archive_error_string(a))
                errors.append(("<header>", (_la.archive_error_string(a) or b"").decode()))
                break
            raw = _la.archive_entry_pathname_utf8(entry) or _la.archive_entry_pathname(entry) or b""
            name = raw.decode("utf-8", "replace")
            if _la.archive_entry_filetype(entry) == AE_IFDIR or not want(name):
                _la.archive_read_data_skip(a)
                continue
            chunks, bad = [], False
            while True:
                n = _la.archive_read_data(a, buf, len(buf))
                if n < 0:
                    if errors is None:
                        raise IOError(_la.archive_error_string(a))
                    errors.append((name, (_la.archive_error_string(a) or b"").decode()))
                    bad = True
                    break
                if n == 0:
                    break
                chunks.append(buf.raw[:n])
            if not bad:
                yield name, b"".join(chunks)
    finally:
        _la.archive_read_free(a)
