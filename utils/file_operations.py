import os, errno


def silent_delete(filename):
    try:
        os.remove(filename)
    except OSError as e:
        # If the error cause is something else than non-existent
        # file or directory, re-raise the exception
        if e.errno != errno.ENOENT:
            raise
