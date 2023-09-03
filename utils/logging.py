import time
from IPython.display import display, clear_output


def print_training_progress(batch, total_batches, train_error, validation_error, start_time):
    # Move the terminal cursor up 3 times
    clear_output(wait=True)

    current_time = time.time()
    elapsed_time = current_time - start_time
    formatted_time = format_time(elapsed_time)

    batch_log = f'Batch: {batch}/{total_batches}'
    error_log = f'Training error: {train_error}, Validation error: {validation_error}'
    time_log = f'Elapsed time: {formatted_time}'

    display(batch_log, error_log, time_log)

def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
