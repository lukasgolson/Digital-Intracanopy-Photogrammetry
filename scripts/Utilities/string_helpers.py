
def get_numeric_from_string(string):
    return int(''.join(filter(str.isdigit, str(string)))) if any(c.isdigit() for c in str(string)) else 0



def format_elapsed_time(elapsed_time):
    """
    Format the elapsed time in seconds to a human-readable format
    :param elapsed_time:
    :return:
    """
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    time_format = ""
    if hours > 0:
        time_format += f"{hours} hour{'s' if hours > 1 else ''} "

    if minutes > 0:
        time_format += f"{minutes} minute{'s' if minutes > 1 else ''} "

    time_format += f"{seconds} second{'s' if seconds > 1 else ''}"

    return time_format