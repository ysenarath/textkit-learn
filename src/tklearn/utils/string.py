import textwrap


def shorten(x, dtype):
    """Shorten the string representation of a value.

    Parameters
    ----------
    x : Any
        The value to be shortened.
    dtype : type
        The type of the value.

    Returns
    -------
    str
        The shortened string representation of the value.
    """
    x = textwrap.shorten(str(x), width=20)
    return "'" + str(x) + "'" if dtype is str else str(x)


def repr_list(data, dtype=None):
    """Return a string representation of a list.

    Parameters
    ----------
    data : list
        The list to be represented.
    dtype : type
        The type of the list elements.

    Returns
    -------
    str
        The string representation of the list.
    """
    result = "["
    if len(data) > 0:
        result += shorten(str(data[0]), dtype)
    if len(data) > 1:
        result += ", " + shorten(str(data[1]), dtype)
    if len(data) > 3:
        result += ", ..."
    if len(data) > 2:
        result += ", " + shorten(str(data[-1]), dtype)
    result += "]"
    return result
