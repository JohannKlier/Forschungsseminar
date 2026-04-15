from __future__ import annotations


def sort_category_values(values):
    def sort_key(value):
        as_string = str(value)
        try:
            return (0, float(as_string), as_string)
        except ValueError:
            return (1, as_string.lower(), as_string)

    return sorted([str(value) for value in values], key=sort_key)
