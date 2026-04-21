"""Miscellaneous helpers that don't belong to a single concern."""

from functools import partial
from operator import is_not


def safe_flatten_list(alist: list | None, extras: list | None = None) -> list | None:
    """Flatten one level of ``alist``, drop ``None`` entries, and prepend ``extras``.

    The result is order-preserving and de-duplicated. Strings are treated as
    atoms (not iterated). Both ``alist`` and ``extras`` are read-only — the
    caller's lists are never mutated. Returns ``None`` when the result would
    be empty so callers can use the truthiness as a "no grouping" signal.
    """
    if alist is None:
        alist = []
    alist = list(filter(partial(is_not, None), alist))
    alist = [item for sublist in [[item] if type(item) is not list else item for item in alist] for item in sublist]
    alist = list(filter(partial(is_not, None), alist))
    unique_alist: list = list(extras) if extras else []
    seen_ids: set[int] = {id(x) for x in unique_alist}
    seen_hashable: set = set()
    for x in unique_alist:
        try:
            seen_hashable.add(x)
        except TypeError:
            pass
    for item in alist:
        try:
            if item in seen_hashable:
                continue
            seen_hashable.add(item)
        except TypeError:
            if id(item) in seen_ids:
                continue
        seen_ids.add(id(item))
        unique_alist.append(item)
    return unique_alist or None


# TODO: perhaps the color / plot utils should move into a separate file
def legend_color_order(fig):
    """Orders legend colors alphabetically in order to provide pega color
    consistency among different categories
    """
    colorway = [
        "#001F5F",  # dark blue
        "#10A5AC",
        "#F76923",  # orange
        "#661D34",  # wine
        "#86CAC6",  # mint
        "#005154",  # forest
        "#86CAC6",  # mint
        "#5F67B9",  # violet
        "#FFC836",  # yellow
        "#E63690",  # pink
        "#AC1361",  # berry
        "#63666F",  # dark grey
        "#A7A9B4",  # medium grey
        "#D0D1DB",  # light grey
    ]
    colors = []
    for trace in fig.data:
        if trace.legendgroup is not None:
            colors.append(trace.legendgroup)
    colors.sort()

    # https://github.com/pegasystems/pega-datascientist-tools/issues/201
    if len(colors) >= len(colorway):
        return fig

    indexed_colors = {k: v for v, k in enumerate(colors)}
    for trace in fig.data:
        if trace.legendgroup is not None:
            try:
                trace.marker.color = colorway[indexed_colors[trace.legendgroup]]
                trace.line.color = colorway[indexed_colors[trace.legendgroup]]
            except AttributeError:  # pragma: no cover
                pass

    return fig
