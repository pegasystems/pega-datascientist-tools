import plotly.graph_objects as go
import plotly.io as pio

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
neutral_positive = [(0, "#FFB546"), (0.1, "#FCE880"), (0.6, "#66CB66"), (1, "#27803E")]

negative_positive = [
    (0, "#FFB546"),
    (0.1, "#FCE880"),
    (0.6, "#66CB66"),
    (1, "#27803E"),
]
positive_negative = [
    (0, "#27803E"),
    (0.01, "#66CB66"),
    (0.5, "#FCE880"),
    (0.6, "#FFB546"),
    (0.8, "#FF853D"),
    (1, "#DE4342"),
]
performance = [
    (0, "#DE4342"),
    (0.1, "#FFB546"),
    (0.3, "#66CB66"),
    (0.6, "#27803E"),
    (0.9, "#FFB546"),
    (1, "#0000FF"),
]

success = [(0, "#DE4342"), (0.2, "#66CB66"), (1, "#27803E")]

pio.templates["pega"] = go.layout.Template(
    layout={
        "colorway": colorway,
        "hovermode": "closest",
        "font": {"family": "Open Sans, Arial, sans-serif", "size": 12},
        "title": {"font": {"size": 16}, "x": 0.02, "xanchor": "left"},
        "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
        "legend": {
            "bgcolor": "rgba(255,255,255,0.7)",
            "borderwidth": 0,
        },
        "xaxis": {"automargin": True, "title": {"standoff": 8}},
        "yaxis": {"automargin": True, "title": {"standoff": 8}},
        "annotationdefaults": {"font": {"size": 12}},
    },
)

pio.templates["neutral_positive"] = go.layout.Template(
    layout={"colorway": colorway, "colorscale": {"sequential": neutral_positive}},
)

pio.templates["negative_positive"] = go.layout.Template(
    layout={"colorway": colorway, "colorscale": {"sequential": negative_positive}},
)

pio.templates["performance"] = go.layout.Template(
    layout={"colorway": colorway, "colorscale": {"sequential": performance}},
)

pio.templates["success"] = go.layout.Template(
    layout={"colorway": colorway, "colorscale": {"sequential": success}},
)
