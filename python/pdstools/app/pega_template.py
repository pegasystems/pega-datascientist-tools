import plotly.graph_objects as go
import plotly.io as pio

pio.templates["pega"] = go.layout.Template(
    # LAYOUT
    layout={
        # Fonts
        # Note - 'family' must be a single string, NOT a list or dict!
        "title": {
            "font": {
                "family": '"Open-Sans, Sans-serif, HelveticaNeue-CondensedBold, Helvetica',
                "size": 18,
                "color": "#333",
            }
        },
        "font": {
            "family": '"Open-Sans, Helvetica Neue, Helvetica, Sans-serif',
            "size": 12,
            "color": "#333",
        },
        # Colorways
        "colorway": [
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
        ],
        # Keep adding others as needed below
        "hovermode": "closest",
    },
)