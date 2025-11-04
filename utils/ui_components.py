from shiny import ui

def kpi(title: str, value: str, sub: str | None = None):
    return ui.div(
        ui.div(title, class_="kpi-title"),
        ui.div(value, class_="kpi-value"),
        ui.div(sub, class_="kpi-sub") if sub else None,
        class_="kpi",
    )

def section_caption(text: str):
    return ui.div({"class": "small-muted"}, text)

def placeholder(text: str, height: int = 420):
    return ui.div({"class": "placeholder", "style": f"height:{height}px"}, text)
