from shiny import App, ui
from pathlib import Path
from modules import page_appendix, page_predict, page_report

www_dir = Path(__file__).parent / "www"

plotly_head = ui.tags.head(
ui.tags.script(src="https://cdn.plot.ly/plotly-2.35.2.min.js")
)


app_ui = ui.page_fluid(
    plotly_head,
    ui.page_navbar(
    ui.nav_panel("실시간 예측", page_predict.predict_ui()),
    ui.nav_panel("1-11월 분석 리포트", page_report.report_ui()),
    ui.nav_panel("부록", page_appendix.appendix_ui()),
    title="실시간 전기요금 모니터링 대시보드",
    id="main_nav",
    inverse=True,
    ),
)

def server(input, output, session):

    page_predict.predict_server(input, output, session)
    page_report.report_server(input, output, session)
    page_appendix.appendix_server(input, output, session)

app = App(app_ui, server, static_assets=www_dir)