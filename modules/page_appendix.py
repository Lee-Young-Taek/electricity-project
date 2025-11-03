# =============================
# modules/page_appendix.py — Appendix (그래프 + 요약 스토리라인 포함)
# =============================
from shiny import ui, render
from shared import report_df
from viz.appendix_plots import (
    render_data_head,
    render_data_schema,
    render_basic_stats,
    render_missing_summary,
    render_outlier_summary,
    plot_distribution,
    plot_correlation_heatmap,
    plot_time_trend,
    plot_hourly_pattern,
    plot_weekday_pattern,
    plot_worktype_distribution,
    render_pipeline_accordion,
    render_feature_summary,
    render_scaling_info,
    render_leakage_check,
    render_eda_storyline_panels,  # ⬅️ 스토리(요약 텍스트) + 그래프 묶음
)


def appendix_ui():
    return ui.page_fluid(
        ui.tags.link(rel="stylesheet", href="appendix.css"),

        # ===== Header (outside tabs) =====
        ui.div(
            ui.div(
                ui.h4("데이터 부록 (Appendix)", class_="billx-title"),
                ui.span("분석 맥락과 데이터 사전", class_="billx-sub"),
                class_="billx-titlebox",
            ),
            class_="billx-ribbon billx apx-header",
        ),

        ui.navset_card_pill(
            # ========= 개요 =========
            ui.nav_panel(
                "개요",
                ui.layout_columns(
                    ui.div(
                        ui.h5("📋 프로젝트 개요", class_="billx-panel-title"),
                        ui.div(
                            ui.tags.h6("🎯 목적", class_="mt-2 mb-1"),
                            ui.tags.p("공장 전력 사용량/요금 분석 및 예측", class_="ms-1 small"),
                            ui.tags.h6("📅 데이터 기간", class_="mt-3 mb-1"),
                            ui.tags.p("2024년 1월 ~ 11월", class_="ms-1 small"),
                            ui.tags.h6("⏱️ 측정 간격", class_="mt-3 mb-1"),
                            ui.tags.p("15분 단위 (일 96개 레코드)", class_="ms-1 small"),
                            ui.tags.h6("🎯 예측 타겟", class_="mt-3 mb-1"),
                            ui.tags.p("전기요금(원)", class_="ms-1 fw-bold text-primary"),
                            ui.tags.h6("📊 주요 입력 변수", class_="mt-3 mb-1"),
                            ui.tags.ul(
                                ui.tags.li("전력사용량(kWh)"),
                                ui.tags.li("무효전력량 (지상/진상)"),
                                ui.tags.li("역률 (지상/진상)"),
                                ui.tags.li("작업유형"),
                                class_="ms-2",
                            ),
                            class_="billx-panel-body",
                        ),
                        class_="billx-panel",
                    ),
                    ui.div(
                        ui.h5("📚 데이터 사전 (Data Dictionary)", class_="billx-panel-title"),
                        ui.output_ui("apx_schema_table"),
                        class_="billx-panel",
                    ),
                    col_widths=[5, 7],
                ),
                ui.div(
                    ui.h5("🔍 데이터 스냅샷 (상위 10행)", class_="billx-panel-title"),
                    ui.output_ui("apx_head_table"),
                    ui.div({"class": "small-muted mt-2"}, "※ 좌우 스크롤하여 전체 컬럼 확인 가능"),
                    class_="billx-panel",
                ),
            ),

            # ========= EDA =========
            ui.nav_panel(
                "EDA",
                # 0) 스토리라인(요약 텍스트 + 그래프 묶음) — "그래프랑 요약된 내용" 요구사항 반영
                ui.output_ui("apx_eda_storyline"),

                # 1) 기본 테이블/품질/분포/상관/패턴
                ui.div(ui.h5("📊 기초 통계량", class_="billx-panel-title"), ui.output_ui("apx_basic_stats"), class_="billx-panel"),
                ui.layout_columns(
                    ui.div(ui.h5("🔍 데이터 품질 점검", class_="billx-panel-title"), ui.output_ui("apx_missing_summary"), ui.hr({"class": "soft"}), ui.output_ui("apx_outlier_summary"), class_="billx-panel"),
                    ui.div(ui.h5("📈 주요 변수 분포", class_="billx-panel-title"), ui.output_ui("apx_dist_plot"), class_="billx-panel"),
                    col_widths=[5, 7],
                ),
                ui.layout_columns(
                    ui.div(ui.h5("🔗 변수 간 상관관계", class_="billx-panel-title"), ui.output_ui("apx_corr_heatmap"), class_="billx-panel"),
                    ui.div(ui.h5("⏰ 시간대별 패턴", class_="billx-panel-title"), ui.output_ui("apx_hourly_pattern"), class_="billx-panel"),
                    col_widths=[6, 6],
                ),
                ui.layout_columns(
                    ui.div(ui.h5("📅 요일별 패턴 (주말 강조)", class_="billx-panel-title"), ui.output_ui("apx_weekday_pattern"), class_="billx-panel"),
                    ui.div(ui.h5("🏭 작업유형별 분포", class_="billx-panel-title"), ui.output_ui("apx_worktype_dist"), class_="billx-panel"),
                    col_widths=[6, 6],
                ),
                ui.div(ui.h5("📈 시계열 추이 (일별 집계)", class_="billx-panel-title"), ui.output_ui("apx_time_trend"), class_="billx-panel"),
            ),

            # ========= 전처리 =========
            ui.nav_panel(
                "전처리",
                ui.div(ui.h5("🔧 전처리 파이프라인 (9단계)", class_="billx-panel-title"), ui.output_ui("apx_pipeline_accordion"), class_="billx-panel"),
                ui.div(ui.h5("📝 생성된 피처 요약", class_="billx-panel-title"), ui.output_ui("apx_feature_summary"), class_="billx-panel"),
                ui.layout_columns(
                    ui.div(ui.h5("⚙️ 스케일링/인코딩 전략", class_="billx-panel-title"), ui.output_ui("apx_scaling_info"), class_="billx-panel"),
                    ui.div(ui.h5("🛡️ 데이터 누수 점검", class_="billx-panel-title"), ui.output_ui("apx_leakage_check"), class_="billx-panel"),
                    col_widths=[6, 6],
                ),
            ),

            # ========= 모델링 =========
            ui.nav_panel(
                "모델링",
                ui.layout_columns(
                    ui.div(ui.h5("🏆 실험 보드(Leaderboard)", class_="billx-panel-title"), ui.output_ui("apx_leaderboard"), ui.hr({"class": "soft"}), ui.div({"class": "small-muted"}, "※ RMSE/MAE/R², 추론시간 등"), class_="billx-panel"),
                    ui.div(ui.h5("⚙️ 최종 모델 파라미터", class_="billx-panel-title"), ui.output_ui("apx_model_params"), class_="billx-panel"),
                    col_widths=[7, 5],
                ),
                ui.layout_columns(
                    ui.div(ui.output_ui("apx_train_curve"), class_="billx-panel"),
                    ui.div(ui.output_ui("apx_val_curve"), class_="billx-panel"),
                    col_widths=[6, 6],
                ),
            ),

            # ========= 결과/검증 =========
            ui.nav_panel(
                "결과/검증",
                ui.layout_columns(
                    ui.div(ui.h5("📊 평가 지표", class_="billx-panel-title"), ui.output_ui("apx_metrics_table"), ui.hr({"class": "soft"}), ui.output_ui("apx_residual_plot"), class_="billx-panel"),
                    ui.div(ui.h5("🔍 설명가능성 (XAI)", class_="billx-panel-title"), ui.output_ui("apx_shap_summary"), ui.hr({"class": "soft"}), ui.output_ui("apx_shap_bar"), class_="billx-panel"),
                    col_widths=[6, 6],
                ),
                ui.div(ui.h5("🚀 배포/모니터링 체크리스트", class_="billx-panel-title"), ui.output_ui("apx_checklist"), class_="billx-panel"),
            ),
            id="apx_tabs",
        ),
    )


def appendix_server(input, output, session):
    def _ph(text="여기에 표/그래프가 표시됩니다.", h=260):
        return ui.div(text, class_="placeholder d-flex align-items-center justify-content-center small-muted", style=f"height:{h}px; font-size: 0.98rem;")

    # ===== 개요 =====
    @output
    @render.ui
    def apx_schema_table():
        return render_data_schema()

    @output
    @render.ui
    def apx_head_table():
        return render_data_head(report_df, n=10)

    # ===== EDA Storyline (요약 텍스트 + 그래프) =====
    @output
    @render.ui
    def apx_eda_storyline():
        return render_eda_storyline_panels(report_df)

    # ===== EDA 기타 시각화 =====
    @output
    @render.ui
    def apx_basic_stats():
        return render_basic_stats(report_df)

    @output
    @render.ui
    def apx_missing_summary():
        return render_missing_summary(report_df)

    @output
    @render.ui
    def apx_outlier_summary():
        return render_outlier_summary(report_df)

    @output
    @render.ui
    def apx_dist_plot():
        return plot_distribution(report_df)

    @output
    @render.ui
    def apx_corr_heatmap():
        return plot_correlation_heatmap(report_df)

    @output
    @render.ui
    def apx_time_trend():
        return plot_time_trend(report_df)

    @output
    @render.ui
    def apx_hourly_pattern():
        return plot_hourly_pattern(report_df)

    @output
    @render.ui
    def apx_weekday_pattern():
        return plot_weekday_pattern(report_df)

    @output
    @render.ui
    def apx_worktype_dist():
        return plot_worktype_distribution(report_df)

    # ===== 전처리/모델링/결과 (플레이스홀더 포함) =====
    @output
    @render.ui
    def apx_pipeline_accordion():
        return render_pipeline_accordion()

    @output
    @render.ui
    def apx_feature_summary():
        return render_feature_summary()

    @output
    @render.ui
    def apx_scaling_info():
        return render_scaling_info()

    @output
    @render.ui
    def apx_leakage_check():
        return render_leakage_check()

    @output
    @render.ui
    def apx_leaderboard():
        return _ph("모델 리더보드 (RMSE/MAE/R²/Latency)", 260)

    @output
    @render.ui
    def apx_model_params():
        return _ph("최종 모델 하이퍼파라미터", 220)

    @output
    @render.ui
    def apx_train_curve():
        return _ph("학습 곡선(Train)", 300)

    @output
    @render.ui
    def apx_val_curve():
        return _ph("검증 곡선(Validation)", 300)

    @output
    @render.ui
    def apx_metrics_table():
        return _ph("최종 평가 지표 표 (RMSE/MAE/R² 등)", 220)

    @output
    @render.ui
    def apx_residual_plot():
        return _ph("Residual/에러분포", 300)

    @output
    @render.ui
    def apx_shap_summary():
        return _ph("SHAP Summary Plot", 300)

    @output
    @render.ui
    def apx_shap_bar():
        return _ph("상위 피처 영향 (SHAP Bar)", 260)

    @output
    @render.ui
    def apx_checklist():
        return _ph("배포/모니터링 체크리스트 (알람/드리프트/재학습)", 260)