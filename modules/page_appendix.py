# ============================ modules/page_appendix.py (CSS-scoped) ============================
from shiny import ui, render
from shared import report_df

# ---- Tab: 개요
from viz.appendix_overview import (
    render_data_head,
    render_data_schema,
)

# ---- Tab: EDA
from viz.appendix_eda import (
    render_calendar_alignment_storyline,
    render_calendar_overlay,
    render_midnight_rollover_fix,
    render_eda_storyline_panels,
    render_basic_stats,
    render_missing_summary,
    render_outlier_summary,
    plot_distribution,
    plot_correlation_heatmap,
    plot_time_trend,
    plot_hourly_pattern,
    plot_weekday_pattern,
    plot_worktype_distribution,
    # 파생 피처 근거 분석
    render_lag_window_acf,
    render_how_profile_stability,
    render_holiday_peak_checks,
    render_unitprice_and_logtarget,
)

# ---- Tab: 전처리
from viz.appendix_preproc import (
    render_pipeline_accordion,
    render_feature_summary,
    render_scaling_info,
    render_leakage_check,
)

# ---- Tab: 모델링
from viz.appendix_modeling import (
    render_leaderboard,
    render_model_params,
    render_train_curve,
    render_val_curve,
)

# ---- Tab: 결과/검증
from viz.appendix_results import (
    render_metrics_table,
    render_residual_plot,
    render_shap_summary,
    render_shap_bar,
    render_deploy_checklist,
)


# NOTE: Wrap entire appendix in a unique scope to avoid CSS collisions.
# - All custom classes already prefix with `billx-` and IDs with `apx_`.
# - The `.apx-scope` wrapper isolates any descendant CSS rules you may add in appendix.css
#   (e.g., by writing selectors like `.apx-scope .billx-panel { ... }`).

def appendix_ui():
    return ui.page_fluid(
        # Include stylesheet and a tiny scoping helper
        ui.tags.link(rel="stylesheet", href="appendix.css"),
        ui.tags.style("""
        /* Optional minimal safe defaults inside the scope to avoid global overrides */
        .apx-scope { --apx-gap: 12px; }
        .apx-scope .billx-titlebox { margin-bottom: var(--apx-gap); }
        .apx-scope .billx-panel { background: #fff; border: 1px solid #e2e8f0; border-radius: .5rem; padding: 12px; }
        .apx-scope .billx-panel-title { margin: 0 0 8px 0; font-weight: 600; }
        .apx-scope .soft { opacity: .4; }
        """),
        ui.div(
            # ===== Scope wrapper starts here =====
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
                            ui.h5("프로젝트 개요", class_="billx-panel-title"),
                            ui.div(
                                ui.tags.h6("1. 목적", class_="mt-2 mb-1"),
                                ui.tags.p("공장 전력 사용량/요금 분석 및 예측", class_="ms-1 small"),
                                ui.tags.h6("2. 데이터 기간", class_="mt-3 mb-1"),
                                ui.tags.p("2024년 1월 ~ 11월", class_="ms-1 small"),
                                ui.tags.h6("3. 측정 간격", class_="mt-3 mb-1"),
                                ui.tags.p("15분 단위 (일 96개 레코드)", class_="ms-1 small"),
                                ui.tags.h6("4. 예측 타겟", class_="mt-3 mb-1"),
                                ui.tags.p("전기요금(원)", class_="ms-1 fw-bold text-primary"),
                                ui.tags.h6("5. 주요 입력 변수", class_="mt-3 mb-1"),
                                ui.tags.ul(
                                    ui.tags.li("전력사용량(kWh)"),
                                    ui.tags.li("지상/진상 무효전력량(kVarh)"),
                                    ui.tags.li("지상/진상 역률(%)"),
                                    ui.tags.li("탄소배출량(tCO2)"),
                                    ui.tags.li("작업유형"),
                                    class_="ms-2",
                                ),
                                class_="billx-panel-body",
                            ),
                            class_="billx-panel",
                        ),
                        ui.div(
                            ui.h5("데이터 사전 (Data Dictionary)", class_="billx-panel-title"),
                            ui.output_ui("apx_schema_table"),
                            class_="billx-panel",
                        ),
                        col_widths=[5, 7],
                    ),
                    ui.div(
                        ui.h5("데이터 스냅샷 (상위 10행)", class_="billx-panel-title"),
                        ui.output_ui("apx_head_table"),
                        class_="billx-panel",
                    ),
                ),

                # ========= EDA =========
                ui.nav_panel(
                    "EDA",

                    # === 1. 데이터 품질 검증 ===
                    ui.div(ui.h5("데이터 정합성 검증", class_="billx-panel-title"), class_="billx-panel"),
                    ui.output_ui("apx_calendar_alignment"),

                    ui.div(
                        ui.layout_columns(
                            ui.input_select(
                                "cal_year", "기준 연도 선택",
                                {"2018":"2018","2019":"2019","2021":"2021","2022":"2022","2023":"2023"},
                                selected="2018"
                            ),
                            ui.input_checkbox_group(
                                "cal_mark", "하이라이트 항목",
                                {"weekend":"주말","holiday":"공휴일"},
                                selected=["weekend","holiday"],
                                inline=True,
                            ),
                            col_widths=[4, 8]
                        ),
                        ui.output_ui("apx_calendar_overlay"),
                        class_="mb-3"
                    ),

                    ui.output_ui("apx_midnight_rollover"),

                    ui.hr({"class": "soft"}),

                    # === 2. 기초 통계 & 품질 ===
                    ui.div(ui.h5("기초 통계 & 데이터 품질", class_="billx-panel-title"), class_="billx-panel"),
                    ui.div(ui.output_ui("apx_basic_stats"), class_="billx-panel"),

                    ui.layout_columns(
                        ui.div(ui.h5("결측치 점검", class_="billx-panel-title"), ui.output_ui("apx_missing_summary"), class_="billx-panel"),
                        ui.div(ui.h5("이상치 처리", class_="billx-panel-title"), ui.output_ui("apx_outlier_summary"), class_="billx-panel"),
                        col_widths=[5, 7],
                    ),

                    ui.hr({"class": "soft"}),

                    # === 3. 시계열 패턴 ===
                    ui.output_ui("apx_eda_storyline"),

                    ui.hr({"class": "soft"}),

                    # === 4. 변수 분석 ===
                    ui.div(ui.h5("변수 분석", class_="billx-panel-title"), class_="billx-panel"),

                    ui.layout_columns(
                        ui.div(ui.h5("변수 간 상관관계", class_="billx-panel-title"), ui.output_ui("apx_corr_heatmap"), class_="billx-panel"),
                        ui.div(ui.h5("주요 변수 분포", class_="billx-panel-title"), ui.output_ui("apx_dist_plot"), class_="billx-panel"),
                        col_widths=[6, 6],
                    ),

                    ui.layout_columns(
                        ui.div(ui.h5("시간대별 패턴", class_="billx-panel-title"), ui.output_ui("apx_hourly_pattern"), class_="billx-panel"),
                        ui.div(ui.h5("요일별 패턴", class_="billx-panel-title"), ui.output_ui("apx_weekday_pattern"), class_="billx-panel"),
                        col_widths=[6, 6],
                    ),

                    ui.layout_columns(
                        ui.div(ui.h5("시계열 추이", class_="billx-panel-title"), ui.output_ui("apx_time_trend"), class_="billx-panel"),
                        ui.div(ui.h5("작업유형별 분포", class_="billx-panel-title"), ui.output_ui("apx_worktype_dist"), class_="billx-panel"),
                        col_widths=[6, 6],
                    ),

                    ui.hr({"class": "soft"}),

                    # === 5. 파생 피처 설계 근거 ===
                    ui.div(
                        ui.h5("파생 피처 설계 근거", class_="billx-panel-title"),
                        ui.div("모델 성능 향상을 위한 파생 피처 설계의 통계적 타당성을 검증합니다.", class_="alert alert-info mb-0"),
                        class_="billx-panel",
                    ),

                    ui.layout_columns(
                        ui.div(ui.h5("시차 상관관계 (ACF)", class_="billx-panel-title"), ui.output_ui("apx_lag_acf"), class_="billx-panel"),
                        ui.div(ui.h5("Hour of Week 안정성", class_="billx-panel-title"), ui.output_ui("apx_how_stability"), class_="billx-panel"),
                        col_widths=[6, 6],
                    ),

                    ui.layout_columns(
                        ui.div(ui.h5("피크시간대 영향", class_="billx-panel-title"), ui.output_ui("apx_holiday_peak"), class_="billx-panel"),
                        ui.div(ui.h5("단가 & 로그변환", class_="billx-panel-title"), ui.output_ui("apx_unitprice_log"), class_="billx-panel"),
                        col_widths=[6, 6],
                    ),
                ),

                # ========= 전처리 =========
                ui.nav_panel(
                    "전처리",
                    ui.div(ui.h5("전처리 파이프라인 (9단계)", class_="billx-panel-title"), ui.output_ui("apx_pipeline_accordion"), class_="billx-panel"),
                    ui.div(ui.h5("생성된 피처 요약", class_="billx-panel-title"), ui.output_ui("apx_feature_summary"), class_="billx-panel"),
                    ui.layout_columns(
                        ui.div(ui.h5("스케일링/인코딩 전략", class_="billx-panel-title"), ui.output_ui("apx_scaling_info"), class_="billx-panel"),
                        ui.div(ui.h5("데이터 누수 점검", class_="billx-panel-title"), ui.output_ui("apx_leakage_check"), class_="billx-panel"),
                        col_widths=[6, 6],
                    ),
                ),

                # ========= 모델링 =========
                ui.nav_panel(
                    "모델링",
                    ui.layout_columns(
                        ui.div(ui.h5("실험 보드(Leaderboard)", class_="billx-panel-title"), ui.output_ui("apx_leaderboard"), ui.hr({"class": "soft"}), ui.div({"class": "small"}, "※ RMSE/MAE/R², 추론시간 등"), class_="billx-panel"),
                        ui.div(ui.h5("최종 모델 파라미터", class_="billx-panel-title"), ui.output_ui("apx_model_params"), class_="billx-panel"),
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
                        ui.div(ui.h5("평가 지표", class_="billx-panel-title"), ui.output_ui("apx_metrics_table"), ui.hr({"class": "soft"}), ui.output_ui("apx_residual_plot"), class_="billx-panel"),
                        ui.div(ui.h5("설명가능성 (XAI)", class_="billx-panel-title"), ui.output_ui("apx_shap_summary"), ui.hr({"class": "soft"}), ui.output_ui("apx_shap_bar"), class_="billx-panel"),
                        col_widths=[6, 6],
                    ),
                    ui.div(ui.h5("배포/모니터링 체크리스트", class_="billx-panel-title"), ui.output_ui("apx_checklist"), class_="billx-panel"),
                ),
                id="apx_tabs",
            ),
            class_="apx-scope",  # <<======= CSS scope wrapper
        ),
    )


def appendix_server(input, output, session):
    # ---- 개요
    @output
    @render.ui
    def apx_schema_table():
        return render_data_schema()

    @output
    @render.ui
    def apx_head_table():
        return render_data_head(report_df, n=10)

    # ---- EDA
    @output
    @render.ui
    def apx_calendar_alignment():
        return render_calendar_alignment_storyline(report_df)

    @output
    @render.ui
    def apx_calendar_overlay():
        year = int(input.cal_year() or 2018)
        mark = set(input.cal_mark() or [])
        return render_calendar_overlay(
            report_df,
            year,
            highlight_weekend=("weekend" in mark),
            highlight_holiday=("holiday" in mark),
        )

    @output
    @render.ui
    def apx_midnight_rollover():
        return render_midnight_rollover_fix(report_df)

    @output
    @render.ui
    def apx_eda_storyline():
        return render_eda_storyline_panels(report_df)

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

    # ---- 파생 피처 근거
    @output
    @render.ui
    def apx_lag_acf():
        return render_lag_window_acf(report_df)

    @output
    @render.ui
    def apx_how_stability():
        return render_how_profile_stability(report_df)

    @output
    @render.ui
    def apx_holiday_peak():
        return render_holiday_peak_checks(report_df)

    @output
    @render.ui
    def apx_unitprice_log():
        return render_unitprice_and_logtarget(report_df)

    # ---- 전처리
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

    # ---- 모델링
    @output
    @render.ui
    def apx_leaderboard():
        return render_leaderboard()

    @output
    @render.ui
    def apx_model_params():
        return render_model_params()

    @output
    @render.ui
    def apx_train_curve():
        return render_train_curve()

    @output
    @render.ui
    def apx_val_curve():
        return render_val_curve()

    # ---- 결과/검증
    @output
    @render.ui
    def apx_metrics_table():
        return render_metrics_table()

    @output
    @render.ui
    def apx_residual_plot():
        return render_residual_plot()

    @output
    @render.ui
    def apx_shap_summary():
        return render_shap_summary()

    @output
    @render.ui
    def apx_shap_bar():
        return render_shap_bar()

    @output
    @render.ui
    def apx_checklist():
        return render_deploy_checklist()
