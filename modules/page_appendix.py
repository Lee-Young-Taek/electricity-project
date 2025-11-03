# modules/page_appendix.py
from shiny import ui, render

# ─────────────────────────────────────────────
# Appendix UI
# ─────────────────────────────────────────────
def appendix_ui():
    return ui.page_fluid(
        # ui.tags.link(rel="stylesheet", href="styles.css"),

        ui.navset_card_pill(
            # ========= 개요 =========
            ui.nav_panel(
                "개요",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("프로젝트 개요"),
                        ui.tags.ul(
                            ui.tags.li("목표: 공장 전력 사용량/요금 분석 및 예측"),
                            ui.tags.li("데이터 기간: 2024-01 ~ 2024-11"),
                            ui.tags.li("주요 컬럼: 측정일시, 전력사용량(kWh), 전기요금(원), 작업유형 등"),
                        ),
                        ui.hr({"class": "soft"}),
                        ui.div({"class": "small-muted"}, "※ 상세 정의는 ‘EDA > 데이터 사전’에서 확인"),
                    ),
                    ui.card(
                        ui.card_header("데이터 스냅샷"),
                        ui.output_ui("apx_head_table"),
                        ui.hr({"class": "soft"}),
                        ui.div({"class": "small-muted"}, "상위 5~10행 미리보기"),
                    ),
                    col_widths=[6, 6],
                ),
                ui.card(
                    ui.card_header("데이터 사전 (Data Dictionary)"),
                    ui.output_ui("apx_schema_table"),
                ),
            ),

            # ========= EDA =========
            ui.nav_panel(
                "EDA",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("결측치/이상치 요약"),
                        ui.output_ui("apx_missing_summary"),
                        ui.hr({"class": "soft"}),
                        ui.output_ui("apx_outlier_summary"),
                    ),
                    ui.card(
                        ui.card_header("분포 · 상관"),
                        ui.layout_columns(
                            ui.div(
                                ui.div({"class": "placeholder"}, "분포(히스토그램/커널)"),
                                ui.output_ui("apx_dist_plot"),
                                col_width=6,
                            ),
                            ui.div(
                                ui.div({"class": "placeholder"}, "상관행렬(Heatmap)"),
                                ui.output_ui("apx_corr_heatmap"),
                                col_width=6,
                            ),
                        ),
                    ),
                    col_widths=[4, 8],
                ),
                ui.card(
                    ui.card_header("시계열 트렌드(요금/사용량)"),
                    ui.output_ui("apx_time_trend"),
                ),
            ),

            # ========= 전처리 =========
            ui.nav_panel(
                "전처리",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("파이프라인 개요"),
                        ui.output_ui("apx_pipeline_text"),
                        ui.hr({"class": "soft"}),
                        ui.tags.ul(
                            ui.tags.li("Datetime 파생: 월/주/요일/시간/분(15분)"),
                            ui.tags.li("공휴일/주말 플래그, 계절, 근무/비근무 시간대"),
                            ui.tags.li("라그·롤링·사이클릭 인코딩(hour_sin/cos 등)"),
                        ),
                    ),
                    ui.card(
                        ui.card_header("피처 목록/선정 근거"),
                        ui.output_ui("apx_feature_table"),
                    ),
                    col_widths=[5, 7],
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("스케일링/인코딩 설정"),
                        ui.output_ui("apx_scaling_table"),
                    ),
                    ui.card(
                        ui.card_header("데이터 누수(Leakage) 점검"),
                        ui.output_ui("apx_leakage_check"),
                    ),
                    col_widths=[6, 6],
                ),
            ),

            # ========= 모델링 =========
            ui.nav_panel(
                "모델링",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("실험 보드(Leaderboard)"),
                        ui.output_ui("apx_leaderboard"),
                        ui.hr({"class": "soft"}),
                        ui.div({"class": "small-muted"}, "※ RMSE/MAE/R², 추론시간 등"),
                    ),
                    ui.card(
                        ui.card_header("최종 모델 파라미터"),
                        ui.output_ui("apx_model_params"),
                    ),
                    col_widths=[7, 5],
                ),
                ui.card(
                    ui.card_header("학습/검증 곡선"),
                    ui.layout_columns(
                        ui.div(ui.output_ui("apx_train_curve"), col_width=6),
                        ui.div(ui.output_ui("apx_val_curve"),   col_width=6),
                    ),
                ),
            ),

            # ========= 결과/검증 =========
            ui.nav_panel(
                "결과/검증",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("평가 지표"),
                        ui.output_ui("apx_metrics_table"),
                        ui.hr({"class": "soft"}),
                        ui.output_ui("apx_residual_plot"),
                    ),
                    ui.card(
                        ui.card_header("설명가능성 (XAI)"),
                        ui.output_ui("apx_shap_summary"),
                        ui.hr({"class": "soft"}),
                        ui.output_ui("apx_shap_bar"),
                    ),
                    col_widths=[6, 6],
                ),
                ui.card(
                    ui.card_header("배포/모니터링 체크리스트"),
                    ui.output_ui("apx_checklist"),
                ),
            ),

            # ========= 로그/추적 =========
            ui.nav_panel(
                "로그",
                ui.card(
                    ui.card_header("실험 로그"),
                    ui.output_ui("apx_experiment_log"),
                    ui.hr({"class": "soft"}),
                    ui.div(
                        ui.input_action_button("apx_btn_export_log", "로그 CSV 내보내기", class_="btn btn-outline-primary"),
                    ),
                ),
            ),
            id="apx_tabs"
        ),
    )


# ─────────────────────────────────────────────
# Appendix Server (자리표시자 렌더)
# ─────────────────────────────────────────────
def appendix_server(input, output, session):
    # 아래는 전부 “틀”만. 이후 viz 모듈 연결 시 교체하면 됨.
    def _ph(text="여기에 표/그래프가 표시됩니다.", h=260):
        return ui.div(text, class_="placeholder", style=f"height:{h}px;")

    @output 
    @render.ui
    def apx_head_table():        return _ph("데이터 상위 미리보기", 220)
    @output 
    @render.ui
    def apx_schema_table():      return _ph("데이터 사전 (컬럼명/타입/설명)", 280)
    @output 
    @render.ui
    def apx_missing_summary():   return _ph("결측치 요약 테이블", 220)
    @output 
    @render.ui
    def apx_outlier_summary():   return _ph("이상치(범위/룰/스코어) 요약", 220)
    @output 
    @render.ui
    def apx_corr_heatmap():      return _ph("상관행렬 Heatmap", 360)
    @output 
    @render.ui
    def apx_dist_plot():         return _ph("핵심 수치형 변수 분포", 360)
    @output 
    @render.ui
    def apx_time_trend():        return _ph("시계열 트렌드(요금/사용량) 라인", 360)

    @output 
    @render.ui
    def apx_pipeline_text():     return _ph("전처리 파이프라인(의사코드/다이어그램)", 180)
    @output 
    @render.ui
    def apx_feature_table():     return _ph("피처 목록/중요도/선정 근거", 260)
    @output 
    @render.ui
    def apx_scaling_table():     return _ph("스케일링·인코딩 설정(표)", 220)
    @output 
    @render.ui
    def apx_leakage_check():     return _ph("데이터 누수 체크리스트", 220)

    @output 
    @render.ui
    def apx_leaderboard():       return _ph("모델 리더보드 (RMSE/MAE/R²/Latency)", 260)
    @output 
    @render.ui
    def apx_model_params():      return _ph("최종 모델 하이퍼파라미터", 220)
    @output 
    @render.ui
    def apx_train_curve():       return _ph("학습 곡선(Train)", 300)
    @output 
    @render.ui
    def apx_val_curve():         return _ph("검증 곡선(Validation)", 300)

    @output 
    @render.ui
    def apx_metrics_table():     return _ph("최종 평가 지표 표 (RMSE/MAE/R² 등)", 220)
    @output 
    @render.ui
    def apx_residual_plot():     return _ph("Residual/에러분포", 300)
    @output 
    @render.ui
    def apx_shap_summary():      return _ph("SHAP Summary Plot", 300)
    @output 
    @render.ui
    def apx_shap_bar():          return _ph("상위 피처 영향 (SHAP Bar)", 260)

    @output 
    @render.ui
    def apx_checklist():         return _ph("배포/모니터링 체크리스트 (알람/드리프트/재학습)", 260)

    @output 
    @render.ui
    def apx_experiment_log():    return _ph("실험 로그 표 (실험ID/모델/파라미터/지표/시간)", 320)
