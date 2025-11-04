# =====================================================================
# viz/appendix_preproc.py  (Tab: 전처리)
# =====================================================================
from __future__ import annotations
from shiny import ui


def render_pipeline_accordion():
    return ui.accordion(
        ui.accordion_panel(
            "Step 0: 데이터 정제",
            ui.tags.ul(
                ui.tags.li(ui.tags.b("자정 롤오버 보정:"), " 00:00 → 다음날 00:00 교정"),
                ui.tags.li(ui.tags.b("이상 시점 제거:"), " 2018-11-07 00:00:00"),
                ui.tags.li(ui.tags.b("극단치 제거:"), " 전기요금 상위 0.7% 컷")
            )
        ),
        ui.accordion_panel(
            "Step 1: 기본 시간 변수",
            ui.tags.ul(
                ui.tags.li(ui.tags.b("시간 분해:"), " day/hour/minute/weekday/month"),
                ui.tags.li(ui.tags.b("주말 플래그:"), " is_weekend"),
                ui.tags.li(ui.tags.b("공휴일 플래그:"), " is_holiday/eve/after — 기준연도 사용"),
                ui.tags.li(ui.tags.b("시간대 구분:"), " is_peak_after, is_peak_even, is_night"),
                ui.tags.li(ui.tags.b("hour_of_week:"), " 0~167")
            )
        ),
        ui.accordion_panel(
            "Step 2: 주기성 인코딩",
            ui.tags.ul(
                ui.tags.li(ui.tags.b("1차/2차 고조파:"), " hour_sin/cos, hour_sin2/cos2"),
                ui.tags.li(ui.tags.b("요일 주기:"), " dow_sin/cos")
            )
        ),
        ui.accordion_panel(
            "Step 3: OOF 3변수 (5Fold TS)",
            ui.tags.ul(
                ui.tags.li("oof_kwh, oof_reactive, oof_pf"),
                ui.tags.li("미래 정보 차단 목적의 OOF 방식")
            )
        ),
        ui.accordion_panel(
            "Step 4: 시차(Lag)",
            ui.tags.ul(ui.tags.li("lag1, 1h, 6h, 24h, 48h, 7d"))
        ),
        ui.accordion_panel(
            "Step 5: 롤링/EMA",
            ui.tags.ul(ui.tags.li("roll6h, roll24h, ema24h — shift(1) 후 집계"))
        ),
        ui.accordion_panel(
            "Step 6: 차분",
            ui.tags.ul(ui.tags.li("samehour_d1, samehour_w1, samehour_w2, diff1h"))
        ),
        ui.accordion_panel(
            "Step 7: 비율/프록시",
            ui.tags.ul(ui.tags.li("oof_ratio, oof_pf_proxy, oof_ratio_ema24h"))
        ),
        ui.accordion_panel(
            "Step 8: 프로필/잔차·변화율",
            ui.tags.ul(ui.tags.li("how_profile_kwh, how_resid_kwh, kwh_rate_w1, kwh_rate_w2 — OOF 누적평균 기반"))
        ),
        ui.accordion_panel(
            "Step 9: 카테고리화",
            ui.tags.ul(ui.tags.li("how_cat(=hour_of_week 문자열), 작업유형"))
        ),
        id="pipeline_accordion",
        open=False
    )


def render_feature_summary():
    html = """
    <div class="p-3" style="font-size: 1rem;">
      <table class="table table-hover mt-0" style="font-size: 0.95rem;">
        <thead class="table-light">
          <tr><th>피처 그룹</th><th>개수</th><th>예시</th></tr>
        </thead>
        <tbody>
          <tr><td><b>시간 기본</b></td><td>~15</td><td>hour, weekday, is_weekend, is_holiday, hour_of_week</td></tr>
          <tr><td><b>주기성 인코딩</b></td><td>6</td><td>hour_sin/cos, hour_sin2/cos2, dow_sin/cos</td></tr>
          <tr><td><b>OOF 기본</b></td><td>3</td><td>oof_kwh, oof_reactive, oof_pf</td></tr>
          <tr><td><b>Lag</b></td><td>~18</td><td>각 OOF에 1h/6h/24h/48h/7d</td></tr>
          <tr><td><b>롤링/EMA</b></td><td>~9</td><td>roll6h/24h, ema24h × 3</td></tr>
          <tr><td><b>차분</b></td><td>~12</td><td>samehour_d1/w1/w2, diff1h × 3</td></tr>
          <tr><td><b>비율/프록시</b></td><td>3</td><td>oof_ratio, oof_pf_proxy, oof_ratio_ema24h</td></tr>
          <tr><td><b>프로필/잔차·변화율</b></td><td>4</td><td>how_profile_kwh, how_resid_kwh, kwh_rate_w1/w2</td></tr>
          <tr><td><b>카테고리</b></td><td>2</td><td>작업유형, how_cat</td></tr>
        </tbody>
      </table>
    </div>
    """
    return ui.HTML(html)


def render_scaling_info():
    html = """
    <div class="p-4" style="font-size: 1rem;">
      <div class="alert alert-success">
        <h6 class="mb-2">스케일링 미적용</h6>
        <p class="mb-0">트리 기반 모델에 한해 스케일 영향 미미</p>
      </div>
      <h6 class="mt-3 mb-2">인코딩 전략</h6>
      <table class="table table-sm table-striped">
        <thead><tr><th>타입</th><th>처리</th><th>상세</th></tr></thead>
        <tbody>
          <tr><td>수치형</td><td>원본</td><td>값 그대로</td></tr>
          <tr><td>범주형(작업유형)</td><td>Categorical dtype</td><td>LGBM categorical_feature 사용</td></tr>
          <tr><td>범주형(how_cat)</td><td>String→Categorical</td><td>hour_of_week 168 클래스</td></tr>
          <tr><td>결측</td><td>Median</td><td>Train 기준</td></tr>
        </tbody>
      </table>
    </div>
    """
    return ui.HTML(html)


def render_leakage_check():
    html = """
    <div class="p-4" style="font-size: 1rem;">
      <h6 class="mb-2">데이터 누수 점검</h6>
      <ul>
        <li>TimeSeriesSplit(3-fold) — 시간 순서 보존</li>
        <li>OOF 3변수 — 미래 정보 차단</li>
        <li>lag/rolling — shift 후 집계</li>
        <li>프로필 — 누적평균의 shift(OoF) 방식</li>
        <li>스케일링/결측 — Train 통계로 transform</li>
      </ul>
      <div class="small-muted">주의: 공휴일은 기준연도 캘린더 사용, Test lag는 Train 말단 기준</div>
    </div>
    """
    return ui.HTML(html)
