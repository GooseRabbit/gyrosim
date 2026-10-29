"""Графики. Plotly - для приложения (импортируется лениво); matplotlib - запасной путь для PNG без kaleido."""
from __future__ import annotations

import io

import numpy as np

COLORS = ["#4cc9f0", "#f72585", "#b5e48c", "#ffd166", "#c77dff", "#ff9e6d", "#90e0ef", "#e5e5e5"]
ROW_TITLES = (
    "Угол платформы и IMU, град",
    "Угловая скорость, град/с",
    "Команда сервы (импульс), мкс",
    "Положение вала сервы, град",
    "Ошибка управления",
    "Внешнее воздействие (угол основания), град",
)


def build_main_figure(res):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    p = res.params
    s = res.s
    t = res.t
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.035, subplot_titles=ROW_TITLES)

    def line(row, name, y, color, unit, dash=None, width=2, visible=True):
        fig.add_trace(go.Scatter(x=t, y=y, name=name, mode="lines", legendgroup=f"r{row}",
                                 line=dict(color=color, width=width, dash=dash),
                                 hovertemplate=f"{name}: %{{y:.4g}} {unit}<extra></extra>",
                                 visible=True if visible else "legendonly"), row=row, col=1)

    line(1, "истинный угол платформы", s["theta_true"], COLORS[0], "град", width=2.5)
    line(1, "измеренный угол IMU (пакеты DMP)", s["theta_meas"], COLORS[1], "град", dash="dot")
    fig.add_hline(y=0.0, line_color="#888888", line_dash="dash", row=1, col=1)

    line(2, "истинная скорость платформы", s["omega_true"], COLORS[0], "град/с")
    line(2, "измеренная скорость (гироскоп)", s["omega_meas"], COLORS[1], "град/с", dash="dot")
    if p.ctrl_mode == "cascade":
        line(2, "заданная скорость внутреннего контура", s["omega_ref"], COLORS[3], "град/с", dash="dash")

    line(3, "целевой импульс (выход регулятора)", s["target_pulse"], COLORS[3], "мкс", dash="dash")
    line(3, "фактический импульс PWM", s["actual_pulse"], COLORS[0], "мкс", width=2.5)
    for yv, nm in ((p.center_pulse_us, "центр"), (p.pmin_sw_us, "мин"), (p.pmax_sw_us, "макс")):
        fig.add_hline(y=yv, line_color="#888888", line_dash="dash" if nm == "центр" else "dot", row=3, col=1,
                      annotation_text=f"{nm} {yv:g}", annotation_position="top left")

    line(4, "фактический угол вала", s["servo_angle"], COLORS[0], "град", width=2.5)
    line(4, "заданный угол вала (по фактическому импульсу)", s["cmd_angle"], COLORS[3], "град", dash="dash")
    line(4, "угол вала после кадра и задержки", s["servo_ref_angle"], COLORS[4], "град", dash="dot", visible=False)
    if p.en_backlash and p.backlash_load_deg > 0:
        line(4, "угол нагрузки после люфта", s["load_angle"], COLORS[5], "град", dash="dashdot")

    line(5, "ошибка угла: 0 - измеренный, град", s["e_theta"], COLORS[2], "град")
    if p.ctrl_mode == "cascade":
        line(5, "ошибка скорости, град/с", s["e_omega"], COLORS[5], "град/с", dash="dot")

    line(6, "угол основания", s["base_angle"], COLORS[7], "град")

    fig.update_layout(template="plotly_dark", height=1500, hovermode="x unified",
                      margin=dict(l=60, r=20, t=60, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0, font=dict(size=11)))
    for r in range(1, 7):
        fig.update_xaxes(showgrid=True, gridcolor="#2a2f3a", row=r, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#2a2f3a", zeroline=True, zerolinecolor="#444b5a", row=r, col=1)
    fig.update_xaxes(title_text="время, с", row=6, col=1)
    return fig


def build_compare_figure(results, labels):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    titles = ("Истинный угол платформы, град", "Фактический импульс PWM, мкс", "Фактический угол вала, град",
              "Скорость вала, град/с")
    keys = ("theta_true", "actual_pulse", "servo_angle", "servo_omega")
    units = ("град", "мкс", "град", "град/с")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=titles)
    for ci, (res, lab) in enumerate(zip(results, labels)):
        col = COLORS[ci % len(COLORS)]
        for row, (key, unit) in enumerate(zip(keys, units), start=1):
            fig.add_trace(go.Scatter(x=res.t, y=res.s[key], name=lab, mode="lines", legendgroup=lab,
                                     showlegend=(row == 1), line=dict(color=col, width=2),
                                     hovertemplate=f"{lab}: %{{y:.4g}} {unit}<extra></extra>"), row=row, col=1)
    fig.add_hline(y=0.0, line_color="#888888", line_dash="dash", row=1, col=1)
    fig.update_layout(template="plotly_dark", height=1000, hovermode="x unified", margin=dict(l=60, r=20, t=60, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0.0))
    fig.update_xaxes(title_text="время, с", row=4, col=1)
    for r in range(1, 5):
        fig.update_xaxes(showgrid=True, gridcolor="#2a2f3a", row=r, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#2a2f3a", row=r, col=1)
    return fig


def build_servo_test_figure(test):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    r = test["res"]
    t = r["t"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Импульс PWM, мкс", "Угол вала, град"))
    if "target" in r:
        fig.add_trace(go.Scatter(x=t, y=r["target"], name="целевой PWM", mode="lines", line=dict(color=COLORS[3], dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=r["pulse"], name="фактический PWM", mode="lines", line=dict(color=COLORS[0])), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=t, y=r["pulse"], name="целевой = фактический PWM", mode="lines", line=dict(color=COLORS[0])), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=r["cmd_angle"], name="заданный угол вала", mode="lines", line=dict(color=COLORS[3], dash="dash")), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=r["angle"], name="фактический угол вала", mode="lines", line=dict(color=COLORS[1], width=2.5)), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=520, hovermode="x unified", margin=dict(l=60, r=20, t=50, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.1, x=0.0))
    fig.update_xaxes(title_text="время, с", row=2, col=1)
    return fig


def build_identification_figure(comparisons, labels):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = len(comparisons)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=False, vertical_spacing=0.12,
                        subplot_titles=[f"{lab}: RMSE {c['rmse']:.3g} град, макс. {c['max_abs']:.3g} град"
                                        for c, lab in zip(comparisons, labels)])
    for i, c in enumerate(comparisons, start=1):
        fig.add_trace(go.Scatter(x=c["t"], y=c["measured"], name="измерение", mode="markers",
                                 marker=dict(color=COLORS[3], size=4), legendgroup="m", showlegend=(i == 1)), row=i, col=1)
        fig.add_trace(go.Scatter(x=c["t"], y=c["model"], name="модель", mode="lines",
                                 line=dict(color=COLORS[0], width=2.5), legendgroup="mod", showlegend=(i == 1)), row=i, col=1)
        fig.add_trace(go.Scatter(x=c["t"], y=c["error"], name="ошибка (модель - измерение)", mode="lines",
                                 line=dict(color=COLORS[1], dash="dot"), legendgroup="e", showlegend=(i == 1)), row=i, col=1)
        fig.update_xaxes(title_text="время, с", row=i, col=1)
        fig.update_yaxes(title_text="угол, град", row=i, col=1)
    fig.update_layout(template="plotly_dark", height=380 * n, margin=dict(l=60, r=20, t=60, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0.0))
    return fig


def figure_html_bytes(fig) -> bytes:
    """Автономный HTML (библиотека plotly.js встроена, интернет не нужен)."""
    return fig.to_html(full_html=True, include_plotlyjs=True).encode("utf-8")


def figure_png_bytes(fig) -> bytes:
    """PNG через kaleido; если kaleido недоступен, бросает RuntimeError."""
    try:
        return fig.to_image(format="png", scale=2)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Для PNG-экспорта Plotly нужен пакет kaleido ({exc})") from exc


# ----------------------------------------------------------------------------- matplotlib (запасной путь)
def matplotlib_png(res, title: str = "") -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s, t, p = res.s, res.t, res.params
    fig, axes = plt.subplots(6, 1, figsize=(11, 14), sharex=True)
    ax = axes[0]
    ax.plot(t, s["theta_true"], label="истинный угол платформы", color="#1b9aaa", lw=2)
    ax.plot(t, s["theta_meas"], label="измеренный IMU", color="#e0457b", ls=":")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("град")
    ax.set_title(title or "Угол платформы и IMU")
    ax = axes[1]
    ax.plot(t, s["omega_true"], label="истинная скорость", color="#1b9aaa")
    ax.plot(t, s["omega_meas"], label="измеренная", color="#e0457b", ls=":")
    if p.ctrl_mode == "cascade":
        ax.plot(t, s["omega_ref"], label="заданная", color="#f2a900", ls="--")
    ax.set_ylabel("град/с")
    ax = axes[2]
    ax.plot(t, s["target_pulse"], label="целевой импульс", color="#f2a900", ls="--")
    ax.plot(t, s["actual_pulse"], label="фактический импульс", color="#1b9aaa", lw=2)
    for yv in (p.center_pulse_us, p.pmin_sw_us, p.pmax_sw_us):
        ax.axhline(yv, color="gray", ls=":", lw=0.8)
    ax.set_ylabel("мкс")
    ax = axes[3]
    ax.plot(t, s["servo_angle"], label="фактический угол вала", color="#1b9aaa", lw=2)
    ax.plot(t, s["cmd_angle"], label="заданный угол вала", color="#f2a900", ls="--")
    ax.set_ylabel("град")
    ax = axes[4]
    ax.plot(t, s["e_theta"], label="ошибка угла", color="#4f9d2f")
    if p.ctrl_mode == "cascade":
        ax.plot(t, s["e_omega"], label="ошибка скорости (град/с)", color="#d9822b", ls=":")
    ax.set_ylabel("град")
    ax = axes[5]
    ax.plot(t, s["base_angle"], label="угол основания", color="#555555")
    ax.set_ylabel("град")
    ax.set_xlabel("время, с")
    for a in axes:
        a.grid(True, alpha=0.3)
        a.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


def matplotlib_compare_png(results, labels) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for ci, (res, lab) in enumerate(zip(results, labels)):
        col = COLORS[ci % len(COLORS)]
        for ax, key in zip(axes, ("theta_true", "actual_pulse", "servo_angle", "servo_omega")):
            ax.plot(res.t, res.s[key], label=lab, color=col)
    for ax, ttl in zip(axes, ("угол платформы, град", "импульс, мкс", "угол вала, град", "скорость вала, град/с")):
        ax.set_ylabel(ttl)
        ax.grid(True, alpha=0.3)
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("время, с")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()
