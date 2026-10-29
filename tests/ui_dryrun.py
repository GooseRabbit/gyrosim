"""Сухой прогон app.py на СТЕНДАХ-ЗАМЕНИТЕЛЯХ streamlit и plotly.

Это НЕ запуск настоящего интерфейса: проверяется только, что код приложения выполняется без исключений и обращается к
библиотекам допустимым образом (типы и диапазоны number_input, значения selectbox, однородные колонки DataFrame,
допустимые аргументы Plotly, сигнатуры старого и нового Streamlit). Отрисовка, стили и поведение браузера не проверяются.

Запуск:  python tests/ui_dryrun.py
"""
import importlib
import os
import sys
import types

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

# ----------------------------------------------------------------------------------- заменитель plotly
SCATTER_OK = {"x", "y", "name", "mode", "legendgroup", "line", "hovertemplate", "visible", "showlegend", "marker"}
HLINE_OK = {"y", "line_color", "line_dash", "row", "col", "annotation_text", "annotation_position", "line_width"}
LAYOUT_OK = {"template", "height", "width", "hovermode", "margin", "legend", "title", "autosize", "showlegend"}
XAXES_OK = {"showgrid", "gridcolor", "row", "col", "title_text", "zeroline", "zerolinecolor"}
DASH_OK = {None, "solid", "dot", "dash", "longdash", "dashdot", "longdashdot"}


class Scatter:
    def __init__(self, **kw):
        bad = set(kw) - SCATTER_OK
        assert not bad, f"недопустимые аргументы Scatter: {bad}"
        line = kw.get("line") or {}
        assert set(line) <= {"color", "width", "dash"}, line
        assert line.get("dash") in DASH_OK, line
        assert kw.get("visible", True) in (True, False, "legendonly")
        assert len(kw["x"]) == len(kw["y"]), "длины x и y различаются"
        self.kw = kw


class Figure:
    def __init__(self, rows=1):
        self.traces = []
        self.rows = rows

    def add_trace(self, tr, row=None, col=None):
        assert isinstance(tr, Scatter)
        if row is not None:
            assert 1 <= row <= self.rows and col == 1, (row, col)
        self.traces.append(tr)

    def add_hline(self, **kw):
        assert set(kw) <= HLINE_OK, set(kw) - HLINE_OK

    def update_layout(self, **kw):
        assert set(kw) <= LAYOUT_OK, set(kw) - LAYOUT_OK

    def update_xaxes(self, **kw):
        assert set(kw) <= XAXES_OK, set(kw) - XAXES_OK

    def update_yaxes(self, **kw):
        assert set(kw) <= XAXES_OK, set(kw) - XAXES_OK

    def to_html(self, full_html=True, include_plotlyjs=True):
        return "<html>fake</html>"

    def to_image(self, format="png", scale=1):
        raise ValueError("kaleido is not installed (stand-in)")


def install_fake_plotly():
    go = types.ModuleType("plotly.graph_objects")
    go.Scatter = Scatter
    go.Figure = Figure
    sub = types.ModuleType("plotly.subplots")

    def make_subplots(rows=1, cols=1, shared_xaxes=False, vertical_spacing=None, subplot_titles=None):
        assert cols == 1
        if subplot_titles is not None:
            assert len(subplot_titles) == rows, (len(subplot_titles), rows)
        return Figure(rows)

    sub.make_subplots = make_subplots
    root = types.ModuleType("plotly")
    root.graph_objects = go
    root.subplots = sub
    sys.modules.update({"plotly": root, "plotly.graph_objects": go, "plotly.subplots": sub})


# ----------------------------------------------------------------------------------- заменитель streamlit
class SessionState(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


class Ctx:
    def __init__(self, st):
        self._st = st

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(self._st, name)


class Upload:
    def __init__(self, name, data):
        self.name = name
        self._d = data

    def getvalue(self):
        return self._d


def check_arrow(df):
    for c in df.columns:
        if df[c].dtype == object:
            types_ = {type(v) for v in df[c].dropna()}
            assert len(types_) <= 1, f"колонка «{c}» смешивает типы {types_}: Arrow/Streamlit упадёт"


def make_streamlit(api: str, pressed=(), uploads=None):
    st = types.ModuleType("streamlit")
    st.session_state = SessionState()
    st.log = []
    st.pressed = set(pressed)
    st.uploads = uploads or {}
    st.sidebar = Ctx(st)
    seen_upload = {}

    def rec(kind, *a):
        st.log.append((kind,) + a)

    def number_input(label, min_value=None, max_value=None, value="min", step=None, format=None, key=None, help=None,
                     on_change=None, *, disabled=False, label_visibility="visible"):
        ts = {type(x) for x in (min_value, max_value, step) if x is not None}
        assert len(ts) <= 1, f"{label}: смешаны int/float в min/max/step: {ts}"
        if key in st.session_state:
            v = st.session_state[key]
            assert type(v) in ts or not ts, f"{label}: тип значения {type(v)} не совпал с {ts}"
        else:
            v = min_value if value == "min" else value
            if key is not None:
                st.session_state[key] = v
        if min_value is not None:
            assert v >= min_value, f"{label}: {v} < min {min_value}"
        if max_value is not None:
            assert v <= max_value, f"{label}: {v} > max {max_value}"
        rec("number_input", key)
        return v

    def selectbox(label, options, index=0, format_func=str, key=None, help=None, *, disabled=False):
        options = list(options)
        if key in st.session_state:
            v = st.session_state[key]
            assert v in options, f"{label}: значение {v!r} не из списка опций"
        else:
            v = options[index]
            if key is not None:
                st.session_state[key] = v
        for o in options:
            format_func(o)
        rec("selectbox", key)
        return v

    def checkbox(label, value=False, key=None, help=None, *, disabled=False):
        if key in st.session_state:
            assert isinstance(st.session_state[key], bool), f"{label}: не bool"
            return st.session_state[key]
        if key is not None:
            st.session_state[key] = value
        return value

    def text_input(label, value="", key=None, help=None):
        if key in st.session_state:
            return st.session_state[key]
        if key is not None:
            st.session_state[key] = value
        return value

    def _pressed(label, key):
        return any(p in (label, key) or (p in label) for p in st.pressed if p)

    if api == "old":
        def button(label, key=None, help=None, on_click=None, args=None, kwargs=None, *, type="secondary", disabled=False,
                   use_container_width=False):
            hit = _pressed(label, key)
            if hit and on_click:
                on_click()
            rec("button", label, hit)
            return hit

        def download_button(label, data, file_name=None, mime=None, key=None, help=None, on_click=None, *, type="secondary",
                            disabled=False, use_container_width=False):
            assert isinstance(data, (bytes, str)), f"{label}: data не bytes/str"
            rec("download", label, len(data))
            return False

        def dataframe(data=None, width=None, height=None, *, use_container_width=False):
            check_arrow(data)
            rec("dataframe", data.shape)

        def plotly_chart(figure_or_data, use_container_width=False, *, theme="streamlit", key=None):
            assert isinstance(figure_or_data, Figure)
            assert figure_or_data.traces, "график без линий"
            rec("plotly", key, len(figure_or_data.traces))
    else:
        def button(label, key=None, help=None, on_click=None, args=None, kwargs=None, *, type="secondary", disabled=False,
                   width="content"):
            assert width in ("content", "stretch") or isinstance(width, int)
            hit = _pressed(label, key)
            if hit and on_click:
                on_click()
            rec("button", label, hit)
            return hit

        def download_button(label, data, file_name=None, mime=None, key=None, help=None, on_click=None, *, type="secondary",
                            disabled=False, width="content"):
            assert isinstance(data, (bytes, str)), f"{label}: data не bytes/str"
            rec("download", label, len(data))
            return False

        def dataframe(data=None, *, width="stretch", height="auto"):
            assert width in ("stretch", "content") or isinstance(width, int)
            check_arrow(data)
            rec("dataframe", data.shape)

        def plotly_chart(figure_or_data, *, width="stretch", theme="streamlit", key=None):
            assert width in ("stretch", "content") or isinstance(width, int)
            assert isinstance(figure_or_data, Figure)
            assert figure_or_data.traces, "график без линий"
            rec("plotly", key, len(figure_or_data.traces))

    def file_uploader(label, type=None, accept_multiple_files=False, key=None, help=None, on_change=None):
        up = st.uploads.get(key)
        st.session_state[key] = up
        if up is not None and seen_upload.get(key) is not up:
            seen_upload[key] = up
            if on_change:
                on_change()
        return up

    def simple(kind):
        def f(*a, **kw):
            rec(kind, a[0] if a else None)
        return f

    def columns(spec, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        return [Ctx(st) for _ in range(n)]

    def tabs(labels):
        return [Ctx(st) for _ in labels]

    st.number_input, st.selectbox, st.checkbox, st.text_input = number_input, selectbox, checkbox, text_input
    st.button, st.download_button, st.dataframe, st.plotly_chart = button, download_button, dataframe, plotly_chart
    st.file_uploader = file_uploader
    for name in ("markdown", "caption", "info", "warning", "error", "success", "latex", "code", "write", "metric",
                 "set_page_config"):
        setattr(st, name, simple(name))
    st.columns, st.tabs = columns, tabs
    st.expander = lambda label, expanded=False: Ctx(st)
    st.spinner = lambda text="": Ctx(st)
    return st


def run_script(st):
    for m in [k for k in sys.modules if k == "app"]:
        del sys.modules[m]
    sys.modules["streamlit"] = st
    st.log.clear()
    importlib.import_module("app")        # выполняет main() при импорте
    return list(st.log)


def count(log, kind):
    return sum(1 for e in log if e[0] == kind)


def scenario(api):
    install_fake_plotly()
    out = []
    st = make_streamlit(api)
    log = run_script(st)
    out.append(("первый вход (пустой результат)", count(log, "number_input") > 40 and st.session_state["result"] is None))

    st.pressed = {"Запустить симуляцию"}
    log = run_script(st)
    res = st.session_state["result"]
    out.append(("запуск симуляции", res is not None and count(log, "plotly") >= 1))

    st.pressed = set()
    log = run_script(st)
    out.append(("повторное отображение результата", count(log, "plotly") >= 1 and count(log, "dataframe") >= 1))

    st.session_state["ctrl_mode"] = "cascade"
    st.session_state["scenario"] = "sine"
    st.pressed = {"Запустить симуляцию"}
    run_script(st)
    out.append(("каскад + синусоидальное возмущение", st.session_state["result"].params.ctrl_mode == "cascade"))

    st.session_state["ctrl_mode"] = "PD"
    st.session_state["scenario"] = "random"
    st.pressed = {"Перезапустить"}
    run_script(st)
    out.append(("PD + случайное, перезапуск", st.session_state["result"].params.ctrl_mode == "PD"))

    st.session_state["ctrl_mode"] = "P"
    st.session_state["scenario"] = "initial"
    st.pressed = {"Сравнить коэффициенты"}
    run_script(st)
    comp = st.session_state["compare"]
    out.append(("сравнение KP", comp is not None and len(comp["results"]) == 3))

    st.session_state["kp_list_text"] = "5; 11,11; 18"
    st.pressed = {"Сравнить коэффициенты (эта вкладка)"}
    run_script(st)
    out.append(("список KP с десятичной запятой", st.session_state["compare"]["kps"] == [5.0, 11.11, 18.0]))
    st.session_state["kp_list_text"] = "5, abc"
    run_script(st)
    out.append(("некорректный список KP не роняет приложение", st.session_state["error"] is not None))
    st.session_state["kp_list_text"] = "5, 11.11, 18"

    st.pressed = {"Запустить испытания сервопривода"}
    run_script(st)
    st.pressed = set()
    log = run_script(st)
    out.append(("испытания сервопривода", bool(st.session_state["servo_tests"]) and count(log, "plotly") >= 1))

    st.pressed = {"Загрузить демонстрационные данные"}
    run_script(st)
    st.pressed = {"Подобрать параметры"}
    run_script(st)
    fit = st.session_state["ident_fit"]
    out.append(("идентификация на демо-данных", fit is not None and fit["rmse_after"] < 0.3))
    st.pressed = {"Применить найденные параметры"}
    run_script(st)
    out.append(("применение подобранных параметров", st.session_state["auto_omega_nl"] is False))
    st.pressed = {"Запустить симуляцию"}
    run_script(st)
    out.append(("запуск после применения параметров", st.session_state["result"] is not None))

    st.pressed = {"Подготовить графики"}
    run_script(st)
    st.pressed = set()
    log = run_script(st)
    names = [e[1] for e in log if e[0] == "download"]
    out.append(("экспорт: CSV, JSON, HTML, PNG (запасной путь matplotlib)",
                any("CSV" in n for n in names) and any("HTML" in n for n in names) and any("PNG" in n for n in names)))

    from gimbal_sim import params as P
    new = P.defaults()
    new["kp_us_per_deg"] = 18.0
    new["ctrl_mode"] = "cascade"
    st.uploads["params_upload"] = Upload("p.json", P.to_json(new).encode("utf-8"))
    run_script(st)
    out.append(("загрузка параметров из JSON", st.session_state["kp_us_per_deg"] == 18.0 and st.session_state["ctrl_mode"] == "cascade"))
    st.uploads["params_upload"] = Upload("bad.json", b"{oops")
    run_script(st)
    out.append(("битый JSON сообщает об ошибке и не роняет", st.session_state["kp_us_per_deg"] == 18.0))
    st.uploads.clear()

    st.pressed = {"Сбросить настройки"}
    run_script(st)
    out.append(("сброс настроек", st.session_state["kp_us_per_deg"] == 11.11 and st.session_state["result"] is None))

    st.pressed = {"Запустить автопроверки"}
    run_script(st)
    rep = st.session_state["autotest_report"]
    out.append(("автопроверки из интерфейса", rep is not None and rep[1] == 0 and rep[2] == 0))
    return out, rep


if __name__ == "__main__":
    bad = 0
    for api in ("old", "new"):
        print(f"=== стенд Streamlit: сигнатуры {'старой' if api == 'old' else 'новой'} версии API ===")
        results, rep = scenario(api)
        for name, ok in results:
            print(f"  [{'OK ' if ok else 'FAIL'}] {name}")
            bad += 0 if ok else 1
        print(f"  встроенные автопроверки: выполнено {rep[0]}, провалено {rep[1]}, ошибок {rep[2]}")
    print("ИТОГ:", "все сценарии прошли" if bad == 0 else f"провалено сценариев: {bad}")
    sys.exit(1 if bad else 0)
