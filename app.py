from __future__ import annotations

import io
import hashlib
import html
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image as PdfImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from image_processing import process_sulfide_inclusions, process_talc_concentration
from ore_classifier import OrePrediction, load_resnet18_model, model_predict
from sam_predictor import SAMPrediction, load_sam_model, predict_sam


APP_TITLE = "ImageLab"
SUPPORTED_TYPES = ["jpg", "jpeg", "png", "bmp", "webp", "tiff", "tif"]


@dataclass
class ModelResult:
    name: str
    image: Image.Image
    details: dict[str, str]
    metrics: dict[str, float] = field(default_factory=dict)


def configure_page() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🔬", layout="wide")
    st.markdown(
        """
        <style>
        :root { color-scheme:light; --blue:#1769e0; --blue-dark:#0f4ea8; --sky:#eaf4ff;
                --green:#16845a; --green-soft:#eaf8f2; --ink:#152238; --muted:#50627a; }
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background:linear-gradient(180deg,#f7fbff 0,#ffffff 380px);
            color:var(--ink);
        }
        .block-container { max-width: 1320px; padding-top: 2rem; padding-bottom: 4rem; }
        .hero { padding: 1.65rem 1.8rem; border:1px solid #d8e9fb; border-radius:22px;
                background:linear-gradient(135deg,#fff 0%,#edf7ff 72%,#eafaf3 100%);
                box-shadow:0 12px 35px rgba(24,94,153,.08); margin-bottom:1.2rem; }
        .eyebrow { color:var(--green); font-size:.78rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
        .hero h1 { color:var(--blue-dark); font-size:2.2rem; margin:.25rem 0 .4rem; }
        .hero p { color:var(--muted); margin:0; max-width:760px; }
        .section-title { font-size:1.1rem; font-weight:750; color:#173d70; margin:.35rem 0 .7rem; }
        .info-card { background:#fff; border:1px solid #dfebf7; border-radius:16px; padding:1rem 1.1rem;
                     box-shadow:0 5px 18px rgba(36,93,140,.05); min-height:116px; }
        .info-label { color:#6f8094; font-size:.75rem; text-transform:uppercase; letter-spacing:.07em; }
        .info-value { color:#163b68; font-size:1.05rem; font-weight:700; margin-top:.25rem; word-break:break-word; }
        .summary-card { background:#fff; border:1px solid #cfe1f3; border-left:5px solid var(--green);
                        border-radius:16px; padding:1.1rem 1.2rem; min-height:150px;
                        box-shadow:0 7px 22px rgba(25,85,132,.07); }
        .summary-label { color:#435970; font-size:.86rem; font-weight:650; line-height:1.35; }
        .summary-value { color:var(--blue-dark); font-size:2rem; font-weight:800; line-height:1.15;
                         margin:.45rem 0 .65rem; letter-spacing:-.02em; }
        .summary-caption { color:#596e84; font-size:.82rem; line-height:1.45; }
        .classifier-card { background:linear-gradient(135deg,#eef6ff,#edf9f4); border:1px solid #c9dfef;
                           border-radius:16px; padding:1.15rem 1.25rem; margin-top:.5rem; }
        .classifier-class { color:var(--blue-dark); font-size:1.45rem; font-weight:800; margin:.25rem 0; }
        .classifier-meta { color:#50677d; font-size:.85rem; line-height:1.5; }
        div[data-testid="stFileUploader"] { background:#fff; color:var(--ink); border:1px dashed #91bee9; border-radius:18px; padding:.4rem 1rem; }
        div[data-testid="stFileUploader"] section { background:#f7fbff; border-color:#91bee9; }
        div[data-testid="stFileUploader"] small,
        div[data-testid="stFileUploader"] span,
        div[data-testid="stFileUploader"] p { color:var(--muted) !important; }
        div[data-testid="stFileUploader"] button { color:var(--blue-dark) !important; background:#fff !important; border-color:#82b7e8 !important; }
        div[data-testid="stImage"] img { border-radius:16px; border:1px solid #dce8f4; background:#f3f7fb; }
        .stButton > button[kind="primary"], div[data-testid="stDownloadButton"] button {
            background:linear-gradient(90deg,var(--blue),var(--green)); border:0; color:#fff !important;
            font-weight:700;
        }
        .stButton > button[kind="primary"] p, div[data-testid="stDownloadButton"] button p { color:#fff !important; }
        .stButton > button[kind="primary"]:hover, div[data-testid="stDownloadButton"] button:hover {
            box-shadow:0 6px 18px rgba(23,105,224,.22); filter:brightness(.96);
        }
        [data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"] p,
        [data-testid="stWidgetLabel"] p { color:var(--ink); }
        [data-testid="stCaptionContainer"] p { color:#596e84 !important; }
        [data-testid="stSpinner"] { color:var(--blue-dark) !important; }
        [data-testid="stAlert"] { color:var(--ink); }
        [data-testid="stMetric"] { background:#fff; border:1px solid #cfe1f3; border-radius:16px; padding:1rem; }
        [data-testid="stMetricLabel"] p { color:#435970 !important; }
        [data-testid="stMetricValue"] { color:var(--blue-dark) !important; }
        hr { border-color:#d7e5f2 !important; }
        .placeholder { height:260px; display:flex; align-items:center; justify-content:center; text-align:center;
                       border:1px dashed #9dbbd3; border-radius:16px; color:#5d758b; background:#f8fbfe; padding:2rem; }
        footer { visibility:hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode not in ("RGB", "L"):
        background = Image.new("RGB", image.size, "white")
        if image.mode == "RGBA":
            background.paste(image, mask=image.getchannel("A"))
        else:
            background.paste(image.convert("RGB"))
        image = background
    return image.convert("RGB")


def run_model_a(image: Image.Image) -> ModelResult:
    result = process_talc_concentration(image)
    return ModelResult(
        "Карта концентрации талька",
        result["image"],
        {
            "Результат": "Синим отмечена область высокой концентрации (класс 2)",
            "Доля талька на всём изображении": f'{result["talc_percentage_total"]:.2f}%',
        },
        {"talc_percentage_total": result["talc_percentage_total"]},
    )


def run_model_b(image: Image.Image) -> ModelResult:
    result = process_sulfide_inclusions(image)
    return ModelResult(
        "Включения внутри сульфида",
        result["image"],
        {
            "Результат": "Белым отмечены включения внутри области сульфида",
            "Доля включений относительно площади сульфида": f'{result["dark_inclusions_percentage"]:.2f}%',
        },
        {"dark_inclusions_percentage": result["dark_inclusions_percentage"]},
    )


def process_parallel(image: Image.Image) -> list[ModelResult]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_model_a, image.copy()), executor.submit(run_model_b, image.copy())]
        return [future.result() for future in futures]


@st.cache_resource(show_spinner=False)
def get_ore_model():
    return load_resnet18_model(device="cpu")


def classify_ore(image: Image.Image) -> OrePrediction:
    return model_predict(image, get_ore_model(), device="cpu")


@st.cache_resource(show_spinner=False)
def get_sam_model():
    return load_sam_model(device="cpu")


def run_sam_prediction(image: Image.Image) -> SAMPrediction:
    return predict_sam(image, get_sam_model(), device="cpu")


def handle_sam_request(image: Image.Image) -> None:
    """Callback выполняет долгий inference до основного rerun и не дублирует виджеты."""
    st.session_state.pop("sam_prediction", None)
    st.session_state.pop("sam_error", None)
    try:
        st.session_state.sam_prediction = run_sam_prediction(image)
    except Exception as error:
        st.session_state.sam_error = str(error)


def requires_expert_review(prediction: OrePrediction, inclusions_percentage: float) -> bool:
    return (
        inclusions_percentage > 35 and prediction.class_index == 0
    ) or (
        inclusions_percentage < 35 and prediction.class_index == 1
    )


def image_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def pdf_image(image: Image.Image, max_width: float = 162 * mm, max_height: float = 190 * mm) -> PdfImage:
    ratio = min(max_width / image.width, max_height / image.height)
    return PdfImage(
        io.BytesIO(image_bytes(image, "JPEG")),
        width=image.width * ratio,
        height=image.height * ratio,
    )


def register_pdf_font() -> str:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                pdfmetrics.registerFont(TTFont("ReportFont", str(path)))
                return "ReportFont"
            except Exception:
                pass
    return "Helvetica"


def build_pdf(
    filename: str,
    source: Image.Image,
    results: list[ModelResult],
    prediction: OrePrediction | None = None,
    review_warning: bool = False,
    sam_prediction: SAMPrediction | None = None,
) -> bytes:
    output = io.BytesIO()
    font = register_pdf_font()
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleRu", parent=styles["Title"], fontName=font, textColor=colors.HexColor("#1558AE"), alignment=TA_CENTER)
    heading = ParagraphStyle("HeadingRu", parent=styles["Heading2"], fontName=font, textColor=colors.HexColor("#1769E0"), spaceBefore=8)
    body = ParagraphStyle("BodyRu", parent=styles["BodyText"], fontName=font, textColor=colors.HexColor("#24374E"), alignment=TA_LEFT, leading=15)
    doc = SimpleDocTemplate(output, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm, title=f"Отчёт — {filename}")
    story = [
        Paragraph("Отчёт об обработке изображения", title), Spacer(1, 6*mm),
        Table(
            [[Paragraph("Дата и время", body), Paragraph(datetime.now().strftime("%d.%m.%Y %H:%M:%S"), body)],
             [Paragraph("Имя файла", body), Paragraph(filename.replace("&", "&amp;"), body)],
             [Paragraph("Размер", body), Paragraph(f"{source.width} × {source.height} px", body)]],
            colWidths=[42*mm, 120*mm],
            style=TableStyle([("BACKGROUND",(0,0),(0,-1),colors.HexColor("#EAF4FF")),("GRID",(0,0),(-1,-1),.5,colors.HexColor("#C9DCEC")),("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),7)]),
        ), Spacer(1, 7*mm), Paragraph("Исходное изображение", heading),
        pdf_image(source),
    ]
    for result in results:
        story.extend([PageBreak(), Paragraph(result.name, heading), Spacer(1, 3*mm),
                      pdf_image(result.image), Spacer(1, 6*mm)])
        rows = [[Paragraph(str(key), body), Paragraph(str(value), body)] for key, value in result.details.items()]
        story.append(Table(rows, colWidths=[48*mm,114*mm], style=TableStyle([("BACKGROUND",(0,0),(0,-1),colors.HexColor("#EAF8F2")),("GRID",(0,0),(-1,-1),.5,colors.HexColor("#C9DCEC")),("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),7)])))
    if prediction is not None:
        classification_rows = [
            [Paragraph("Итоговый класс", body), Paragraph(prediction.class_name, body)],
            [Paragraph("Индекс класса", body), Paragraph(str(prediction.class_index), body)],
            [Paragraph("Доля голосов за итоговый класс", body), Paragraph(f"{prediction.vote_percentage:.2f}%", body)],
            [Paragraph("Тайлы: рядовая руда", body), Paragraph(str(prediction.ordinary_tiles), body)],
            [Paragraph("Тайлы: труднообрабатываемая руда", body), Paragraph(str(prediction.difficult_tiles), body)],
        ]
        story.extend([
            PageBreak(),
            Paragraph("Классификация руды ResNet18", heading),
            Spacer(1, 4*mm),
            Table(classification_rows, colWidths=[82*mm,80*mm], style=TableStyle([("BACKGROUND",(0,0),(0,-1),colors.HexColor("#EAF4FF")),("GRID",(0,0),(-1,-1),.5,colors.HexColor("#C9DCEC")),("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),7)])),
        ])
        if review_warning:
            story.extend([
                Spacer(1, 6*mm),
                Paragraph("Внимание: результат классификации расходится с правилом по доле включений. Обратите внимание на маску включений и уточните результат у специалиста.", body),
            ])
    if sam_prediction is not None:
        sam_rows = [
            [Paragraph("Тип результата", body), Paragraph("Дополнительная сегментация талька моделью SAM", body)],
            [Paragraph("Доля сегментированной области", body), Paragraph(f"{sam_prediction.positive_percentage:.2f}%", body)],
            [Paragraph("Положительных пикселей", body), Paragraph(str(sam_prediction.positive_pixels), body)],
        ]
        story.extend([
            PageBreak(),
            Paragraph("Дополнительное предсказание SAM", heading),
            Spacer(1, 3*mm),
            pdf_image(sam_prediction.overlay),
            Spacer(1, 6*mm),
            Table(sam_rows, colWidths=[72*mm,90*mm], style=TableStyle([("BACKGROUND",(0,0),(0,-1),colors.HexColor("#EAF8F2")),("GRID",(0,0),(-1,-1),.5,colors.HexColor("#C9DCEC")),("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),7)])),
        ])
    doc.build(story)
    return output.getvalue()


def render_metadata(filename: str, image: Image.Image, size_bytes: int) -> None:
    values = [("Файл", filename), ("Разрешение", f"{image.width} × {image.height} px"), ("Формат", image.format or "Определён автоматически"), ("Размер", f"{size_bytes / 1024:.1f} КБ")]
    columns = st.columns(4)
    for column, (label, value) in zip(columns, values):
        column.markdown(f'<div class="info-card"><div class="info-label">{html.escape(label)}</div><div class="info-value">{html.escape(value)}</div></div>', unsafe_allow_html=True)


def render_summary_card(label: str, value: str, caption: str) -> None:
    st.markdown(
        '<div class="summary-card">'
        f'<div class="summary-label">{html.escape(label)}</div>'
        f'<div class="summary-value">{html.escape(value)}</div>'
        f'<div class="summary-caption">{html.escape(caption)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_classification(prediction: OrePrediction, automatic: bool) -> None:
    launch_mode = "Автоматический запуск: доля талька менее 10%" if automatic else "Принудительный запуск пользователем"
    st.markdown(
        '<div class="classifier-card">'
        '<div class="summary-label">Результат ResNet18</div>'
        f'<div class="classifier-class">{html.escape(prediction.class_name)}</div>'
        f'<div class="classifier-meta">Класс {prediction.class_index} · '
        f'{prediction.vote_percentage:.2f}% тайлов проголосовали за итоговый класс · '
        f'{prediction.total_tiles} тайлов обработано<br>{html.escape(launch_mode)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    configure_page()
    st.markdown('<div class="hero"><div class="eyebrow">Платформа компьютерного зрения</div><h1>ImageLab</h1><p>Загрузите изображение, запустите параллельную обработку и получите единый PDF-отчёт с результатами.</p></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Загрузите изображение", type=SUPPORTED_TYPES, help="JPG, PNG, BMP, WebP или TIFF")
    if uploaded is None:
        st.markdown('<div class="placeholder">Перетащите изображение в область загрузки выше.<br>Здесь появятся исходник и результаты двух моделей.</div>', unsafe_allow_html=True)
        return

    file_data = uploaded.getvalue()
    try:
        raw_image = Image.open(io.BytesIO(file_data))
        detected_format = raw_image.format
        source = normalize_image(raw_image)
        source.format = detected_format
    except Exception:
        st.error("Не удалось прочитать файл. Проверьте, что это корректное изображение поддерживаемого формата.")
        return

    file_id = hashlib.sha256(file_data).hexdigest()
    if st.session_state.get("active_file") != file_id:
        st.session_state.active_file = file_id
        st.session_state.pop("results", None)
        st.session_state.pop("ore_prediction", None)
        st.session_state.pop("classification_automatic", None)
        st.session_state.pop("classification_error", None)
        st.session_state.pop("sam_prediction", None)
        st.session_state.pop("sam_error", None)

    st.write("")
    st.markdown('<div class="section-title">Обработка</div>', unsafe_allow_html=True)
    st.write("Анализ концентрации талька и поиск включений в сульфиде запускаются одновременно.")
    if st.button("Запустить обработку", type="primary", use_container_width=True):
        try:
            with st.spinner("Выполняются два алгоритма обработки…"):
                processed_results = process_parallel(source)
                st.session_state.results = processed_results
            st.session_state.pop("ore_prediction", None)
            st.session_state.pop("classification_automatic", None)
            st.session_state.pop("classification_error", None)
            talc_percentage = processed_results[0].metrics["talc_percentage_total"]
            if talc_percentage < 10:
                try:
                    with st.spinner("Доля талька менее 10% — выполняется автоматическая классификация руды…"):
                        st.session_state.ore_prediction = classify_ore(source)
                        st.session_state.classification_automatic = True
                except Exception as classification_error:
                    st.session_state.classification_error = str(classification_error)
        except Exception as error:
            st.session_state.pop("results", None)
            st.error(f"Не удалось обработать изображение: {error}")

    results: list[ModelResult] | None = st.session_state.get("results")
    st.divider()
    result_columns = st.columns(3)
    with result_columns[0]:
        st.markdown('<div class="section-title">Исходное изображение</div>', unsafe_allow_html=True)
        st.image(source, use_container_width=True)
    if not results:
        titles = ["Карта концентрации талька", "Маска включений"]
        for title, column in zip(titles, result_columns[1:]):
            with column:
                st.markdown(f'<div class="section-title">{title}</div><div class="placeholder">Результат появится после запуска обработки</div>', unsafe_allow_html=True)
        return

    for column, result in zip(result_columns[1:], results):
        with column:
            st.markdown(f'<div class="section-title">{result.name}</div>', unsafe_allow_html=True)
            st.image(result.image, use_container_width=True)

    st.divider()
    st.markdown('<div class="section-title">Единая сводка по изображению</div>', unsafe_allow_html=True)
    render_metadata(uploaded.name, source, uploaded.size)
    st.write("")
    summary_columns = st.columns(2)
    with summary_columns[0]:
        render_summary_card(
            "Доля талька на всём изображении",
            results[0].details["Доля талька на всём изображении"],
            results[0].details["Результат"],
        )
    with summary_columns[1]:
        render_summary_card(
            "Доля включений в площади сульфида",
            results[1].details["Доля включений относительно площади сульфида"],
            results[1].details["Результат"],
        )

    st.write("")
    st.markdown('<div class="section-title">Классификация руды</div>', unsafe_allow_html=True)
    description_column, button_column = st.columns([2.4, 1])
    with description_column:
        st.write("ResNet18 предсказывает два класса: **0 — рядовая руда**, **1 — труднообрабатываемая руда**. При доле талька менее 10% модель запускается автоматически.")
    with button_column:
        force_classification = st.button("Классифицировать руду", type="primary", use_container_width=True)

    if force_classification:
        try:
            with st.spinner("ResNet18 классифицирует изображение…"):
                st.session_state.ore_prediction = classify_ore(source)
                st.session_state.classification_automatic = False
                st.session_state.pop("classification_error", None)
        except Exception as classification_error:
            st.session_state.classification_error = str(classification_error)

    prediction: OrePrediction | None = st.session_state.get("ore_prediction")
    classification_error = st.session_state.get("classification_error")
    if classification_error:
        st.error(f"Не удалось классифицировать руду: {classification_error}")

    inclusions_percentage = results[1].metrics["dark_inclusions_percentage"]
    review_warning = False
    if prediction is not None:
        render_classification(prediction, st.session_state.get("classification_automatic", False))
        review_warning = requires_expert_review(prediction, inclusions_percentage)
        if review_warning:
            st.warning(
                f"Обратите внимание на маску включений: ResNet18 определила класс «{prediction.class_name}», "
                f"а доля включений относительно площади сульфида составила {inclusions_percentage:.2f}%. "
                "Для уточнения результата требуется оценка специалиста."
            )
    elif results[0].metrics["talc_percentage_total"] >= 10:
        st.info("Автоматическая классификация не запускалась, поскольку доля талька составляет 10% или более. При необходимости используйте кнопку выше.")

    st.write("")
    st.markdown('<div class="section-title">Вспомогательная модель SAM</div>', unsafe_allow_html=True)
    sam_description_column, sam_button_column = st.columns([2.4, 1])
    with sam_description_column:
        st.write("Получить дополнительное предсказание от модели SAM: модель сформирует вспомогательную маску сегментации талька.")
    with sam_button_column:
        st.button(
            "Запустить SAM",
            type="primary",
            use_container_width=True,
            on_click=handle_sam_request,
            args=(source.copy(),),
            help="На CPU обработка может занять несколько минут.",
        )

    sam_prediction: SAMPrediction | None = st.session_state.get("sam_prediction")
    sam_error = st.session_state.get("sam_error")
    if sam_error:
        st.error(f"Не удалось получить предсказание SAM: {sam_error}")
    if sam_prediction is not None:
        st.markdown('<div class="section-title">Дополнительное предсказание SAM</div>', unsafe_allow_html=True)
        sam_image_column, sam_info_column = st.columns([1.35, 1])
        with sam_image_column:
            st.image(sam_prediction.overlay, caption="Синим отмечена область, выделенная SAM", use_container_width=True)
        with sam_info_column:
            render_summary_card(
                "Доля области, сегментированной SAM",
                f"{sam_prediction.positive_percentage:.2f}%",
                f"Положительных пикселей: {sam_prediction.positive_pixels:,} из {sam_prediction.total_pixels:,}".replace(",", " "),
            )

    st.write("")
    pdf = build_pdf(uploaded.name, source, results, prediction, review_warning, sam_prediction)
    st.download_button("Скачать PDF-отчёт", data=pdf, file_name=f"report_{Path(uploaded.name).stem}.pdf", mime="application/pdf", type="primary", use_container_width=True)


if __name__ == "__main__":
    main()
