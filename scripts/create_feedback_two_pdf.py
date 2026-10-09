"""Create the ELEC5305 Project Feedback Two submission PDF."""

import hashlib
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfdoc, pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# Older Windows/Anaconda hashlib builds do not accept ReportLab's
# ``usedforsecurity`` keyword. Keep the compatibility shim local to ReportLab.
_SYSTEM_MD5 = hashlib.md5
pdfdoc.md5 = lambda data=b"", **_kwargs: _SYSTEM_MD5(data)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "ELEC5305_Project_Feedback_Two_550275044.pdf"
SITE = "https://decarre.github.io/elec5305-project-550275044/"
REPOSITORY = "https://github.com/Decarre/elec5305-project-550275044"


def register_fonts():
    """Use the same Calibri family as the submitted Feedback One PDF."""
    font_dir = Path(r"C:\Windows\Fonts")
    pdfmetrics.registerFont(TTFont("Calibri", str(font_dir / "calibri.ttf")))
    pdfmetrics.registerFont(TTFont("Calibri-Bold", str(font_dir / "calibrib.ttf")))
    pdfmetrics.registerFont(TTFont("Calibri-Italic", str(font_dir / "calibrii.ttf")))
    pdfmetrics.registerFontFamily(
        "Calibri",
        normal="Calibri",
        bold="Calibri-Bold",
        italic="Calibri-Italic",
        boldItalic="Calibri-Bold",
    )


def build_pdf(output_path=OUTPUT):
    register_fonts()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    blue = colors.HexColor("#2E75B6")
    text = colors.HexColor("#111111")
    grid = colors.HexColor("#B7C9D8")
    light_blue = colors.HexColor("#EAF2F8")

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName="Calibri-Bold",
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceAfter=12 * mm,
    )
    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontName="Calibri-Bold",
        fontSize=14,
        leading=17,
        textColor=blue,
        spaceBefore=5.5 * mm,
        spaceAfter=3.5 * mm,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Calibri",
        fontSize=10.5,
        leading=15.5,
        alignment=TA_JUSTIFY,
        textColor=text,
        spaceAfter=3.2 * mm,
    )
    info = ParagraphStyle(
        "Info",
        parent=body,
        leftIndent=5 * mm,
        alignment=TA_JUSTIFY,
        spaceAfter=1.6 * mm,
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=body,
        leftIndent=6 * mm,
        firstLineIndent=-4 * mm,
        bulletIndent=0,
        alignment=TA_LEFT,
        spaceAfter=2.2 * mm,
    )
    note = ParagraphStyle(
        "Note",
        parent=body,
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#333333"),
    )

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Calibri", 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawRightString(185 * mm, 14 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=25 * mm,
        leftMargin=25 * mm,
        topMargin=24 * mm,
        bottomMargin=23 * mm,
        title="ELEC5305 Project Feedback Two - Ryan Hu",
        author="Ryan Hu",
        subject="Progress update and GitHub Project Site link",
    )

    story = [
        Paragraph("ELEC5305 Project Feedback Two", title),
        Paragraph("Project Title", heading),
        Paragraph(
            "<b>Attention-Based Temporal Localisation of Musical Instruments in Polyphonic Music</b>",
            body,
        ),
        Paragraph("Student Information", heading),
        Paragraph("<b>Full Name:</b> Ryan Hu", info),
        Paragraph("<b>Student ID (SID):</b> 550275044", info),
        Paragraph(
            f'<b>GitHub Project Site:</b> <link href="{SITE}" color="#0563C1">{SITE}</link>',
            info,
        ),
        Paragraph(
            f'<b>GitHub Repository:</b> <link href="{REPOSITORY}" color="#0563C1">{REPOSITORY}</link>',
            info,
        ),
        Paragraph("Project Progress", heading),
        Paragraph(
            "This project studies whether a weakly supervised audio model can recognise instruments in "
            "polyphonic music and show when each instrument is active. The model learns from instrument "
            "labels for a whole audio clip, without using frame-level timing labels during training. I compare "
            "mean pooling with instrument-specific attention while keeping the audio features and frame encoder "
            "the same. The main question is whether attention can provide useful temporal information as well as "
            "clip-level instrument predictions.",
            body,
        ),
        Paragraph("Achievements to Date", heading),
        Paragraph(
            "<b>Implementation:</b> I completed log-mel feature extraction, matched mean, max and "
            "AttentionMIC-style models, temporal post-processing, and a command-line workflow for validation, "
            "training and evaluation.",
            bullet,
            bulletText="-",
        ),
        Paragraph(
            "<b>Fair evaluation:</b> The pipeline supports artist-separated data splits, stable track and clip "
            "lists, validation-only threshold selection, saved model checkpoints, and one final test evaluation "
            "with fixed thresholds.",
            bullet,
            bulletText="-",
        ),
        Paragraph(
            "<b>Dataset audit:</b> The official MedleyDB v1/v2 metadata contains 196 tracks from 116 artists. "
            "In total, 117 tracks have matching time-based activation files. The two public sample songs also "
            "passed the audio and annotation checks.",
            bullet,
            bulletText="-",
        ),
        Paragraph(
            "<b>Verification:</b> All 13 automated tests pass. They check model pooling, metrics, "
            "artist-separated data lists, saved checkpoints and temporal processing.",
            bullet,
            bulletText="-",
        ),
        PageBreak(),
        Paragraph("Preliminary Engineering Result", heading),
    ]

    table_data = [
        ["Model", "Initial BCE", "Final BCE", "Micro-F1", "Macro-F1"],
        ["Mean pooling", "0.4715", "0.0357", "0.8721", "0.7717"],
        ["Attention pooling", "0.4509", "0.0135", "0.9977", "0.9967"],
    ]
    result_table = Table(table_data, colWidths=[43 * mm, 27 * mm, 27 * mm, 27 * mm, 27 * mm])
    result_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), blue),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Calibri-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Calibri-Bold"),
                ("FONTNAME", (1, 1), (-1, -1), "Calibri"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("BACKGROUND", (0, 1), (-1, -1), light_blue),
                ("GRID", (0, 0), (-1, -1), 0.5, grid),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5 * mm),
            ]
        )
    )
    story.extend(
        [
            result_table,
            Spacer(1, 4 * mm),
            Paragraph(
                "These values are engineering checks from 230 non-overlapping two-second clips taken from two "
                "public sample songs. The same clips were used for training and measurement. Therefore, the "
                "results confirm that the pipeline works, but they do not show performance on unseen music or "
                "prove that attention is better than mean pooling.",
                note,
            ),
            Paragraph("Current Limitation and Next Step", heading),
            Paragraph(
                "Access to the full MedleyDB audio has been requested and is still waiting for approval. If it is "
                "approved, complete artists will be separated into training, validation and test sets before the "
                "audio is divided into clips. Thresholds and temporal processing settings will be selected only "
                "on the validation set. The final model will then be tested once on unseen artists.",
                body,
            ),
            Paragraph(
                "If MedleyDB access is delayed, the duplicate-free Slakh2100-redux split will be used for the main "
                "quantitative experiment. MedleyDB can then be used later as an external real-audio check. This "
                "backup plan keeps the same research question and prevents the dataset approval process from "
                "stopping the project.",
                body,
            ),
            Paragraph("Project Resources", heading),
            Paragraph(
                f'<b>Repository:</b> <link href="{REPOSITORY}" color="#0563C1">{REPOSITORY}</link><br/>'
                f'<b>Project Site:</b> <link href="{SITE}" color="#0563C1">{SITE}</link>',
                body,
            ),
            Paragraph("AI Use Statement", heading),
            Paragraph(
                "OpenAI Codex was used to help plan the workflow, draft and revise Python code, debug the local "
                "environment, prepare tests, and improve the English in this progress report. The author checked "
                "the code, test outputs, links and reported numbers against the repository files. The author made "
                "the final project decisions and remains responsible for the submitted work. AI was not used to "
                "create experimental data.",
                body,
            ),
        ]
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output_path


if __name__ == "__main__":
    print(build_pdf())
