"""Create the one-page ELEC5305 Project Feedback Two submission PDF."""

import hashlib
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase import pdfdoc


# Older Windows/Anaconda hashlib builds do not accept ReportLab's
# ``usedforsecurity`` keyword. Keep the compatibility shim local to ReportLab.
_SYSTEM_MD5 = hashlib.md5
pdfdoc.md5 = lambda data=b"", **_kwargs: _SYSTEM_MD5(data)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "ELEC5305_Project_Feedback_Two_550275044.pdf"
SITE = "https://decarre.github.io/elec5305-project-550275044/"
REPOSITORY = "https://github.com/Decarre/elec5305-project-550275044"


def build_pdf(output_path=OUTPUT):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    navy = colors.HexColor("#16324F")
    blue = colors.HexColor("#246B9E")
    pale_blue = colors.HexColor("#EAF3F8")
    pale_gray = colors.HexColor("#F4F6F8")
    text = colors.HexColor("#263238")
    muted = colors.HexColor("#53636F")

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=22,
        alignment=TA_CENTER,
        textColor=navy,
        spaceAfter=3 * mm,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=12,
        alignment=TA_CENTER,
        textColor=muted,
        spaceAfter=4 * mm,
    )
    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=13,
        textColor=navy,
        spaceBefore=2.2 * mm,
        spaceAfter=1.4 * mm,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.65,
        leading=11.2,
        textColor=text,
        spaceAfter=1.4 * mm,
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=body,
        leftIndent=4.2 * mm,
        firstLineIndent=-3.0 * mm,
        bulletIndent=0,
        spaceAfter=0.8 * mm,
    )
    callout = ParagraphStyle(
        "Callout",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=9.2,
        leading=12,
        textColor=blue,
        alignment=TA_CENTER,
        spaceAfter=0,
    )
    small = ParagraphStyle(
        "Small",
        parent=body,
        fontSize=7.5,
        leading=9.4,
        textColor=muted,
        spaceAfter=0,
    )

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D6DEE3"))
        canvas.line(18 * mm, 13 * mm, 192 * mm, 13 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(muted)
        canvas.drawString(18 * mm, 8.5 * mm, "ELEC5305 Project Feedback Two - prepared 5 October 2026")
        canvas.drawRightString(192 * mm, 8.5 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=15 * mm,
        bottomMargin=17 * mm,
        title="ELEC5305 Project Feedback Two - Ryan Hu",
        author="Ryan Hu",
        subject="Progress update and GitHub Project Site link",
    )

    story = [
        Paragraph("ELEC5305 Project Feedback Two", title),
        Paragraph(
            "Ryan Hu | SID 550275044 | Attention-Based Temporal Localisation of Musical Instruments",
            subtitle,
        ),
    ]

    site_box = Table(
        [[Paragraph(f'GitHub Project Site: <link href="{SITE}" color="#246B9E">{SITE}</link>', callout)]],
        colWidths=[174 * mm],
    )
    site_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), pale_blue),
                ("BOX", (0, 0), (-1, -1), 0.8, blue),
                ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
            ]
        )
    )
    story.extend([site_box, Spacer(1, 2.2 * mm)])

    story.extend(
        [
            Paragraph("Brief project description", heading),
            Paragraph(
                "This project investigates whether a weakly supervised multi-label audio model can identify "
                "instruments in polyphonic music and recover when each instrument is active. A shared frame "
                "encoder is used for a controlled comparison of global mean pooling and instrument-specific "
                "attention pooling. Models receive clip-level instrument labels during training; time-aligned "
                "MedleyDB activation confidence is reserved for temporal evaluation.",
                body,
            ),
            Paragraph("Achievements to date", heading),
            Paragraph(
                "<b>Implementation:</b> log-mel feature extraction; matched mean, max and AttentionMIC-style "
                "models; separate frame probabilities, attention weights and clip predictions; temporal "
                "post-processing; and a validate/train/evaluate command-line workflow.",
                bullet,
                bulletText="-",
            ),
            Paragraph(
                "<b>Leakage prevention:</b> deterministic class-aware artist splits, stable track/clip "
                "manifests created after splitting, validation-only threshold selection, checkpoint reload, "
                "and frozen-threshold test evaluation.",
                bullet,
                bulletText="-",
            ),
            Paragraph(
                "<b>Dataset audit:</b> official MedleyDB v1/v2 metadata contains 196 tracks from 116 artists; "
                "117 tracks have matching v2 activation-confidence files. Both public sample mixes and all "
                "eight expected instrument stems passed audio/annotation alignment checks.",
                bullet,
                bulletText="-",
            ),
            Paragraph(
                "<b>Verification:</b> 13 automated tests pass, covering model aggregation semantics, metrics, "
                "artist-disjoint manifests, checkpoint behaviour and temporal processing.",
                bullet,
                bulletText="-",
            ),
        ]
    )

    story.append(Paragraph("Preliminary engineering result", heading))
    table_data = [
        ["Model", "Initial BCE", "Final BCE", "Micro-F1", "Macro-F1"],
        ["Mean pooling", "0.4715", "0.0357", "0.8721", "0.7717"],
        ["Attention pooling", "0.4509", "0.0135", "0.9977", "0.9967"],
    ]
    result_table = Table(table_data, colWidths=[48 * mm, 28 * mm, 28 * mm, 28 * mm, 28 * mm])
    result_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), navy),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("BACKGROUND", (0, 1), (-1, -1), pale_gray),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C5CFD6")),
                ("TOPPADDING", (0, 0), (-1, -1), 1.8 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8 * mm),
            ]
        )
    )
    story.extend(
        [
            result_table,
            Spacer(1, 1.2 * mm),
            Paragraph(
                "These values are training-set diagnostics from 230 non-overlapping two-second clips from two "
                "public sample songs. Training and measurement used the same clips. They verify the two model "
                "paths end to end and do not estimate generalisation or establish model superiority.",
                small,
            ),
            Paragraph("Current limitation and next step", heading),
            Paragraph(
                "Access to the full MedleyDB audio has been requested and is awaiting approval. After access is "
                "granted, complete artists will be assigned to train, validation and test before clipping. "
                "Thresholds and temporal post-processing will be selected only on validation artists, then "
                "clip-level and frame-level metrics will be reported once on held-out artists.",
                body,
            ),
            KeepTogether(
                [
                    Paragraph("Project resources", heading),
                    Paragraph(
                        f'Repository: <link href="{REPOSITORY}" color="#246B9E">{REPOSITORY}</link><br/>'
                        f'Project Site: <link href="{SITE}" color="#246B9E">{SITE}</link>',
                        small,
                    ),
                ]
            ),
        ]
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output_path


if __name__ == "__main__":
    print(build_pdf())
