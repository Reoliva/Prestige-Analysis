
#!/usr/bin/env python3
"""
Versión tesis final: genera una selección curada de gráficos PNG a partir de
Reporte_Master_Prestigio_v3.xlsx, privilegiando los resultados más sólidos y
defendibles para la memoria.

Uso:
    python prestige_plots.py --input Reporte_Prestigio.xlsx --output_dir graficos_prestigio
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DPI = 300
FIGSIZE_HEATMAP = (11.5, 6.8)
FIGSIZE_SCATTER = (8.5, 6.2)
FIGSIZE_BAR = (9.5, 6.3)
FIGSIZE_RANKING = (10.5, 6.5)
FIGSIZE_CV = (9.2, 5.8)

plt.rcParams.update(
    {
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
)

LABEL_MAP = {
    "Prest_Abs_Acad_01": "Prestigio absoluto académico [0,1]",
    "Prest_Abs_Exp_01": "Prestigio absoluto experiencial [0,1]",
    "Prest_Abs_Peer_01": "Prestigio absoluto entre pares [0,1]",
    "Prest_Rel_Acad": "Prestigio relativo académico",
    "Prest_Rel_Exp": "Prestigio relativo experiencial",
    "Prest_Rel_Peer": "Prestigio relativo entre pares",
    "Gaze_While_Silent": "Mirada recibida mientras guarda silencio",
    "Gaze_While_Speaking": "Mirada recibida mientras habla",
    "Total_Attention_Received": "Atención total recibida",
    "Attention_Received_Share_Group": "Proporción de atención recibida en el grupo",
    "Total_Speaking_Time": "Tiempo total de habla",
    "SNA_Eigenvector": "Centralidad de eigenvector (SNA)",
    "First_Speaker_Count": "Veces que habló primero",
    "Cursó FA (0 No - 1 Si)": "Cursó FA",
    "Cursó ISF (0 No - 1 Si)": "Cursó ISF",
    "Gender_Clean": "Género",
    "Rol_Clean": "Rol",
}


def human_label(name: object) -> str:
    if name is None:
        return "NA"
    text = str(name)
    if text in LABEL_MAP:
        return LABEL_MAP[text]
    text = text.replace("num__", "").replace("cat__", "")
    text = text.replace("_", " ")
    return " ".join(text.split()).strip()


def p_to_text(p_value: float) -> str:
    if pd.isna(p_value):
        return "p = NA"
    if float(p_value) < 0.001:
        return "p < 0.001"
    return f"p = {float(p_value):.3f}"


def stars_from_p(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    p = float(p_value)
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def corr_strength_text(rho: float) -> str:
    a = abs(float(rho))
    if a >= 0.70:
        return "muy fuerte"
    if a >= 0.50:
        return "fuerte"
    if a >= 0.30:
        return "moderada"
    if a >= 0.10:
        return "débil"
    return "muy débil"


def r2_quality_text(r2: float) -> str:
    if pd.isna(r2):
        return "sin evaluación"
    if r2 >= 0.50:
        return "desempeño bueno"
    if r2 >= 0.25:
        return "desempeño moderado"
    if r2 >= 0.00:
        return "desempeño bajo"
    return "desempeño pobre"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_footer(fig: plt.Figure, input_name: str) -> None:
    fig.text(0.01, 0.01, f"Fuente: {input_name}", ha="left", va="bottom", fontsize=8, alpha=0.8)


def finalize_plot(fig: plt.Figure, path: Path, input_name: str) -> None:
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    add_footer(fig, input_name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def load_sheets(input_path: Path) -> dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(input_path)
    required = {
        "heatmap": "13_Heatmap_Long",
        "scatter": "14_Scatter_Pairs",
        "perm": "11_ML_Permutation",
        "formula": "08_Formula_Ranking",
        "formula_best": "07_Formula_Best",
        "cv": "10_ML_CV_Metrics",
        "corr_abs": "05_Corr_Abs",
        "corr_rel": "06_Corr_Rel",
    }
    data = {}
    for key, sheet in required.items():
        if sheet not in xls.sheet_names:
            raise ValueError(f"No se encontró la hoja requerida: {sheet}")
        data[key] = pd.read_excel(xls, sheet_name=sheet)
    return data


def build_heatmap(df_long: pd.DataFrame, family: str, outpath: Path, input_name: str) -> dict:
    df = df_long[df_long["family"] == family].copy()
    x_order = [c for c in ["Prest_Abs_Acad_01", "Prest_Abs_Exp_01", "Prest_Abs_Peer_01"] if c in df["x_var"].unique()]
    if family == "relative_group_zscore":
        x_order = [c for c in ["Prest_Rel_Acad", "Prest_Rel_Exp", "Prest_Rel_Peer"] if c in df["x_var"].unique()]
    y_order = [
        c for c in [
            "Gaze_While_Silent",
            "SNA_Eigenvector",
            "Total_Attention_Received",
            "Attention_Received_Share_Group",
            "Gaze_While_Speaking",
            "Total_Speaking_Time",
        ] if c in df["y_var"].unique()
    ]
    rho = df.pivot(index="x_var", columns="y_var", values="rho").reindex(index=x_order, columns=y_order)
    pvals = df.pivot(index="x_var", columns="y_var", values="p_value").reindex(index=x_order, columns=y_order)

    vmax = max(0.25, float(np.nanmax(np.abs(rho.values)))) if rho.size else 1.0
    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)
    im = ax.imshow(rho.values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(y_order)))
    ax.set_xticklabels([human_label(x) for x in y_order], rotation=28, ha="right")
    ax.set_yticks(range(len(x_order)))
    ax.set_yticklabels([human_label(x) for x in x_order])

    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            val = rho.iat[i, j]
            if pd.isna(val):
                label = "NA"
            else:
                label = f"{val:.2f}\n{stars_from_p(pvals.iat[i, j])}"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color="black")

    title = "Correlaciones entre prestigio absoluto y métricas de interacción"
    if family == "relative_group_zscore":
        title = "Correlaciones entre prestigio relativo y métricas de interacción"
    ax.set_title(title)
    ax.set_xlabel("Métrica conductual")
    ax.set_ylabel("Indicador de prestigio")
    cbar = fig.colorbar(im, ax=ax, shrink=0.95)
    cbar.set_label("Spearman rho")

    finalize_plot(fig, outpath, input_name)

    strongest = df.sort_values("abs_rho", ascending=False).iloc[0]
    interpretation = (
        f"Resume el bloque de {('prestigio absoluto' if family == 'absolute_scaled' else 'prestigio relativo')} y muestra que "
        f"la asociación más alta se da entre {human_label(strongest['x_var']).lower()} y "
        f"{human_label(strongest['y_var']).lower()} (rho = {strongest['rho']:.3f}, {p_to_text(strongest['p_value'])})."
    )
    return {
        "filename": outpath.name,
        "title": title,
        "type": "heatmap",
        "target": "global",
        "interpretation": interpretation,
    }


def build_scatter(
    scatter_df: pd.DataFrame,
    x_var: str,
    y_var: str,
    outpath: Path,
    input_name: str,
    title: str,
) -> dict:
    d = scatter_df[(scatter_df["x_var"] == x_var) & (scatter_df["y_var"] == y_var)].copy()
    if d.empty:
        raise ValueError(f"No hay datos para scatter {x_var} vs {y_var}")

    d = d.sort_values(by="x_value")
    x = d["x_value"].astype(float).values
    y = d["y_value"].astype(float).values
    rho = float(d["rho"].iloc[0])
    p_value = float(d["p_value"].iloc[0])
    n = int(d["ID"].nunique()) if "ID" in d.columns else len(d)

    fig, ax = plt.subplots(figsize=FIGSIZE_SCATTER)
    ax.scatter(x, y, alpha=0.8)
    if len(np.unique(x)) > 1:
        coef = np.polyfit(x, y, 1)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        yy = coef[0] * xx + coef[1]
        ax.plot(xx, yy, linewidth=2.0)

    ax.set_title(title)
    ax.set_xlabel(human_label(x_var))
    ax.set_ylabel(human_label(y_var))
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.text(
        0.02,
        0.98,
        f"n = {n}\nrho = {rho:.3f}\n{p_to_text(p_value)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85, ec="0.7"),
    )

    finalize_plot(fig, outpath, input_name)

    interpretation = (
        f"Muestra una asociación {corr_strength_text(rho)} y {'positiva' if rho >= 0 else 'negativa'} entre "
        f"{human_label(x_var).lower()} y {human_label(y_var).lower()} "
        f"(rho = {rho:.3f}, {p_to_text(p_value)})."
    )
    return {
        "filename": outpath.name,
        "title": title,
        "type": "scatter",
        "target": y_var,
        "interpretation": interpretation,
    }


def build_permutation_bar(
    perm_df: pd.DataFrame,
    target: str,
    outpath: Path,
    input_name: str,
    title: str,
    top_n: int = 8,
) -> dict:
    d = perm_df[perm_df["Target"] == target].copy()
    if d.empty:
        raise ValueError(f"No hay permutation importance para target: {target}")
    d = d.sort_values("Rank").head(top_n).sort_values("Permutation_Importance_Mean", ascending=True)

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    ax.barh([human_label(x) for x in d["Feature"]], d["Permutation_Importance_Mean"])
    ax.set_title(title)
    ax.set_xlabel("Importancia media por permutación")
    ax.set_ylabel("Variable")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)

    finalize_plot(fig, outpath, input_name)

    top_row = d.sort_values("Permutation_Importance_Mean", ascending=False).iloc[0]
    interpretation = (
        f"Destaca qué variables sostienen la predicción de {human_label(target).lower()}. "
        f"La variable con mayor aporte es {human_label(top_row['Feature']).lower()}."
    )
    return {
        "filename": outpath.name,
        "title": title,
        "type": "feature_importance_permutation",
        "target": target,
        "interpretation": interpretation,
    }


def build_formula_ranking(
    formula_df: pd.DataFrame,
    target: str,
    outpath: Path,
    input_name: str,
    title: str,
    top_n: int = 10,
) -> dict:
    d = formula_df[formula_df["Target"] == target].copy()
    if d.empty:
        raise ValueError(f"No hay ranking de fórmulas para target: {target}")
    d = d.sort_values("Spearman_rho", ascending=False).head(top_n).sort_values("Spearman_rho", ascending=True)

    labels = d["Formula_Text"].astype(str).tolist()
    values = d["Spearman_rho"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=FIGSIZE_RANKING)
    ax.barh(labels, values)
    ax.set_title(title)
    ax.set_xlabel("Spearman rho")
    ax.set_ylabel("Fórmula de prestigio absoluto")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)

    finalize_plot(fig, outpath, input_name)

    best = d.sort_values("Spearman_rho", ascending=False).iloc[0]
    interpretation = (
        f"Compara las mejores fórmulas para {human_label(target).lower()}. "
        f"La mejor combinación es {best['Formula_Text']} "
        f"(rho = {best['Spearman_rho']:.3f}, {p_to_text(best['p_value'])})."
    )
    return {
        "filename": outpath.name,
        "title": title,
        "type": "formula_ranking",
        "target": target,
        "interpretation": interpretation,
    }


def build_cv_r2(cv_df: pd.DataFrame, outpath: Path, input_name: str) -> dict:
    summary = (
        cv_df.groupby("Target", dropna=False)
        .agg(Mean_R2=("Mean_R2", "max"), Std_R2=("Std_R2", "max"))
        .reset_index()
        .sort_values("Mean_R2", ascending=True)
    )

    labels = [human_label(x) for x in summary["Target"]]
    fig, ax = plt.subplots(figsize=FIGSIZE_CV)
    ax.barh(
        labels,
        summary["Mean_R2"],
        xerr=summary["Std_R2"],
        capsize=4,
    )
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_title("Desempeño predictivo por target (R² promedio, GroupKFold)")
    ax.set_xlabel("R² promedio")
    ax.set_ylabel("Target")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)

    finalize_plot(fig, outpath, input_name)

    best = summary.sort_values("Mean_R2", ascending=False).iloc[0]
    worst = summary.sort_values("Mean_R2", ascending=True).iloc[0]
    interpretation = (
        f"Resume la capacidad de generalización del modelo. "
        f"El mejor target es {human_label(best['Target']).lower()} (R² = {best['Mean_R2']:.3f}; {r2_quality_text(best['Mean_R2'])}) "
        f"y el más débil es {human_label(worst['Target']).lower()} (R² = {worst['Mean_R2']:.3f}; {r2_quality_text(worst['Mean_R2'])})."
    )
    return {
        "filename": outpath.name,
        "title": "Desempeño predictivo por target (R² promedio, GroupKFold)",
        "type": "cv",
        "target": "global",
        "interpretation": interpretation,
    }


def write_manifest(records: list[dict], outdir: Path) -> Path:
    df = pd.DataFrame(records)
    df.insert(0, "figure_number", range(1, len(df) + 1))
    path = outdir / "00_manifest_tesis_final.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_caption_doc(records: list[dict], outdir: Path, input_name: str, cv_df: pd.DataFrame) -> Path:
    summary = (
        cv_df.groupby("Target", dropna=False)
        .agg(Mean_R2=("Mean_R2", "max"), Std_R2=("Std_R2", "max"))
        .reset_index()
        .sort_values("Mean_R2", ascending=False)
    )
    best = summary.iloc[0]
    worst = summary.iloc[-1]

    lines = []
    lines.append("# Selección final de gráficos para tesis")
    lines.append("")
    lines.append(f"Archivo base analizado: `{input_name}`")
    lines.append("")
    lines.append("## Criterio de selección")
    lines.append("")
    lines.append("- Se priorizaron resultados con mayor fuerza de asociación, mejor interpretabilidad y mejor desempeño predictivo.")
    lines.append("- Se dejaron fuera las visualizaciones centradas en **Tiempo total de habla** y **Mirada recibida mientras habla** como evidencia principal, porque su generalización fue baja o negativa.")
    lines.append("- Se privilegió **permutation importance** por ser más defendible que impurity para argumentar relevancia de variables.")
    lines.append("")
    lines.append("## Lectura global")
    lines.append("")
    lines.append(
        f"El patrón más consistente del análisis es que el componente **experiencial** del prestigio domina tanto en correlaciones como en modelos predictivos, especialmente para **{human_label(best['Target']).lower()}**. "
        f"En cambio, **{human_label(worst['Target']).lower()}** muestra un desempeño insuficiente para ser usado como resultado central."
    )
    lines.append("")
    lines.append("## Figuras seleccionadas y caption sugerido")
    lines.append("")

    for i, rec in enumerate(records, start=1):
        lines.append(f"### Figura {i}. {rec['title']}")
        lines.append("")
        lines.append(f"**Archivo:** `{rec['filename']}`")
        lines.append("")
        lines.append(f"**Caption sugerido:** {rec['interpretation']}")
        lines.append("")
        use_note = "Figura principal" if i in {1,2,3,4,5,7,9,10} else "Figura complementaria"
        lines.append(f"**Uso recomendado:** {use_note}.")
        lines.append("")

    lines.append("## Orden recomendado en la tesis")
    lines.append("")
    lines.append("1. Heatmap de correlaciones absolutas.")
    lines.append("2. Heatmap de correlaciones relativas.")
    lines.append("3. Scatter prestigio absoluto experiencial vs mirada en silencio.")
    lines.append("4. Scatter prestigio relativo experiencial vs mirada en silencio.")
    lines.append("5. Scatter prestigio relativo experiencial vs centralidad SNA.")
    lines.append("6. Scatter prestigio absoluto experiencial vs atención total.")
    lines.append("7. Importancia por permutación para mirada en silencio.")
    lines.append("8. Importancia por permutación para atención total.")
    lines.append("9. Ranking de fórmulas para mirada en silencio.")
    lines.append("10. R² promedio por target.")
    lines.append("")
    lines.append("## Nota metodológica sugerida")
    lines.append("")
    lines.append(
        "En la redacción conviene presentar las correlaciones y los modelos como evidencia convergente, no como prueba causal. "
        "También conviene enfatizar que la validación se realizó con partición por grupos (GroupKFold), lo que reduce el riesgo de sobreajuste por contexto grupal compartido."
    )

    path = outdir / "00_figuras_tesis_final.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_readme(outdir: Path, input_path: Path, created_files: list[Path], manifest_path: Path, caption_path: Path) -> Path:
    lines = [
        "Gráficos finales seleccionados para tesis",
        "",
        f"Input utilizado: {input_path.name}",
        f"Cantidad de gráficos: {sum(1 for p in created_files if p.suffix.lower() == '.png')}",
        "",
        "Este paquete incluye solo las figuras más defendibles para el cuerpo principal de la tesis.",
        "Además se adjunta un documento con captions sugeridos y orden recomendado.",
        "",
        f"Manifest: {manifest_path.name}",
        f"Documento de captions: {caption_path.name}",
    ]
    path = outdir / "00_README_tesis_final.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def zip_output_dir(output_dir: Path) -> Path:
    zip_base = output_dir.parent / output_dir.name
    return Path(shutil.make_archive(str(zip_base), "zip", root_dir=output_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Ruta al archivo Excel fuente.")
    parser.add_argument("--output_dir", required=True, help="Directorio de salida.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())

    data = load_sheets(input_path)
    created_files: list[Path] = []
    records: list[dict] = []

    selection = [
        ("heatmap_abs", "01_heatmap_correlaciones_absolutas.png"),
        ("heatmap_rel", "02_heatmap_correlaciones_relativas.png"),
        ("scatter", "03_scatter_prestigio_absoluto_experiencial_vs_mirada_silencio.png", "Prest_Abs_Exp_01", "Gaze_While_Silent", "Prestigio absoluto experiencial vs mirada recibida mientras guarda silencio"),
        ("scatter", "04_scatter_prestigio_relativo_experiencial_vs_mirada_silencio.png", "Prest_Rel_Exp", "Gaze_While_Silent", "Prestigio relativo experiencial vs mirada recibida mientras guarda silencio"),
        ("scatter", "05_scatter_prestigio_relativo_experiencial_vs_centralidad_sna.png", "Prest_Rel_Exp", "SNA_Eigenvector", "Prestigio relativo experiencial vs centralidad de eigenvector"),
        ("scatter", "06_scatter_prestigio_absoluto_experiencial_vs_atencion_total.png", "Prest_Abs_Exp_01", "Total_Attention_Received", "Prestigio absoluto experiencial vs atención total recibida"),
        ("perm", "07_importancia_permutacion_mirada_silencio.png", "Gaze_While_Silent", "Importancia de variables para mirada recibida mientras guarda silencio"),
        ("perm", "08_importancia_permutacion_atencion_total.png", "Total_Attention_Received", "Importancia de variables para atención total recibida"),
        ("formula", "09_ranking_formulas_mirada_silencio.png", "Gaze_While_Silent", "Ranking de fórmulas para mirada recibida mientras guarda silencio"),
        ("cv", "10_validacion_cruzada_r2_promedio.png"),
    ]

    rec = build_heatmap(data["heatmap"], "absolute_scaled", output_dir / selection[0][1], input_path.name)
    created_files.append(output_dir / selection[0][1]); records.append(rec)

    rec = build_heatmap(data["heatmap"], "relative_group_zscore", output_dir / selection[1][1], input_path.name)
    created_files.append(output_dir / selection[1][1]); records.append(rec)

    for item in selection[2:6]:
        _, filename, x_var, y_var, title = item
        rec = build_scatter(data["scatter"], x_var, y_var, output_dir / filename, input_path.name, title)
        created_files.append(output_dir / filename); records.append(rec)

    for item in selection[6:8]:
        _, filename, target, title = item
        rec = build_permutation_bar(data["perm"], target, output_dir / filename, input_path.name, title)
        created_files.append(output_dir / filename); records.append(rec)

    _, filename, target, title = selection[8]
    rec = build_formula_ranking(data["formula"], target, output_dir / filename, input_path.name, title)
    created_files.append(output_dir / filename); records.append(rec)

    rec = build_cv_r2(data["cv"], output_dir / selection[9][1], input_path.name)
    created_files.append(output_dir / selection[9][1]); records.append(rec)

    manifest_path = write_manifest(records, output_dir)
    caption_path = write_caption_doc(records, output_dir, input_path.name, data["cv"])
    readme_path = write_readme(output_dir, input_path, created_files, manifest_path, caption_path)
    zip_path = zip_output_dir(output_dir)

    print(f"Listo. Gráficos PNG: {len([p for p in created_files if p.suffix.lower() == '.png'])}")
    print(f"Salida: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Captions: {caption_path}")
    print(f"README: {readme_path}")
    print(f"ZIP: {zip_path}")


if __name__ == "__main__":
    main()
