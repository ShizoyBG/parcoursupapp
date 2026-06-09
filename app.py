import math
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st


BASE_URL = "https://data.enseignementsup-recherche.gouv.fr/api/explore/v2.1/catalog/datasets"

# 2021-2024 use suffixed datasets, 2025 uses the base dataset name without suffix.
DATASET_BY_YEAR = {
    2021: "fr-esr-parcoursup_2021",
    2022: "fr-esr-parcoursup_2022",
    2023: "fr-esr-parcoursup_2023",
    2024: "fr-esr-parcoursup_2024",
    2025: "fr-esr-parcoursup",
}

YEARS = [2021, 2022, 2023, 2024, 2025]

# Fields to fetch for the main analysis.
FIELDS = [
    "session",
    "contrat_etab",
    "cod_uai",
    "g_ea_lib_vx",
    "dep_lib",
    "region_etab_aff",
    "acad_mies",
    "ville_etab",
    "lib_for_voe_ins",
    "select_form",
    "fili",
    "lib_comp_voe_ins",
    "form_lib_voe_acc",
    "fil_lib_voe_acc",
    "detail_forma",
    "capa_fin",
    "voe_tot",
    "voe_tot_f",
    "nb_voe_pp",
    "nb_voe_pc",
    "nb_cla_pp",
    "prop_tot",
    "acc_tot",
    "lib_grp1",
    "ran_grp1",
    "lib_grp2",
    "ran_grp2",
    "lib_grp3",
    "ran_grp3",
    "taux_acces_ens",
    "part_acces_gen",
    "part_acces_tec",
    "part_acces_pro",
    "lien_form_psup",
    "etablissement_id_paysage",
    "composante_id_paysage",
]

# Human-friendly column names for display.
RENAME_MAP = {
    "session": "Session",
    "contrat_etab": "Statut établissement",
    "cod_uai": "Code UAI",
    "g_ea_lib_vx": "Établissement",
    "dep_lib": "Département",
    "region_etab_aff": "Région",
    "acad_mies": "Académie",
    "ville_etab": "Ville",
    "lib_for_voe_ins": "Formation (libellé)",
    "select_form": "Sélectivité",
    "fili": "Filière agrégée",
    "lib_comp_voe_ins": "Formation détaillée",
    "form_lib_voe_acc": "Formation d'accueil",
    "fil_lib_voe_acc": "Formation d'accueil détaillée",
    "detail_forma": "Formation très détaillée",
    "capa_fin": "Capacité",
    "voe_tot": "Vœux totaux",
    "voe_tot_f": "Vœux totaux (filles)",
    "nb_voe_pp": "Vœux phase principale",
    "nb_voe_pc": "Vœux phase complémentaire",
    "nb_cla_pp": "Candidats classés (PP)",
    "prop_tot": "Propositions d'admission",
    "acc_tot": "Admis",
    "lib_grp1": "Groupe 1",
    "ran_grp1": "Dernier appelé groupe 1",
    "lib_grp2": "Groupe 2",
    "ran_grp2": "Dernier appelé groupe 2",
    "lib_grp3": "Groupe 3",
    "ran_grp3": "Dernier appelé groupe 3",
    "taux_acces_ens": "Taux d'accès",
    "part_acces_gen": "Part accès gén",
    "part_acces_tec": "Part accès techno",
    "part_acces_pro": "Part accès pro",
    "lien_form_psup": "Lien Parcoursup",
    "etablissement_id_paysage": "ID établissement (Paysage)",
    "composante_id_paysage": "ID composante (Paysage)",
}

# Optional synonyms to help people search by a few terms.
SEARCHABLE_TEXT_COLUMNS = [
    "g_ea_lib_vx",
    "lib_for_voe_ins",
    "lib_comp_voe_ins",
    "form_lib_voe_acc",
    "fil_lib_voe_acc",
    "detail_forma",
    "dep_lib",
    "region_etab_aff",
    "acad_mies",
    "ville_etab",
    "select_form",
    "contrat_etab",
]


@st.cache_data(show_spinner=False)
def fetch_records(year: int, where: str, select: Optional[str] = None, limit: int = 100) -> Dict:
    dataset = DATASET_BY_YEAR[year]
    url = f"{BASE_URL}/{dataset}/records"
    params = {"where": where, "limit": limit}
    if select:
        params["select"] = select

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False)
def fetch_all_records(year: int, where: str, select: Optional[str] = None, page_size: int = 100) -> pd.DataFrame:
    dataset = DATASET_BY_YEAR[year]
    url = f"{BASE_URL}/{dataset}/records"
    rows = []
    offset = 0

    while True:
        params = {"where": where, "limit": page_size, "offset": offset}
        if select:
            params["select"] = select

        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        batch = payload.get("results", [])
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_where(code: Optional[str] = None, text: Optional[str] = None, exact_name: Optional[str] = None) -> str:
    clauses: List[str] = []
    if code:
        code = re.sub(r"\D", "", str(code))
        if code:
            clauses.append(f"cod_aff_form={code}")
    if exact_name:
        exact_name = exact_name.replace('"', '\\"')
        clauses.append(f'g_ea_lib_vx="{exact_name}"')
    if text:
        text = text.strip().replace('"', '\\"')
        # Search across multiple useful text fields for user-friendly discovery.
        text_clause = " OR ".join([f"{col} ilike \"%{text}%\"" for col in SEARCHABLE_TEXT_COLUMNS])
        clauses.append(f"({text_clause})")

    if not clauses:
        return ""
    return " AND ".join(clauses)


def numeric_or_none(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def compute_year_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    for col in ["capa_fin", "voe_tot", "nb_voe_pp", "nb_voe_pc", "nb_cla_pp", "prop_tot", "acc_tot", "ran_grp1", "ran_grp2", "ran_grp3"]:
        if col in out.columns:
            out[col] = numeric_or_none(out[col])

    # Helpful derived metrics.
    out["Somme des rangs (1+2+3)"] = out[[c for c in ["ran_grp1", "ran_grp2", "ran_grp3"] if c in out.columns]].sum(axis=1, min_count=1)
    out["Moyenne des rangs (1+2+3)"] = out["Somme des rangs (1+2+3)"]

    # If there are multiple rows for the same year, keep a row-level view and also aggregate later.
    return out


def aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    agg = {
        "capa_fin": "mean",
        "voe_tot": "mean",
        "nb_voe_pp": "mean",
        "nb_voe_pc": "mean",
        "nb_cla_pp": "mean",
        "prop_tot": "mean",
        "acc_tot": "mean",
        "ran_grp1": "mean",
        "ran_grp2": "mean",
        "ran_grp3": "mean",
        "Somme des rangs (1+2+3)": "mean",
    }
    present = {k: v for k, v in agg.items() if k in df.columns}

    grouped = df.groupby("Année", as_index=False).agg(present)
    # Add nice derived ratios.
    grouped["Tension (voeux / places)"] = grouped["voe_tot"] / grouped["capa_fin"]
    grouped["Appels par place (propositions / places)"] = grouped["prop_tot"] / grouped["capa_fin"]
    grouped["Classement / appels"] = grouped["nb_cla_pp"] / grouped["prop_tot"]
    grouped["Part appelée parmi les voeux"] = grouped["prop_tot"] / grouped["voe_tot"]
    return grouped


def display_metrics_card(label: str, value: Optional[float], suffix: str = "", help_text: Optional[str] = None):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        st.metric(label, "—", help=help_text)
        return
    if isinstance(value, float):
        if value.is_integer():
            value_str = f"{int(value):,}".replace(",", " ")
        else:
            value_str = f"{value:,.1f}".replace(",", " ")
    else:
        value_str = f"{int(value):,}".replace(",", " ")
    st.metric(label, value_str + suffix, help=help_text)


st.set_page_config(page_title="Parcoursup Explorer", layout="wide")
st.title("Parcoursup Explorer")
st.caption("Recherche une formation, compare les 5 dernières années, et calcule des moyennes simples." )

with st.sidebar:
    st.header("Recherche")
    mode = st.radio("Mode", ["Code formation", "Nom / établissement"], index=0)
    years = st.multiselect("Années", YEARS, default=YEARS)
    st.divider()
    st.subheader("Filtres de données")
    only_first_match = st.checkbox("Ne garder que la première ligne trouvée par année", value=True)
    show_all_rows = st.checkbox("Afficher les lignes brutes", value=False)

    code_input = None
    text_input = None
    if mode == "Code formation":
        code_input = st.text_input("Code formation", value="4329")
    else:
        text_input = st.text_input("Recherche texte", value="CPBx")

    st.caption("Astuce: pour une recherche large, tape un établissement, une ville, ou une filière.")

query_where = build_where(code=code_input if mode == "Code formation" else None, text=text_input if mode == "Nom / établissement" else None)

if not years:
    st.info("Choisis au moins une année dans la barre latérale.")
    st.stop()

if not query_where:
    st.info("Saisis un code ou un texte de recherche.")
    st.stop()

# Search all selected years.
frames = []
search_errors = []
for year in years:
    try:
        df_year = fetch_all_records(year, query_where, select=",".join(FIELDS))
        if df_year.empty:
            continue
        df_year["Année"] = year
        df_year = compute_year_metrics(df_year)
        frames.append(df_year)
    except Exception as exc:
        search_errors.append(f"{year}: {exc}")

if search_errors:
    st.warning("Certaines années n'ont pas pu être chargées: " + " | ".join(search_errors))

if not frames:
    st.warning("Aucun résultat trouvé avec ce filtre.")
    st.stop()

raw = pd.concat(frames, ignore_index=True)

# Rename columns for display.
display_df = raw.rename(columns=RENAME_MAP)

# Prefer one row per year if requested.
if only_first_match:
    # Keep the first row per year after sorting by capacity descending then by formation label.
    sort_cols = [c for c in ["Année", "capa_fin", "voe_tot", "prop_tot"] if c in raw.columns]
    temp = raw.sort_values(by=sort_cols, ascending=[True] + [False] * (len(sort_cols) - 1)) if sort_cols else raw
    one_row = temp.groupby("Année", as_index=False).first()
    one_row_display = one_row.rename(columns=RENAME_MAP)
else:
    one_row = raw
    one_row_display = display_df

# Identify a single canonical row for summary labels.
canonical = one_row.iloc[0]
formation_label = canonical.get("g_ea_lib_vx") or canonical.get("lib_for_voe_ins") or "Formation"
formation_detail = canonical.get("lib_for_voe_ins") or canonical.get("lib_comp_voe_ins") or ""

st.subheader(f"{formation_label}")
if formation_detail and formation_detail != formation_label:
    st.write(formation_detail)

# Main metrics on canonical row (latest year if multiple).
canonical_latest = one_row.sort_values("Année").iloc[-1] if not one_row.empty else canonical

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    display_metrics_card("Capacité", canonical_latest.get("capa_fin"), help_text="Nombre de places annoncées.")
with m2:
    display_metrics_card("Vœux phase principale", canonical_latest.get("nb_voe_pp"), help_text="Demande en phase principale.")
with m3:
    display_metrics_card("Vœux totaux", canonical_latest.get("voe_tot"), help_text="Demande totale sur la campagne.")
with m4:
    display_metrics_card("Propositions", canonical_latest.get("prop_tot"), help_text="Candidats ayant reçu au moins une proposition.")
with m5:
    display_metrics_card("Dernier appelé G1", canonical_latest.get("ran_grp1"), help_text="Rang du dernier appelé du groupe 1.")
with m6:
    sum_ranks = None
    vals = [canonical_latest.get("ran_grp1"), canonical_latest.get("ran_grp2"), canonical_latest.get("ran_grp3")]
    vals = [v for v in vals if pd.notna(v)]
    if vals:
        sum_ranks = float(sum(vals))
    display_metrics_card("Somme des rangs", sum_ranks, help_text="ran_grp1 + ran_grp2 + ran_grp3, quand disponibles.")

st.divider()

# Summary across years.
summary = aggregate_summary(one_row.copy())
if summary.empty:
    st.warning("Impossible de calculer le résumé annuel.")
else:
    st.subheader("Moyennes sur les années sélectionnées")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        display_metrics_card("Capacité moyenne", summary["capa_fin"].mean())
    with c2:
        display_metrics_card("Dernier appelé G1 moyen", summary["ran_grp1"].mean())
    with c3:
        display_metrics_card("Somme des rangs moyenne", summary["Somme des rangs (1+2+3)"].mean())
    with c4:
        display_metrics_card("Tension moyenne", summary["Tension (voeux / places)"].mean())

    st.dataframe(
        summary[[c for c in [
            "Année",
            "capa_fin",
            "voe_tot",
            "nb_voe_pp",
            "nb_cla_pp",
            "prop_tot",
            "ran_grp1",
            "ran_grp2",
            "ran_grp3",
            "Somme des rangs (1+2+3)",
            "Tension (voeux / places)",
            "Appels par place (propositions / places)",
            "Part appelée parmi les voeux",
        ] if c in summary.columns]],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Comparaison par année")

    chart_df = summary.set_index("Année")[[c for c in ["capa_fin", "voe_tot", "nb_voe_pp", "prop_tot", "ran_grp1", "Somme des rangs (1+2+3)"] if c in summary.columns]]
    st.line_chart(chart_df)

st.divider()
st.subheader("Vue brute")
if show_all_rows:
    st.dataframe(display_df.rename(columns=RENAME_MAP), use_container_width=True, hide_index=True)
else:
    st.dataframe(one_row_display.rename(columns=RENAME_MAP), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Notes utiles")
st.markdown(
    """
- `voe_tot` = demande totale.
- `nb_voe_pp` = demande en phase principale.
- `capa_fin` = capacité / places.
- `prop_tot` = candidats ayant reçu au moins une proposition.
- `ran_grp1`, `ran_grp2`, `ran_grp3` = derniers appelés par groupe.
- La moyenne de `ran_grp1 + ran_grp2 + ran_grp3` est calculée comme une somme par ligne, puis moyennée sur les années sélectionnées.
- En 2025, le dataset est `fr-esr-parcoursup`. De 2020 à 2024, le format attendu est `fr-esr-parcoursup_YYYY`.
"""
)



