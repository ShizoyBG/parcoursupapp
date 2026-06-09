from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Parcoursup Data", layout="wide")

BASE_URL = "https://data.enseignementsup-recherche.gouv.fr/api/explore/v2.1/catalog/datasets"
YEARS = [2021, 2022, 2023, 2024, 2025]
DEFAULT_YEARS = [2022, 2023, 2024, 2025]
LATEST_YEAR = 2025

DATASET_BY_YEAR = {
    2021: "fr-esr-parcoursup_2021",
    2022: "fr-esr-parcoursup_2022",
    2023: "fr-esr-parcoursup_2023",
    2024: "fr-esr-parcoursup_2024",
    2025: "fr-esr-parcoursup",
}

CORE_FIELDS = [
    "session",
    "cod_aff_form",
    "g_ea_lib_vx",
    "lib_for_voe_ins",
    "contrat_etab",
    "cod_uai",
    "dep_lib",
    "region_etab_aff",
    "acad_mies",
    "ville_etab",
    "select_form",
    "fili",
    "capa_fin",
    "voe_tot",
    "nb_voe_pp",
    "nb_cla_pp",
    "prop_tot",
    "acc_tot",
    "lib_grp1",
    "ran_grp1",
    "lib_grp2",
    "ran_grp2",
    "lib_grp3",
    "ran_grp3",
]

SELECT_VARIANTS: List[List[str]] = [
    CORE_FIELDS,
    [
        "session",
        "cod_aff_form",
        "g_ea_lib_vx",
        "lib_for_voe_ins",
        "contrat_etab",
        "cod_uai",
        "dep_lib",
        "region_etab_aff",
        "acad_mies",
        "ville_etab",
        "select_form",
        "fili",
        "capa_fin",
        "voe_tot",
        "nb_voe_pp",
        "nb_cla_pp",
        "prop_tot",
        "acc_tot",
        "ran_grp1",
    ],
    [
        "session",
        "cod_aff_form",
        "g_ea_lib_vx",
        "lib_for_voe_ins",
        "capa_fin",
        "voe_tot",
        "nb_voe_pp",
        "nb_cla_pp",
        "prop_tot",
        "ran_grp1",
    ],
]

FRIENDLY_NAMES = {
    "session": "Année",
    "cod_aff_form": "Référence formation",
    "g_ea_lib_vx": "Établissement",
    "lib_for_voe_ins": "Formation",
    "contrat_etab": "Statut",
    "cod_uai": "Code UAI",
    "dep_lib": "Département",
    "region_etab_aff": "Région",
    "acad_mies": "Académie",
    "ville_etab": "Ville",
    "select_form": "Sélectivité",
    "fili": "Filière",
    "capa_fin": "Places",
    "voe_tot": "Candidatures totales",
    "nb_voe_pp": "Candidatures phase principale",
    "nb_cla_pp": "Candidats classés",
    "prop_tot": "Propositions",
    "acc_tot": "Admis",
    "ran_grp1": "Dernier rang phase principale",
    "ran_grp2": "Dernier rang phase complémentaire",
    "ran_grp3": "Dernier rang phase 3",
    "somme_rangs": "Dernier rang final",
    "moyenne_rangs": "Moyenne des rangs",
    "dernier_rang_max": "Rang max",
    "tension": "Tension",
    "appels_par_place": "Appels/place",
    "part_appelee": "% appelés",
    "classement_par_appels": "Classés/appels",
}

CATEGORY_GROUPS = {
    "Places et demande": ["capa_fin", "voe_tot", "nb_voe_pp"],
    "Classement et appels": ["nb_cla_pp", "prop_tot", "acc_tot", "ran_grp1", "ran_grp2", "ran_grp3"],
    "Synthèse": ["somme_rangs", "moyenne_rangs", "dernier_rang_max"],
    "Ratios": ["tension", "appels_par_place", "part_appelee", "classement_par_appels"],
}


# ---------- API helpers ----------

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_all_records(year: int, where: str, select: Sequence[str]) -> pd.DataFrame:
    """Fetch all matching records for one year, with graceful fallbacks for older schemas."""
    dataset = DATASET_BY_YEAR[year]
    url = f"{BASE_URL}/{dataset}/records"

    select_variants: List[List[str]] = [list(select)]
    for variant in SELECT_VARIANTS:
        if variant not in select_variants:
            select_variants.append(variant)

    last_error: Optional[str] = None

    for fields in select_variants:
        try:
            rows: List[Dict] = []
            offset = 0
            page_size = 100
            select_param = ",".join(fields)

            while True:
                params = {"where": where, "limit": page_size, "offset": offset, "select": select_param}
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                payload = response.json()
                batch = payload.get("results", [])
                if not batch:
                    break
                rows.extend(batch)
                if len(batch) < page_size:
                    break
                offset += page_size

            if rows:
                return pd.DataFrame(rows)

            return pd.DataFrame()

        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 400:
                last_error = f"{year}: schéma incompatible."
                continue
            raise
        except requests.RequestException as exc:
            last_error = f"{year}: {exc}"
            continue

    if last_error:
        raise RuntimeError(last_error)
    return pd.DataFrame()


# ---------- Parsing / cleaning ----------

def extract_code(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip()

    for pat in [r"cod_aff_form[=:/?&\- ]+(\d+)", r"g_ea_cod[=:/?&\- ]+(\d+)"]:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            return m.group(1)

    try:
        parsed = urlparse(t)
        qs = parse_qs(parsed.query)
        for key in ["cod_aff_form", "g_ea_cod", "code"]:
            if key in qs and qs[key]:
                val = re.sub(r"\D", "", qs[key][0])
                if val:
                    return val
    except Exception:
        pass

    m = re.search(r"\b(\d{3,6})\b", t)
    return m.group(1) if m else None


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def format_int(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    try:
        if float(value).is_integer():
            return f"{int(value):,}".replace(",", " ")
        return f"{float(value):,.1f}".replace(",", " ")
    except Exception:
        return str(value)


# ---------- Business logic ----------

def enrich_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    for col in ["capa_fin", "voe_tot", "nb_voe_pp", "nb_cla_pp", "prop_tot", "acc_tot", "ran_grp1", "ran_grp2", "ran_grp3"]:
        if col in out.columns:
            out[col] = numeric(out[col])

    if "cod_aff_form" in out.columns:
        out = out[out["cod_aff_form"].notna()]

    out["somme_rangs"] = out[[c for c in ["ran_grp1", "ran_grp2", "ran_grp3"] if c in out.columns]].sum(axis=1, min_count=1)
    out["moyenne_rangs"] = out[[c for c in ["ran_grp1", "ran_grp2", "ran_grp3"] if c in out.columns]].mean(axis=1)
    out["dernier_rang_max"] = out[[c for c in ["ran_grp1", "ran_grp2", "ran_grp3"] if c in out.columns]].max(axis=1)
    out["tension"] = out["voe_tot"] / out["capa_fin"] if "voe_tot" in out.columns and "capa_fin" in out.columns else pd.NA
    out["appels_par_place"] = out["prop_tot"] / out["capa_fin"] if "prop_tot" in out.columns and "capa_fin" in out.columns else pd.NA
    out["part_appelee"] = out["prop_tot"] / out["voe_tot"] if "prop_tot" in out.columns and "voe_tot" in out.columns else pd.NA
    out["classement_par_appels"] = out["nb_cla_pp"] / out["prop_tot"] if "nb_cla_pp" in out.columns and "prop_tot" in out.columns else pd.NA
    return out


def pick_representative_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque année, garde seulement la ligne avec le plus de candidatures."""
    if df.empty or "Année" not in df.columns:
        return df
    
    # Pour chaque année, trouver l'index de la ligne avec le plus de voe_tot (candidatures)
    if "voe_tot" in df.columns:
        idx = df.groupby("Année")["voe_tot"].idxmax()
        return df.loc[idx].reset_index(drop=True)
    elif "prop_tot" in df.columns:
        idx = df.groupby("Année")["prop_tot"].idxmax()
        return df.loc[idx].reset_index(drop=True)
    else:
        # Si ni voe_tot ni prop_tot, on retourne la première ligne par année
        return df.groupby("Année", as_index=False).first().reset_index(drop=True)


def aggregate_by_year(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    agg_map = {
        "capa_fin": "mean",
        "voe_tot": "mean",
        "nb_voe_pp": "mean",
        "nb_cla_pp": "mean",
        "prop_tot": "mean",
        "acc_tot": "mean",
        "ran_grp1": "mean",
        "ran_grp2": "mean",
        "ran_grp3": "mean",
        "somme_rangs": "mean",
        "moyenne_rangs": "mean",
        "dernier_rang_max": "mean",
        "tension": "mean",
        "appels_par_place": "mean",
        "part_appelee": "mean",
        "classement_par_appels": "mean",
    }
    present = {k: v for k, v in agg_map.items() if k in df.columns}
    grouped = df.groupby("Année", as_index=False).agg(present)
    return grouped


def metric_card(label: str, value: Optional[float], help_text: Optional[str] = None) -> None:
    st.metric(label, format_int(value), help=help_text)


# ---------- Session state ----------

if "formation_url" not in st.session_state:
    st.session_state.formation_url = ""


# ---------- UI ----------

st.title("Parcoursup - Données")
st.caption("Colle l'URL de la fiche Parcoursup et tu vois les stats")

url_tab, data_tab = st.tabs(["URL", "Données"])

# Sidebar parameters
with st.sidebar:
    st.header("Paramètres")
    years = st.multiselect("Années", YEARS, default=DEFAULT_YEARS)
    show_raw = st.checkbox("Tableau brut")
    one_row_per_year = st.checkbox("Une ligne par année", value=True)

selected_metric_keys: List[str] = ["capa_fin", "voe_tot", "nb_voe_pp", "nb_cla_pp", "prop_tot", "acc_tot", "ran_grp1", "tension", "appels_par_place"]

# URL Tab
with url_tab:
    st.markdown("### Colle l'URL")
    st.write("Va sur parcoursup.fr, trouve la fiche, copie l'URL et colle-la ici.")
    
    with st.form("url_form"):
        raw_input = st.text_area(
            "URL",
            value=st.session_state.formation_url,
            placeholder="https://www.parcoursup.fr/...",
            height=100,
        )
        submitted = st.form_submit_button("Charger", use_container_width=True)

    if submitted:
        url = raw_input.strip()
        code = extract_code(url)
        if code:
            st.session_state.formation_url = url
            st.success("URL chargée ! Va dans l'onglet Données.")
        else:
            st.error("URL pas reconnue")

    st.write("")
    st.markdown("""
### Comment faire :
1. Va sur parcoursup.fr
2. Cherche une formation
3. Copie l'URL de la barre du navigateur
4. Colle-la ci-dessus
5. Clique sur Données
    """)


# Data Tab
with data_tab:
    formation_url = st.session_state.formation_url
    
    if not formation_url:
        st.info("D'abord colle une URL dans l'onglet URL.")
        st.stop()

    if not years:
        st.warning("Sélectionne au moins une année dans la barre latérale.")
        st.stop()

    code = extract_code(formation_url)
    if not code:
        st.error("Impossible de lire cette URL. Assure-toi que c'est un lien Parcoursup.")
        st.stop()

    query_where = f"cod_aff_form={code}"
    frames: List[pd.DataFrame] = []
    errors: List[str] = []

    for year in years:
        try:
            df_year = fetch_all_records(year, query_where, select=CORE_FIELDS)
            if df_year.empty:
                continue
            df_year = enrich_metrics(df_year)
            if "session" in df_year.columns:
                df_year = df_year.rename(columns={"session": "Année"})
            df_year["Année"] = year
            frames.append(df_year)
        except Exception as exc:
            errors.append(f"{year}: {exc}")

    if errors:
        st.warning(" | ".join(errors))

    if not frames:
        st.error("Pas de données trouvées pour cette formation.")
        st.stop()

    raw = pd.concat(frames, ignore_index=True)
    representative = pick_representative_rows(raw) if one_row_per_year else raw.copy()
    summary = aggregate_by_year(representative)

    # Header
    latest_row = representative.sort_values("Année").iloc[-1]
    establishment = latest_row.get("g_ea_lib_vx", "")
    formation = latest_row.get("lib_for_voe_ins", "")
    title_line = establishment if establishment else formation
    subtitle_line = formation if formation and formation != title_line else ""

    if title_line:
        st.markdown(f"## {title_line}")
    if subtitle_line:
        st.caption(subtitle_line)

    st.divider()

    # Key metrics cards
    st.subheader("Stats principales (dernière année)")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        metric_card("Places", latest_row.get("capa_fin"))
    with col2:
        metric_card("Candidatures", latest_row.get("voe_tot"))
    with col3:
        metric_card("Propositions", latest_row.get("prop_tot"))
    with col4:
        metric_card("Dernier rang phase principale", latest_row.get("ran_grp1"))
    with col5:
        metric_card("Tension", latest_row.get("tension"), "candidatures/places")

    st.divider()

    # Average stats
    st.subheader("Moyennes sur les années sélectionnées")
    avg_row = summary.mean(numeric_only=True)
    avg_col1, avg_col2, avg_col3, avg_col4, avg_col5 = st.columns(5)
    with avg_col1:
        metric_card("Places moy", avg_row.get("capa_fin"))
    with avg_col2:
        metric_card("Candidatures moy", avg_row.get("voe_tot"))
    with avg_col3:
        metric_card("Propositions moy", avg_row.get("prop_tot"))
    with avg_col4:
        metric_card("Rang phase pple moy", avg_row.get("ran_grp1"))
    with avg_col5:
        metric_card("Tension moy", avg_row.get("tension"))

    st.divider()

    # Summary table with averages
    st.subheader("Évolution par année")
    
    # Créer la ligne de moyennes avec les mêmes colonnes que summary
    avg_row_dict = summary.mean(numeric_only=True).to_dict()
    avg_row_dict["Année"] = "Moyennes"
    
    # Fusionner summary avec la ligne de moyennes
    summary_with_avg = pd.concat([summary, pd.DataFrame([avg_row_dict])], ignore_index=True)
    
    # Renommer une seule fois
    summary_display = summary_with_avg.rename(columns=FRIENDLY_NAMES)
    
    cols_to_show = ["Année", "Places", "Candidatures totales", "Propositions", "Dernier rang phase principale", "Dernier rang final", "Tension"]
    cols_to_show = [c for c in cols_to_show if c in summary_display.columns]
    st.dataframe(summary_display[cols_to_show], use_container_width=True, hide_index=True)

    st.divider()

    # Chart
    st.subheader("Graphique")
    chart_cols = ["Places", "Candidatures totales", "Propositions"]
    chart_cols = [c for c in chart_cols if c in summary_display.columns]
    if chart_cols:
        chart_df = summary_display.set_index("Année")[chart_cols]
        st.line_chart(chart_df)
    else:
        st.info("Pas de données à afficher.")

    if show_raw:
        st.divider()
        st.subheader("Tableau complet")
        display_table = representative.drop(columns=["session"], errors="ignore").rename(columns=FRIENDLY_NAMES)
        st.dataframe(display_table, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("""
### C'est quoi ces chiffres
- **Places** = nombre de places disponibles
- **Candidatures** = nombre de candidats qui se sont inscrits
- **Propositions** = nombre d'admis (ceux qui ont eu oui)
- **Dernier rang phase principale** = jusqu'où ils ont appelé dans leur classement (plus bas = plus de monde appelé)
- **Dernier rang phase complémentaire** = idem pour la phase 2
- **Tension** = candidatures divisé par places (plus c'est élevé, plus c'est demandé)
    """)
