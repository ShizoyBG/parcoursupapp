from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Parcoursup Explorer", page_icon="🎓", layout="wide")

BASE_URL = "https://data.enseignementsup-recherche.gouv.fr/api/explore/v2.1/catalog/datasets"
YEARS = [2021, 2022, 2023, 2024, 2025]
DEFAULT_YEARS = [2022, 2023, 2024, 2025]
LATEST_YEAR = 2025

# 2025 = dataset sans suffixe. 2020-2024 = suffixe _YYYY.
DATASET_BY_YEAR = {
    2021: "fr-esr-parcoursup_2021",
    2022: "fr-esr-parcoursup_2022",
    2023: "fr-esr-parcoursup_2023",
    2024: "fr-esr-parcoursup_2024",
    2025: "fr-esr-parcoursup",
}

# Champs les plus utiles. On garde une version riche, puis des variantes réduites
# pour survivre aux années où certains champs n’existent pas.
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

SEARCH_FIELDS = [
    "cod_aff_form",
    "g_ea_lib_vx",
    "lib_for_voe_ins",
    "ville_etab",
    "dep_lib",
    "region_etab_aff",
    "acad_mies",
    "select_form",
    "fili",
]

FRIENDLY_NAMES = {
    "session": "Année",
    "cod_aff_form": "Code formation",
    "g_ea_lib_vx": "Établissement",
    "lib_for_voe_ins": "Formation",
    "contrat_etab": "Statut de l’établissement",
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
    "prop_tot": "Candidats ayant reçu une proposition",
    "acc_tot": "Candidats admis",
    "lib_grp1": "Groupe 1",
    "ran_grp1": "Dernier rang appelé (groupe 1 / phase principale)",
    "lib_grp2": "Groupe 2",
    "ran_grp2": "Dernier rang appelé (groupe 2)",
    "lib_grp3": "Groupe 3",
    "ran_grp3": "Dernier rang appelé (groupe 3)",
    "somme_rangs": "Somme des rangs (g1+g2+g3)",
    "moyenne_rangs": "Moyenne des rangs (g1+g2+g3)",
    "dernier_rang_max": "Dernier rang le plus loin",
    "tension": "Tension (candidatures / places)",
    "appels_par_place": "Propositions par place",
    "part_appelee": "Part des candidatures ayant reçu une proposition",
    "classement_par_appels": "Candidats classés / propositions",
}

DISPLAY_METRICS = [
    "capa_fin",
    "voe_tot",
    "nb_voe_pp",
    "nb_cla_pp",
    "prop_tot",
    "acc_tot",
    "ran_grp1",
    "ran_grp2",
    "ran_grp3",
    "somme_rangs",
    "moyenne_rangs",
    "dernier_rang_max",
    "tension",
    "appels_par_place",
    "part_appelee",
    "classement_par_appels",
]

CATEGORY_GROUPS = {
    "Places et demande": ["capa_fin", "voe_tot", "nb_voe_pp"],
    "Classement et appels": ["nb_cla_pp", "prop_tot", "acc_tot", "ran_grp1", "ran_grp2", "ran_grp3"],
    "Synthèse des rangs": ["somme_rangs", "moyenne_rangs", "dernier_rang_max"],
    "Ratios": ["tension", "appels_par_place", "part_appelee", "classement_par_appels"],
    "Contexte": ["g_ea_lib_vx", "lib_for_voe_ins", "ville_etab", "dep_lib", "region_etab_aff", "acad_mies", "select_form", "contrat_etab"],
}

SEARCHABLE_TEXT_COLUMNS = [
    "g_ea_lib_vx",
    "lib_for_voe_ins",
    "ville_etab",
    "dep_lib",
    "region_etab_aff",
    "acad_mies",
    "select_form",
    "fili",
]


# ---------- API helpers ----------


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_all_records(year: int, where: str, select: Sequence[str]) -> pd.DataFrame:
    """Fetch all matching records for one year, with graceful fallbacks for older schemas."""
    dataset = DATASET_BY_YEAR[year]
    url = f"{BASE_URL}/{dataset}/records"

    # Try the requested select first, then narrower variants if the schema complains.
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

            # No rows for this year/query: return immediately.
            return pd.DataFrame()

        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 400:
                last_error = f"{year}: schéma incompatible pour les champs demandés, tentative suivante."
                continue
            raise
        except requests.RequestException as exc:
            last_error = f"{year}: {exc}"
            continue

    if last_error:
        raise RuntimeError(last_error)
    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def search_formations(year: int, keyword: str, limit: int = 10) -> pd.DataFrame:
    keyword = keyword.strip().replace('"', '\\"')
    if not keyword:
        return pd.DataFrame()

    clauses = [f'{col} ilike "%{keyword}%"' for col in SEARCHABLE_TEXT_COLUMNS]
    where = " OR ".join(clauses)
    df = fetch_all_records(year, where, select=["cod_aff_form", "g_ea_lib_vx", "lib_for_voe_ins", "ville_etab", "dep_lib", "select_form"])
    if df.empty:
        return df

    # Deduplicate and make the result easy to read.
    if "cod_aff_form" in df.columns:
        df = df.drop_duplicates(subset=["cod_aff_form", "g_ea_lib_vx", "lib_for_voe_ins"])
    return df.head(limit).reset_index(drop=True)


# ---------- Parsing / cleaning ----------


def extract_code(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip()

    # Prefer explicit parameters when the user pastes a Parcoursup URL.
    for pat in [r"cod_aff_form[=:/?&\- ]+(\d+)", r"g_ea_cod[=:/?&\- ]+(\d+)"]:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            return m.group(1)

    # Try URL query parameters.
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

    # Fall back to the first 3-6 digit number.
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

    # Remove pure empty lines if any.
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
    """Keep one row per year. We prefer the row with the biggest values, which usually
    corresponds to the main/complete line for the formation when several rows exist."""
    if df.empty:
        return df

    sort_cols = [c for c in ["Année", "voe_tot", "prop_tot", "nb_cla_pp", "capa_fin"] if c in df.columns]
    if not sort_cols:
        return df

    temp = df.sort_values(by=sort_cols, ascending=[True] + [False] * (len(sort_cols) - 1))
    return temp.groupby("Année", as_index=False).first()


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

if "formation_code" not in st.session_state:
    st.session_state.formation_code = "4329"
if "formation_query" not in st.session_state:
    st.session_state.formation_query = "CPBx"
if "search_year" not in st.session_state:
    st.session_state.search_year = 2025
if "keyword_results" not in st.session_state:
    st.session_state.keyword_results = pd.DataFrame()


# ---------- UI ----------

st.title("Parcoursup Explorer")
st.caption("Un tableau simple pour comparer les formations Parcoursup sans jargon inutile.")

explorer_tab, find_tab, help_tab = st.tabs(["Explorer", "Trouver le code", "Aide"])

with st.sidebar:
    st.header("Affichage")
    years = st.multiselect("Années à comparer", YEARS, default=DEFAULT_YEARS)
    selected_groups = st.multiselect(
        "Catégories à afficher",
        list(CATEGORY_GROUPS.keys()),
        default=["Places et demande", "Classement et appels", "Ratios"],
    )
    show_raw = st.checkbox("Afficher la table brute", value=False)
    one_row_per_year = st.checkbox("Une ligne par année", value=True)
    st.divider()
    st.caption("Les données sont récupérées depuis l’open data Parcoursup, avec cache pour aller plus vite.")

selected_metric_keys: List[str] = []
for group in selected_groups:
    selected_metric_keys.extend(CATEGORY_GROUPS[group])
selected_metric_keys = list(dict.fromkeys(selected_metric_keys))  # keep order, remove duplicates

with find_tab:
    st.subheader("Trouver une formation")
    st.write("Tu peux entrer un code, coller une URL Parcoursup, ou chercher par mot-clé.")

    find_mode = st.radio(
        "Méthode",
        ["Code / URL", "Mot-clé"],
        horizontal=True,
        label_visibility="visible",
    )

    if find_mode == "Code / URL":
        with st.form("code_form"):
            raw_input = st.text_input(
                "Code ou URL Parcoursup",
                value=st.session_state.formation_code,
                help="Colle un code comme 4329, ou une URL de fiche Parcoursup.",
            )
            submitted = st.form_submit_button("Utiliser cette formation")

        if submitted:
            code = extract_code(raw_input)
            if code:
                st.session_state.formation_code = code
                st.success(f"Code reconnu : {code}")
            else:
                st.error("Impossible d’extraire un code de ce texte.")

    else:
        with st.form("keyword_form"):
            st.session_state.search_year = st.selectbox("Année de recherche", YEARS, index=YEARS.index(st.session_state.search_year) if st.session_state.search_year in YEARS else len(YEARS) - 1)
            keyword = st.text_input("Mot-clé", value=st.session_state.formation_query, help="Exemple : CPBx, ENSEIRB, licence maths, BUT info...")
            search_clicked = st.form_submit_button("Rechercher")

        if search_clicked:
            st.session_state.formation_query = keyword
            try:
                st.session_state.keyword_results = search_formations(st.session_state.search_year, keyword)
            except Exception as exc:
                st.session_state.keyword_results = pd.DataFrame()
                st.error(f"Recherche impossible : {exc}")

        results = st.session_state.keyword_results
        if isinstance(results, pd.DataFrame) and not results.empty:
            view = results.copy()
            view = view.rename(columns={
                "cod_aff_form": "Code",
                "g_ea_lib_vx": "Établissement",
                "lib_for_voe_ins": "Formation",
                "ville_etab": "Ville",
                "dep_lib": "Département",
                "select_form": "Sélectivité",
            })
            st.dataframe(view, use_container_width=True, hide_index=True)

            options = []
            mapping = {}
            for _, row in results.iterrows():
                code = str(row.get("cod_aff_form", "")).strip()
                establishment = str(row.get("g_ea_lib_vx", "")).strip()
                formation = str(row.get("lib_for_voe_ins", "")).strip()
                city = str(row.get("ville_etab", "")).strip()
                label = f"{code} — {establishment} — {formation} — {city}"
                options.append(label)
                mapping[label] = code

            chosen = st.selectbox("Choisir un résultat", options)
            if st.button("Utiliser cette formation"):
                st.session_state.formation_code = mapping[chosen]
                st.success(f"Formation chargée : {st.session_state.formation_code}")
        elif search_clicked:
            st.warning("Aucun résultat trouvé pour ce mot-clé.")

    st.divider()
    st.subheader("Aide rapide")
    st.markdown(
        """
- Le plus simple : colle le **code formation**.
- Tu peux aussi coller l’**URL de la fiche Parcoursup** : le code est extrait automatiquement.
- Tu peux rechercher par nom d’école, ville, diplôme, ou mot-clé.
- Si une formation a plusieurs lignes par année, l’app prend par défaut la ligne la plus représentative.
        """
    )

with help_tab:
    st.subheader("Mode d’emploi")
    st.markdown(
        """
**1. Trouver le code**
- Va sur la fiche Parcoursup de la formation.
- Copie le code si tu le vois.
- Sinon, colle l’URL dans l’onglet **Trouver le code**.

**2. Comparer les années**
- Coche les années à garder dans la barre latérale.
- Coche les catégories que tu veux voir dans le tableau et le graphique.

**3. Lire les indicateurs**
- **Places** = capacité d’accueil.
- **Candidatures totales** = demande globale.
- **Candidats ayant reçu une proposition** = personnes appelées.
- **Dernier rang appelé (groupe 1)** = profondeur d’appel en phase principale.
- **Somme des rangs** = `ran_grp1 + ran_grp2 + ran_grp3`.
        """
    )

with explorer_tab:
    code = st.session_state.formation_code
    st.subheader(f"Formation chargée : {code}")

    if not years:
        st.info("Choisis au moins une année dans la barre latérale.")
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
        st.warning("Certaines années n'ont pas pu être chargées : " + " | ".join(errors))

    if not frames:
        st.warning("Aucune donnée trouvée pour ce code.")
        st.stop()

    raw = pd.concat(frames, ignore_index=True)
    representative = pick_representative_rows(raw) if one_row_per_year else raw.copy()
    summary = aggregate_by_year(representative)

    # Canonical label
    latest_row = representative.sort_values("Année").iloc[-1]
    establishment = latest_row.get("g_ea_lib_vx", "")
    formation = latest_row.get("lib_for_voe_ins", "")
    title_line = establishment if establishment else formation
    subtitle_line = formation if formation and formation != title_line else ""

    if title_line:
        st.markdown(f"### {title_line}")
    if subtitle_line:
        st.caption(subtitle_line)

    # Friendly headline cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        metric_card("Places", latest_row.get("capa_fin"), "Capacité d’accueil")
    with c2:
        metric_card("Candidatures totales", latest_row.get("voe_tot"), "Demande totale")
    with c3:
        metric_card("Phase principale", latest_row.get("nb_voe_pp"), "Demande en phase principale")
    with c4:
        metric_card("Propositions", latest_row.get("prop_tot"), "Personnes ayant reçu au moins une proposition")
    with c5:
        metric_card("Dernier rang appelé", latest_row.get("ran_grp1"), "Groupe 1 / phase principale")
    with c6:
        metric_card("Somme des rangs", latest_row.get("somme_rangs"), "ran_grp1 + ran_grp2 + ran_grp3")

    st.divider()
    st.subheader("Moyennes sur les années sélectionnées")

    # Small KPIs from averages.
    avg_row = summary.mean(numeric_only=True)
    a1, a2, a3, a4, a5, a6 = st.columns(6)
    with a1:
        metric_card("Places moyennes", avg_row.get("capa_fin"))
    with a2:
        metric_card("Candidatures moyennes", avg_row.get("voe_tot"))
    with a3:
        metric_card("Propositions moyennes", avg_row.get("prop_tot"))
    with a4:
        metric_card("Dernier rang moyen", avg_row.get("ran_grp1"))
    with a5:
        metric_card("Somme rangs moyenne", avg_row.get("somme_rangs"))
    with a6:
        metric_card("Tension moyenne", avg_row.get("tension"), "candidatures / places")

    st.caption("Les moyennes sont calculées sur les années cochées, après récupération des données de chaque année.")

    # Pick columns to show
    table_base_cols = ["Année", "g_ea_lib_vx", "lib_for_voe_ins"]
    table_metric_cols = selected_metric_keys
    table_cols = list(dict.fromkeys(table_base_cols + table_metric_cols))

    display_table = representative.copy()
    if show_raw:
        st.subheader("Table brute")
        raw_display = display_table.drop(columns=["session"], errors="ignore").rename(columns=FRIENDLY_NAMES)
        cols_to_show = [FRIENDLY_NAMES.get(c, c) for c in table_cols if c in display_table.columns or c in FRIENDLY_NAMES]
        cols_to_show = [c for c in cols_to_show if c in raw_display.columns]
        if cols_to_show:
            st.dataframe(raw_display[cols_to_show], use_container_width=True, hide_index=True)
        else:
            st.dataframe(raw_display, use_container_width=True, hide_index=True)

    st.subheader("Tableau résumé")
    summary_display = summary.rename(columns=FRIENDLY_NAMES)
    cols_to_show = [FRIENDLY_NAMES.get(c, c) for c in ["Année"] + selected_metric_keys if FRIENDLY_NAMES.get(c, c) in summary_display.columns]
    if not cols_to_show:
        cols_to_show = [c for c in summary_display.columns if c != "Année"]
        cols_to_show = ["Année"] + cols_to_show if "Année" in summary_display.columns else cols_to_show
    st.dataframe(summary_display[cols_to_show], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Graphique")
    chart_metric_names = [FRIENDLY_NAMES.get(k, k) for k in selected_metric_keys if FRIENDLY_NAMES.get(k, k) in summary_display.columns]
    if chart_metric_names:
        chart_df = summary_display.set_index("Année")[chart_metric_names]
        st.line_chart(chart_df)
    else:
        st.info("Choisis au moins une catégorie à afficher pour voir un graphique.")

    st.divider()
    with st.expander("Voir les données détaillées de la formation"):
        details = representative.drop(columns=["session"], errors="ignore").rename(columns=FRIENDLY_NAMES)
        show_cols = [
            "Année",
            "Établissement",
            "Formation",
            "Places",
            "Candidatures totales",
            "Candidatures phase principale",
            "Candidats classés",
            "Candidats ayant reçu une proposition",
            "Candidats admis",
            "Dernier rang appelé (groupe 1 / phase principale)",
            "Dernier rang appelé (groupe 2)",
            "Dernier rang appelé (groupe 3)",
            "Somme des rangs (g1+g2+g3)",
            "Moyenne des rangs (g1+g2+g3)",
            "Dernier rang le plus loin",
            "Sélectivité",
            "Filière",
            "Ville",
            "Département",
            "Région",
            "Statut de l’établissement",
        ]
        show_cols = [c for c in show_cols if c in details.columns]
        st.dataframe(details[show_cols], use_container_width=True, hide_index=True)

    st.divider()
    st.markdown(
        """
**Lecture simple**
- Plus les **candidatures** sont grandes par rapport aux **places**, plus la formation est demandée.
- **Propositions** = personnes à qui la formation a dit oui à un moment de la campagne.
- **Dernier rang appelé** = jusqu’où la formation a dû descendre dans son classement pour faire ses appels.
- **Somme des rangs** = `ran_grp1 + ran_grp2 + ran_grp3`.
        """
    )

