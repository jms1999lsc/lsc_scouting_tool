import io, json
from datetime import date
from collections import defaultdict
import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode, JsCode
import altair as alt


# ---- PASSWORD LOGIN ----
def check_password():
    """Password simples para proteger a app"""
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # apaga da memória
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("❌ Password incorreta")
        return False
    else:
        return True

if not check_password():
    st.stop()

st.set_page_config(
    page_title="LSC Scouting Tool",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* menos “ar” no topo/baixo da página */
div.block-container { padding-top: .6rem; padding-bottom: .4rem; }
/* tabs mais juntinhas */
.stTabs [role="tablist"] { margin-bottom: .25rem; }
/* métricas (KPIs) com menos altura */
.css-1xarl3l, .stMetric { padding: .25rem .5rem; }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* Sidebar mais compacta e colada ao topo */
section[data-testid="stSidebar"] div[data-testid="stSidebarContent"]{
  padding-top: 0px !important;
}

/* Largura consistente da sidebar (ajusta a gosto) */
[data-testid="stSidebar"][aria-expanded="true"]{
  min-width: 260px;
  max-width: 260px;
}

/* Logo centrado e sem espaço extra */
section[data-testid="stSidebar"] img{
  display:block;
  margin: -50px auto 6px auto;   /* topo, direita, baixo, esquerda */
}

/* Expanders mais “magros” */
div[role="button"][data-baseweb="accordion"]{
  padding: 2px 8px !important;
}
div[data-testid="stExpander"] div[role="button"] p{
  margin: 4px 0 !important;
}

/* Linha separadora da sidebar com cor do clube */
section[data-testid="stSidebar"] hr{
  border: none;
  border-top: 1px solid #e7e9ee;
  margin: 14px 0;
}
</style>
""", unsafe_allow_html=True)

# CSS leve: esconde menu/rodapé e ajusta paddings
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container {padding-top: 0.8rem; padding-bottom: 2rem;}
section[data-testid="stSidebar"] .stSlider {margin-bottom: .6rem;}
h1, h2, h3 { letter-spacing: 0.2px; }
</style>
""", unsafe_allow_html=True)

# ----------------------- Helpers -----------------------
def read_csv_flex(file_bytes):
    for sep in [",",";","\t","|"]:
        try:
            t = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
            if 1 < t.shape[1] <= 300:
                return t
        except Exception:
            pass
    return None

def per90_from_raw(raw: pd.Series, minutes: pd.Series) -> pd.Series:
    m = pd.to_numeric(minutes, errors="coerce").fillna(0).clip(lower=1)
    x = pd.to_numeric(raw, errors="coerce").fillna(0)
    return x * (90 / m)

def is_per90_colname(name: str) -> bool:
    n = name.lower()
    return ("p90" in n) or ("per90" in n) or ("per 90" in n)

def zscore_group(series: pd.Series, group: pd.Series) -> pd.Series:
    g = series.groupby(group, dropna=False)
    mu = g.transform("mean")
    sd = g.transform(lambda x: x.std(ddof=0) if x.std(ddof=0) != 0 else 1)
    return (series - mu) / sd

def pct_group(series: pd.Series, group: pd.Series) -> pd.Series:
    return series.groupby(group, dropna=False).rank(pct=True)

def to_date_any(x):
    """Converte para date aceitando formatos comuns; devolve fim do mês quando dia não é dado."""
    if pd.isna(x):
        return None
    s = str(x).strip().strip('"').strip("'")
    for dayfirst in (True, False):
        try:
            d = pd.to_datetime(s, dayfirst=dayfirst, errors="raise")
            return d.date()
        except Exception:
            pass
    for fmt in ("%b %y", "%b %Y", "%m/%Y", "%Y-%m"):  # ex.: Jun 29, Jun 2029, 06/2029, 2029-06
        try:
            d = pd.to_datetime(s, format=fmt)
            d = d + pd.offsets.MonthEnd(1)
            return d.date()
        except Exception:
            pass
    return None

def guess(tokens, default=None):
    for c in df.columns:
        cl = c.lower()
        if any(tok in cl for tok in tokens):
            return c
    return default or df.columns[0]

def infer_already_normalized(series: pd.Series, minutes: pd.Series) -> bool:
    """
    Heurística:
    - Se valores parecem %/taxa (0–1 ou 0–100) -> True (já normalizada)
    - Senão, mede correlação |r| com minutos; r<0.35 -> True (provável per90); r>=0.35 -> False (raw)
    """
    s = pd.to_numeric(series, errors="coerce")
    m = pd.to_numeric(minutes, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    valid = s.notna() & m.notna()
    s2, m2 = s[valid], m[valid]
    if len(s2) >= 20:
        q1, q99 = s2.quantile(0.01), s2.quantile(0.99)
        if 0 <= q1 and q99 <= 1.0:
            return True
        if 0 <= q1 and q99 <= 100.0:
            return True
        r = np.corrcoef(s2, m2)[0,1]
        return abs(r) < 0.35
    return is_per90_colname(series.name)

def make_unique(columns):
    """Garante nomes de colunas únicos: repete com sufixos .1, .2, ..."""
    counts = {}
    new = []
    for c in map(str, columns):
        if c in counts:
            counts[c] += 1
            new.append(f"{c}.{counts[c]}")
        else:
            counts[c] = 0
            new.append(c)
    return new


# --- LOGO compacto e centrado (substitui o bloco anterior do logo) ---
from pathlib import Path
logo_path = Path(__file__).with_name("logo.png")  # precisa que o ficheiro se chame logo.png

with st.sidebar:
    # centrado e pequeno
    _l, _c, _r = st.columns([1, 2, 1])
    with _c:
        if logo_path.exists():
            st.image(str(logo_path), width=100)  # ajusta 60–100 a gosto
        else:
            st.caption("logo.png não encontrado")

    st.markdown(
        "<h3 style='text-align:center; color:#bd0003; margin-top:6px;'>Leixões SC - Dept. Scouting</h3>",
        unsafe_allow_html=True
    )
    st.markdown("---")


# Pré‑visualização opcional
show_preview = st.sidebar.checkbox("Mostrar pré-visualização do CSV", value=False)
if show_preview:
    st.subheader("Pré‑visualização")
    st.dataframe(df.head(20), use_container_width=True)

# ----------------------- Upload -----------------------
# ------- Upload (em expander para não ocupar o topo) -------
with st.expander("📁 Dados — Carregar & Pré‑visualizar", expanded=False):
    uploaded = st.file_uploader("Carrega um CSV", type=["csv"], label_visibility="visible")
    st.caption("Limite 200MB por ficheiro • CSV")

if uploaded is None:
    st.info("Carrega um CSV em **📁 Dados — Carregar & Pré‑visualizar** para começar.")
    st.stop()

content = uploaded.read()
df = read_csv_flex(content)
if df is None:
    st.error("Não consegui ler o CSV (verifica o separador).")
    st.stop()


# ----------------------- Mapeamento mínimo -----------------------
name_col       = guess(["name","player","jogador"])
team_col_g     = guess(["team","equipa","clube"], default=None)
division_col_g = guess(["division","league","competition","competição","liga","season"], default=None)
age_col_g      = guess(["age","idade"], default=None)
pos_col        = guess(["pos","posição","position","role"])
minutes_col    = guess(["min","minutes","mins","minutos"])
value_col      = guess(["market","valor","value","valormercado"], default=None)
contract_col   = guess(["contract","contrato","expiry","end"], default=None)


with st.sidebar.expander("⚙️ Mapeamento", expanded=True):
    name_col    = st.selectbox("Nome do jogador", options=df.columns, index=list(df.columns).index(name_col))
    team_col    = st.selectbox("Equipa (opcional)", options=["(não usar)"] + list(df.columns),
                               index=(0 if team_col_g is None else list(df.columns).index(team_col_g)+1))
    division_col = st.selectbox("Divisão/Liga (opcional)", options=["(não usar)"] + list(df.columns),
                                index=(0 if division_col_g is None else list(df.columns).index(division_col_g)+1))
    age_col = st.selectbox("Idade (opcional)", options=["(não usar)"] + list(df.columns),
                           index=(0 if age_col_g is None else list(df.columns).index(age_col_g)+1))
    pos_col     = st.selectbox("Posição (texto)", options=df.columns, index=list(df.columns).index(pos_col))
    minutes_col = st.selectbox("Minutos", options=df.columns, index=list(df.columns).index(minutes_col))
    value_col = st.selectbox(
        "Valor de mercado (opcional)", options=["(não usar)"] + list(df.columns),
        index=(0 if value_col is None else list(df.columns).index(value_col)+1)
    )
    contract_col = st.selectbox(
        "Fim de contrato (opcional)", options=["(não usar)"] + list(df.columns),
        index=(0 if contract_col is None else list(df.columns).index(contract_col)+1)
    )

# ----------------------- Filtros (juntos) -----------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Filtros")
min_minutes = st.sidebar.slider("Minutos mínimos", 0, 4500, 900, 30)
age_range = None          # ← NEW
val_range = None
d_from = d_to = None

# ----------------------- Preparar dataframe -----------------------
dfw = df.copy()
dfw[minutes_col] = pd.to_numeric(dfw[minutes_col], errors="coerce")
dfw = dfw[dfw[minutes_col].notna() & (dfw[minutes_col] >= min_minutes)].copy()

# ---- IDADE (novo) ----
if age_col != "(não usar)":
    dfw["_age"] = pd.to_numeric(dfw[age_col], errors="coerce")

# ---- VALOR DE MERCADO (igual ao que tinhas) ----
if value_col != "(não usar)":
    dfw["_market_value"] = pd.to_numeric(dfw[value_col], errors="coerce")
    dfw["_market_value_flt"] = dfw["_market_value"].fillna(0)  # <- NOVO (só para filtro)

# ---- FIM DE CONTRATO (igual ao que tinhas) ----
if contract_col != "(não usar)":
    dfw["_contract_end"] = dfw[contract_col].apply(to_date_any)

# ----------------------- Controlo dos filtros (no mesmo grupo) -----------------------
# Valor de mercado — filtro por MÁXIMO (inclui sem valor → tratados como 0)
if "_market_value_flt" in dfw.columns:
    mv_ceiling = float(dfw["_market_value_flt"].max()) if np.isfinite(dfw["_market_value_flt"].max()) else 0.0
    mv_max_sel = st.sidebar.slider(
        "Valor de Mercado",
        min_value=0.0,
        max_value=mv_ceiling,
        value=mv_ceiling,
    )
else:
    mv_max_sel = None

if mv_max_sel is not None and "_market_value_flt" in dfw.columns:
    dfw = dfw[dfw["_market_value_flt"] <= mv_max_sel].copy()


# Fim de contrato
if "_contract_end" in dfw.columns and dfw["_contract_end"].notna().any():
    dates_present = [d for d in dfw["_contract_end"] if d is not None]
    if dates_present:
        dmin, dmax = min(dates_present), max(dates_present)
        d_from, d_to = st.sidebar.date_input("Fim de contrato entre", value=(dmin, dmax))
    else:
        d_from = d_to = None
else:
    d_from = d_to = None

# Idade (novo)
# ---- IDADE (slider) ----
if age_col != "(não usar)":
    dfw["_age_num"] = pd.to_numeric(dfw[age_col], errors="coerce")
    if dfw["_age_num"].notna().any():
        a_min = int(np.nanmin(dfw["_age_num"]))
        a_max = int(np.nanmax(dfw["_age_num"]))
        # limites defensivos
        a_min = max(15, a_min)
        a_max = min(45, a_max)
        age_range = st.sidebar.slider("Idade", min_value=a_min, max_value=a_max,
                                      value=(a_min, a_max))
    else:
        age_range = None
else:
    age_range = None

# aplica idade diretamente em dfw (antes do perfil/etiquetas)
if age_range and "_age_num" in dfw.columns:
    dfw = dfw[dfw["_age_num"].between(age_range[0], age_range[1])].copy()


# ----------------------- Perfis & defaults -----------------------
KEYS = {
    "prog_passes": ["prog pass","progress pass","progressive","passes progressivos","passe progressivo"],
    "vertical_passes": ["vertical pass","vertical","passe vertical"],
    "first_phase": ["first phase","build","deep third","saída","constru","build-up","fase inicial"],
    "key_passes": ["key pass","pass to shot","assist","assistência","passe chave"],
    "final_third": ["final third","terço final","passe 3º terço","último terço"],
    "pen_area": ["penalty area","area pass","p/ área","caixa","pen area","área"],
    "recoveries": ["recovery","recuper","recuperações"],
    "press_succ": ["press success","successful press","counterpress","gegenpress","pressão","pressão bem-sucedida"],
    "interceptions": ["intercep","interceptações"],
    "tackles_won": ["tackle won","desarme","tackles","tackles ganhos"],
    "aerial_won": ["aerial won","header won","duelo aéreo ganho","cabeceamentos ganhos"],
    "clearances": ["clearance","alívio","alívios"],
    "blocks": ["block","bloqueios","remates bloqueados"],
    "carries": ["carry","progressive run","condução","conduções"],
    "dribbles": ["dribble","1v1","dribles","dribles bem-sucedidos"],
    "shots": ["shot","remate","remates"],
    "xg": ["xg","expected goals","xg total","golos esperados"],
    "touches_box": ["touches in box","area touches","toques na área"],
    "crosses_acc": ["cross acc","accurate cross","cruz","cruzamentos certos","cross accuracy"],
    # --- Guarda-Redes ---
    "gk_saves": ["save","saves","save %","save pct","% saves","saves in box",
                 "inside box","shots saved","defesas","paradas"],
    "gk_claims": ["claim","claims","claim accuracy","high claim","cross stopped",
                  "crosses stopped","saídas","bolas altas"],
    "gk_long": ["long pass","goal kick","launch","long distribution",
                "passes longos","pontapé longo","reposições longas"],
}

PROFILES = {
 "Guarda Redes":                ["gk_saves","gk_claims","gk_long","prog_passes","press_succ"],
 "Guarda Redes construtor":     ["gk_long","prog_passes","first_phase","gk_claims","gk_saves"],
 "Lateral Profundo":            ["crosses_acc","final_third","prog_passes","press_succ","recoveries"],
 "Lateral Associativo":         ["prog_passes","first_phase","key_passes","carries","dribbles"],
 "Defesa Central":              ["clearances","aerial_won","interceptions","tackles_won","blocks"],
 "Defesa Central com Bola":     ["prog_passes","vertical_passes","first_phase","carries","crosses_acc"],
 "Médio Defensivo":             ["recoveries","interceptions","press_succ","tackles_won","prog_passes"],
 "Médio Defensivo Construtor":  ["first_phase","prog_passes","vertical_passes","final_third","recoveries"],
 "Médio Centro Progressivo":    ["prog_passes","vertical_passes","carries","key_passes","press_succ"],
 "Extremo":                     ["dribbles","carries","key_passes","crosses_acc","touches_box"],
 "Ponta de Lança":              ["xg","shots","touches_box","key_passes","carries"],
 "Ponta de Lança Referência":   ["xg","shots","aerial_won","touches_box","key_passes"],
}

# Sugerir etiquetas (posições) por Perfil
PROFILE_TO_LABELS = {
    "Guarda Redes":                ["GK"],
    "Guarda Redes construtor":     ["GK"],
    "Lateral Profundo":            ["DL", "DML", "DR", "DMR"],
    "Lateral Associativo":         ["DL", "DML", "DR", "DMR"],
    "Defesa Central":              ["DC"],
    "Defesa Central com Bola":     ["DC"],
    "Médio Defensivo":             ["DMC", "MC"],
    "Médio Defensivo Construtor":  ["DMC", "MC"],
    "Médio Centro Progressivo":    ["MC", "AMC", "DMC"],
    "Extremo":                     ["FWL", "AML", "ML", "FWD", "AMR", "MR"],
    "Ponta de Lança":              ["FW"],
    "Ponta de Lança Referência":   ["FW"],
}

# Candidatos de métricas = todas as colunas à direita de "Minutos"
cols_order = list(dfw.columns)
try:
    idx_minutes = cols_order.index(minutes_col)
except ValueError:
    idx_minutes = -1
metric_candidates = cols_order[idx_minutes + 1:]

def suggest_defaults(profile_key_list, candidates):
    chosen = []
    clower = {c: c.lower() for c in candidates}
    for key in profile_key_list:
        kws = KEYS.get(key, [])
        pick = None
        for c in candidates:
            if any(k in clower[c] for k in kws):
                pick = c
                break
        if pick and pick not in chosen:
            chosen.append(pick)

    # >>> NÃO completar com qualquer métrica — só devolve os matches encontrados
    return chosen

# ----------------------- Sidebar: Perfil / Etiquetas / Métricas / Pesos -----------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Perfil / Etiquetas")
profile = st.sidebar.selectbox("Perfil a ranquear", list(PROFILES.keys()))
unique_pos_vals = sorted(map(str, dfw[pos_col].dropna().unique().tolist()))

# Valores disponíveis na coluna de posição
unique_pos_vals = sorted(map(str, dfw[pos_col].dropna().unique().tolist()))

# Sugestões automáticas para o perfil atual (intersetadas com o que existe no CSV)
suggested = PROFILE_TO_LABELS.get(profile, [])
default_labels = [x for x in suggested if x in unique_pos_vals]

# Multiselect (reinicia quando mudas o perfil; podes editar livremente)
profile_labels = st.sidebar.multiselect(
    f"Etiquetas (posições) associadas a '{profile}'",
    options=unique_pos_vals,
    default=default_labels,                # <- preenchido pelas sugestões do perfil
    placeholder="Seleciona uma ou mais posições…",
    key=f"labels_{profile}"               # <- força reset ao mudar de perfil
)

# Métricas (5) + override "já é per90/%"
st.sidebar.subheader("Métricas (5)")
defaults = suggest_defaults(PROFILES[profile], metric_candidates)

PLACEHOLDER = "(escolher métrica)"
metric_slots, already_norm_flags = [], []

for i in range(5):
    options = [PLACEHOLDER] + (metric_candidates if metric_candidates else [])
    # índice: se houver default válido, usa-o; senão fica no placeholder
    if i < len(defaults) and defaults[i] in metric_candidates:
        idx = 1 + metric_candidates.index(defaults[i])
    else:
        idx = 0

    mcol = st.sidebar.selectbox(
        f"Métrica {i+1}",
        options=options,
        index=idx,
        key=f"metric_sel_{i}"
    )

    if mcol == PLACEHOLDER:
        # slot vazio (utilizador ainda não escolheu) → não cria checkbox
        st.sidebar.caption("Escolhe uma métrica para este slot.")
        metric_slots.append(None)
        already_norm_flags.append(False)
        continue

    # inferir automaticamente se já é per90/% (podes corrigir no checkbox)
    infer_norm = infer_already_normalized(dfw[mcol], dfw[minutes_col])
    flag = st.sidebar.checkbox(
        "Já é per90/percentual (não converter)",
        value=bool(infer_norm),
        key=f"metric_norm_{i}"
    )
    st.sidebar.caption("Deteção sugere 'já normalizada'." if infer_norm else "Deteção sugere 'raw' → converter p/90.")

    metric_slots.append(mcol)
    already_norm_flags.append(flag)

# Aviso de repetidas (ignora slots vazios)
chosen_metrics = [m for m in metric_slots if m]
if len(set(chosen_metrics)) < len(chosen_metrics):
    st.sidebar.warning("⚠️ Tens métricas repetidas nos 5 slots — considera escolher 5 diferentes.")

# Pesos (soma OBRIGATÓRIA = 1.00, sem normalizar e sem cálculo automático)
st.sidebar.subheader("Pesos (total deve ser 1.00)")

weights = {}
chosen_metrics = [m for m in metric_slots if m]  # ignora slots vazios

if not chosen_metrics:
    st.sidebar.info("Escolhe pelo menos 1 métrica para definir pesos.")
else:
    # sliders independentes (passo mais fino para ser fácil acertar 1.00)
    for i, met in enumerate(chosen_metrics):
        weights[met] = st.sidebar.slider(met, 0.0, 1.0, 0.20, 0.01, key=f"w_{i}")

    total_w = sum(weights.values())
    eps = 1e-6  # tolerância numérica

    if total_w > 1.0 + eps:
        st.sidebar.error(f"❌ Os pesos somam {total_w:.2f} (> 1.00). Reduz um ou mais pesos.")
        st.stop()
    elif total_w < 1.0 - eps:
        st.sidebar.error(f"❌ Os pesos somam {total_w:.2f} (< 1.00). Aumenta os pesos até perfazer 1.00.")
        st.stop()
    else:
        st.sidebar.caption("✅ Total = 1.00")

# ----------------------- Preparar colunas per90 conforme flags -----------------------
per90_cols = []
for col, is_norm in zip(metric_slots, already_norm_flags):
    if not col:
        continue
    if is_norm or is_per90_colname(col):
        dfw[col] = pd.to_numeric(dfw[col], errors="coerce").fillna(0)
        per90_cols.append(col)
    else:
        out_col = f"{col}_p90"
        if out_col not in dfw.columns:
            dfw[out_col] = per90_from_raw(dfw[col], dfw[minutes_col])
        per90_cols.append(out_col)

# ----------------------- Ranking -----------------------
if profile_labels:
    mask_pos = dfw[pos_col].astype(str).isin(profile_labels)
else:
    mask_pos = pd.Series(True, index=dfw.index)   # <- sem seleção = todas as posições
dfp = dfw[mask_pos].copy()

for met in set(per90_cols):
    dfp[met + "_z"]  = zscore_group(dfp[met], dfp[pos_col])
    dfp[met + "_pct"] = pct_group(dfp[met], dfp[pos_col])

if val_range and "_market_value" in dfp.columns:
    dfp = dfp[dfp["_market_value"].between(val_range[0], val_range[1])]
if (d_from and d_to) and "_contract_end" in dfp.columns:
    dfp = dfp[dfp["_contract_end"].apply(lambda x: x is not None and d_from <= x <= d_to)]

if not len(dfp):
    st.warning("Nenhum jogador cumpre os filtros/etiquetas selecionados.")
    st.stop()

dfp["score"] = sum(
    weights[src] * dfp[(src if (flag or is_per90_colname(src)) else f"{src}_p90") + "_z"]
    for src, flag in zip(metric_slots, already_norm_flags)
    if src and src in weights  # <— ignora slots vazios
)
dfp["score_0_100"] = (dfp["score"].rank(pct=True) * 100).round(1)

# --- Badge simples de qualidade de amostra (🟩/🟨/🟥) ---
def _sample_quality_row(row):
    mins_ok = pd.to_numeric(row[minutes_col], errors="coerce") >= 900
    # heurística leve: se existir pelo menos 1 coluna *_pct, damos mais 1 ponto
    pct_cols = [c for c in dfp.columns if str(c).endswith("_pct")]
    pct_ok = len(pct_cols) > 0
    score = int(mins_ok) + int(pct_ok)  # 0..2
    if score >= 2: return "🟩"
    if score == 1: return "🟨"
    return "🟥"

dfp["_sample_quality"] = dfp.apply(_sample_quality_row, axis=1)

# ----------------------- Output (único, dedup robusto) -----------------------
# --- KPIs rápidos ---
k1, k2, k3 = st.columns([1,1,1])
# ---- util para obter série de datas de contrato, qualquer que seja o nome da coluna
def _contract_series(df):
    if "contract_end" in df.columns:
        return pd.to_datetime(df["contract_end"], errors="coerce")
    if "_contract_end" in df.columns:
        return pd.to_datetime(df["_contract_end"], errors="coerce")
    return None

_s_contract = _contract_series(dfp)
if _s_contract is not None:
    _days = (_s_contract - pd.Timestamp.today().normalize()).dt.days
    _n_expiring = int((_days <= 365).sum())
else:
    _n_expiring = 0
k1.metric("Jogadores", f"{len(dfp):,}".replace(",","."))
if age_col != "(não usar)" and age_col in dfp.columns:
    k2.metric("Idade média", f"{pd.to_numeric(dfp[age_col], errors='coerce').mean():.1f}")
else:
    k2.metric("Idade média", "—")
if "contract_end" in dfp.columns:
    _days = (pd.to_datetime(dfp["contract_end"], errors="coerce") - pd.Timestamp.today()).dt.days
    k3.metric("Contrato < 12 meses", _n_expiring)
else:
    k3.metric("Contrato < 12 meses", 0)
st.markdown("<hr>", unsafe_allow_html=True)
# Ordem base: Nome, Equipa, Posição, Divisão, Idade, Minutos, extras, Scores, Métricas(+pct)
show_cols = [name_col]
# inserir a badge de qualidade logo a seguir ao nome
if "_sample_quality" in dfp.columns and "_sample_quality" not in show_cols:
    show_cols.insert(1, "_sample_quality")
if team_col != "(não usar)":
    show_cols.append(team_col)
show_cols.append(pos_col)
if division_col != "(não usar)":
    show_cols.append(division_col)
if age_col != "(não usar)":
    show_cols.append(age_col)
show_cols.append(minutes_col)

if "_market_value" in dfp.columns: show_cols.append("_market_value")
if "_contract_end" in dfp.columns: show_cols.append("_contract_end")
show_cols += ["score", "score_0_100"]

for src, flag in zip(metric_slots, already_norm_flags):
    if not src:
        continue
    per90_name = src if (flag or is_per90_colname(src)) else f"{src}_p90"
    show_cols += [per90_name, per90_name + "_pct"]

# dedupe da lista
seen = set()
ordered_unique = []
for c in show_cols:
    if c not in seen:
        ordered_unique.append(c)
        seen.add(c)

# construir coluna-a-coluna com nomes finais únicos (aplicando renames de _market/_contract)
series_list = []
name_counts = defaultdict(int)

def target_name(src_name: str) -> str:
    if src_name == "_market_value": return "market_value"
    if src_name == "_contract_end": return "contract_end"
    return str(src_name)

for src in ordered_unique:
    if src not in dfp.columns:
        continue
    tgt = target_name(src)
    existing = [s.name for s in series_list]
    if tgt in existing:
        name_counts[tgt] += 1
        tgt = f"{tgt}.{name_counts[tgt]}"
    else:
        name_counts[tgt] = 0
    s = dfp[src].copy()
    s.name = tgt
    series_list.append(s)

out = pd.concat(series_list, axis=1).sort_values("score", ascending=False).reset_index(drop=True)

# unicidade final por segurança
out.columns = make_unique(out.columns)

def _style_df(df_):
    sty = df_.style

    # Definir formatos para cada tipo de coluna
    sty = sty.format({
        "Age": "{:.0f}",             # sem casas decimais
        "Minutes": "{:.0f}",         # sem casas decimais
        "market_value": "{:,.0f}",   # inteiro com separador
        "score": "{:.3f}",           # 3 casas
        "score_0_100": "{:.1f}",     # 1 casa
    })

    # Aplicar também às métricas % (terminam em _pct) -> 3 casas
    pct_cols = [c for c in df_.columns if str(c).endswith("_pct")]
    for c in pct_cols:
        sty = sty.format({c: "{:.3f}"})

    # Gradientes visuais
    if "score_0_100" in df_.columns:
        sty = sty.background_gradient(subset=["score_0_100"], cmap="Greens")
    if pct_cols:
        sty = sty.background_gradient(subset=pct_cols, cmap="Blues")

    # Contrato a vermelho se expira em < 12 meses
    if "contract_end" in df_.columns:
        def warn_contract(col):
            today = pd.Timestamp.today().date()
            def colorize(x):
                try:
                    d = pd.to_datetime(x).date()
                    months = (d.year - today.year) * 12 + (d.month - today.month)
                    return "background-color: rgba(189,0,3,0.08)" if months <= 12 else ""
                except Exception:
                    return ""
            return [colorize(v) for v in col]
        sty = sty.apply(warn_contract, subset=["contract_end"])

    return sty

    
st.caption("Score bruto = soma(peso × z‑score). Score (0–100) = percentil do score dentro do conjunto filtrado.")

# ... código acima que prepara o "out" ...

# --- FORMATAÇÃO NUMÉRICA ANTES DE ESTILIZAR ---
out_fmt = out.copy()

# 0 casas: idade, minutos, market_value
for col0 in [c for c in [age_col, minutes_col, "market_value"] if c in out_fmt.columns]:
    out_fmt[col0] = pd.to_numeric(out_fmt[col0], errors="coerce").round(0).astype("Int64")

# 3 casas: score bruto
if "score" in out_fmt.columns:
    out_fmt["score"] = pd.to_numeric(out_fmt["score"], errors="coerce").round(4)

# 1 casa: score 0–100
if "score_0_100" in out_fmt.columns:
    out_fmt["score_0_100"] = pd.to_numeric(out_fmt["score_0_100"], errors="coerce").round(2)

# 3 casas: todas as colunas que terminem com _pct
for c in [c for c in out_fmt.columns if str(c).endswith("_pct")]:
    out_fmt[c] = pd.to_numeric(out_fmt[c], errors="coerce").round(3)

# --- ESTILO ---
def _style_df(df_):
    sty = df_.style
    pct_cols = [c for c in df_.columns if str(c).endswith("_pct")]
    if pct_cols:
        sty = sty.background_gradient(subset=pct_cols, cmap="Blues")
    if "contract_end" in df_.columns:
        def warn_contract(col):
            today = pd.Timestamp.today().date()
            def colorize(x):
                try:
                    d = pd.to_datetime(x).date()
                    months = (d.year - today.year) * 12 + (d.month - today.month)
                    return "background-color: rgba(189,0,3,0.08)" if months <= 12 else ""
                except Exception:
                    return ""
            return [colorize(v) for v in col]
        sty = sty.apply(warn_contract, subset=["contract_end"])
    return sty

# ================== TABELA (AgGrid) COM FORMATAÇÃO + TABS ==================
# 1) Preparar cópia para formatação
table = out.copy()

# Datas human-readable
if "contract_end" in table.columns:
    table["contract_end"] = pd.to_datetime(table["contract_end"], errors="coerce").dt.strftime("%Y-%m-%d")

# Formatters / cell styles
fmt_3dec = JsCode("function(p){ if(p.value==null) return ''; return Number(p.value).toFixed(3); }")
fmt_1dec = JsCode("function(p){ if(p.value==null) return ''; return Number(p.value).toFixed(1); }")

cell_divergent = JsCode("""
function(p){
  if (p.value == null) return {};
  const v = Number(p.value);
  const clip = Math.max(-3, Math.min(3, v));
  const hue  = (clip >= 0) ? 210 : 0;            // 210=azul, 0=vermelho
  const light = 100 - (Math.abs(clip)/3)*60;     // 100→40
  const color = `hsl(${hue}, 82%, ${light}%)`;
  const txt   = (light < 55) ? 'white' : 'black';
  return {'backgroundColor': color, 'color': txt};
}
""")

cell_blue_grad = JsCode("""
function(p){
  if (p.value == null) return {};
  const v = Math.max(0, Math.min(100, Number(p.value)));
  const light = 100 - v*0.5;                     // 100→50
  const color = `hsl(210, 85%, ${light}%)`;
  const txt   = (light < 55) ? 'white' : 'black';
  return {'backgroundColor': color, 'color': txt};
}
""")

cell_contract_warn = JsCode("""
function(p){
  if (!p.value) return {};
  var d = new Date(p.value);
  if (isNaN(d)) return {};
  var today = new Date();
  var diffDays = (d - today) / (1000*60*60*24);
  if (diffDays < 0) {
    return {'backgroundColor':'#ffd6d6', 'color':'#7a0000'};   // expirado
  }
  if (diffDays <= 365) {
    return {'backgroundColor':'#ffecec', 'color':'#7a0000'};   // < 12m
  }
  return {};
}
""")

# 2) GridOptions
gb = GridOptionsBuilder.from_dataframe(table)
gb.configure_default_column(filter=True, sortable=True, resizable=True, floatingFilter=True)

# tooltips principais
tooltips = {}
tooltips[str(name_col)] = "Nome do jogador"
if team_col != "(não usar)" and team_col in table.columns:
    tooltips[team_col] = "Equipa / Clube"
if "score" in table.columns:
    tooltips["score"] = "Soma ponderada de z-scores (negativo/positivo)"
if "score_0_100" in table.columns:
    tooltips["score_0_100"] = "Percentil do score dentro do conjunto filtrado"
if "contract_end" in table.columns:
    tooltips["contract_end"] = "Fim de contrato (vermelho = < 12 meses ou expirado)"
for col, tip in tooltips.items():
    if col in table.columns:
        gb.configure_column(col, headerTooltip=tip, tooltipField=col)

# filtro de texto explícito em Name
gb.configure_column(str(name_col), filter="agTextColumnFilter")

# alinhamento numérico à direita
for c in table.columns:
    if pd.api.types.is_numeric_dtype(table[c]):
        gb.configure_column(c, type=["rightAligned"])

# formatação e cores
if "score" in table.columns:
    gb.configure_column("score", valueFormatter=fmt_3dec, cellStyle=cell_divergent)
if "score_0_100" in table.columns:
    gb.configure_column("score_0_100", valueFormatter=fmt_1dec, cellStyle=cell_blue_grad)
for c in table.columns:
    if str(c).endswith("_pct"):
        gb.configure_column(c, valueFormatter=fmt_1dec, cellStyle=cell_blue_grad)
if "contract_end" in table.columns:
    gb.configure_column("contract_end", cellStyle=cell_contract_warn)

# colunas pinadas + sort padrão
# colunas pinadas + sort padrão
pin_cols = [str(name_col)]
if "_sample_quality" in table.columns:
    pin_cols.append("_sample_quality")    # <- pin badge
for c in [team_col, pos_col]:
    if c and c in table.columns and c not in pin_cols:
        pin_cols.append(c)
gb.configure_columns(pin_cols, pinned=True)

if "score" in table.columns:
    gb.configure_column("score", sort="desc")

go = gb.build()

# linhas/cabeçalho mais compactos
go["rowHeight"] = 30             # default ~ 37
go["headerHeight"] = 34          # default ~ 42


# pesquisa global (quick filter) e autofit em render/resize
q = st.text_input("🔎 Pesquisa global na tabela", "", placeholder="Nome, equipa, liga…")
if q:
    go["quickFilterText"] = q
go["onFirstDataRendered"] = JsCode("function(p){p.api.sizeColumnsToFit();}")
go["onGridSizeChanged"]   = JsCode("function(p){p.api.sizeColumnsToFit();}")

# 3) Tabs: Ranking / Gráficos rápidos
tab1, tab2 = st.tabs(["📊 Ranking", "📈 Gráficos rápidos"])
with tab1:
    AgGrid(
        table,
        gridOptions=go,
        theme="balham",
        height=780,
        allow_unsafe_jscode=True
    )

with tab2:
    # gráfico simples: distribuição do score ou percentis
    _opts = [c for c in table.columns if c in ("score","score_0_100") or str(c).endswith("_pct")]
    if _opts:
        sel = st.selectbox("Distribuição de:", _opts, index=0)
        chart = alt.Chart(table).mark_bar().encode(
            x=alt.X(f"{sel}:Q", bin=alt.Bin(maxbins=30)),
            y='count()',
            tooltip=[str(name_col), sel]
        ).properties(height=320)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Sem colunas numéricas selecionáveis para gráfico.")
# ================== /TABELA (AgGrid) ==================

# Exportações
csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Exportar CSV", data=csv_bytes, file_name=f"ranking_{profile}.csv", mime="text/csv")

try:
    import io as _io
    import xlsxwriter  # noqa
    buf = _io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        out.to_excel(writer, index=False, sheet_name="ranking")
    st.download_button("⬇️ Exportar Excel", data=buf.getvalue(), file_name=f"ranking_{profile}.xlsx")
except Exception:
    st.info("Para exportar em Excel, instala:  pip install XlsxWriter")

# ----------------------- Presets -----------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Presets")
preset = {
    "name_col": name_col, "team_col": team_col, "division_col": division_col, "age_col": age_col,
    "pos_col": pos_col, "minutes_col": minutes_col,
    "value_col": value_col, "contract_col": contract_col,
    "profile": profile, "profile_labels": profile_labels,
    "metric_slots": metric_slots, "already_norm_flags": already_norm_flags,
    "weights": weights, "min_minutes": int(min_minutes),
}
st.sidebar.download_button("💾 Guardar preset", data=json.dumps(preset, ensure_ascii=False).encode("utf-8"),
                           file_name=f"preset_{profile}.json", mime="application/json")
preset_up = st.sidebar.file_uploader("Carregar preset (.json)", type=["json"], label_visibility="collapsed")
if preset_up:
    try:
        P = json.loads(preset_up.read().decode("utf-8"))
        st.sidebar.success("Preset carregado (aplica manualmente as escolhas na UI).")
    except Exception as e:
        st.sidebar.error(f"Preset inválido: {e}")












































































