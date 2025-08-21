import io, json
from datetime import date
from collections import defaultdict
import numpy as np
import pandas as pd
import streamlit as st

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
    page_title="LSC Scouting",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# ----------------------- Upload -----------------------
uploaded = st.file_uploader("Carrega o CSV", type=["csv"])
if uploaded is None:
    st.info("Faz upload do CSV para começar.")
    st.stop()

content = uploaded.read()
df = read_csv_flex(content)
if df is None:
    st.error("Não consegui ler o CSV (verifica o separador).")
    st.stop()

# Força nomes de colunas únicos no CSV carregado
df.columns = make_unique(df.columns)

# Pré‑visualização opcional
show_preview = st.sidebar.checkbox("Mostrar pré‑visualização do CSV", value=False)
if show_preview:
    st.subheader("Pré‑visualização")
    st.dataframe(df.head(20), use_container_width=True)

# ----------------------- Mapeamento mínimo -----------------------
name_col       = guess(["name","player","jogador"])
team_col_g     = guess(["team","equipa","clube"], default=None)
division_col_g = guess(["division","league","competition","competição","liga","season"], default=None)
age_col_g      = guess(["age","idade"], default=None)
pos_col        = guess(["pos","posição","position","role"])
minutes_col    = guess(["min","minutes","mins","minutos"])
value_col      = guess(["market","valor","value","valormercado"], default=None)
contract_col   = guess(["contract","contrato","expiry","end"], default=None)

st.sidebar.header("Mapeamento")
name_col    = st.sidebar.selectbox("Nome do jogador", options=df.columns, index=list(df.columns).index(name_col))
team_col    = st.sidebar.selectbox("Equipa (opcional)", options=["(não usar)"] + list(df.columns),
                                   index=(0 if team_col_g is None else list(df.columns).index(team_col_g)+1))
division_col = st.sidebar.selectbox("Divisão/Liga (opcional)", options=["(não usar)"] + list(df.columns),
                                    index=(0 if division_col_g is None else list(df.columns).index(division_col_g)+1))
age_col      = st.sidebar.selectbox("Idade (opcional)", options=["(não usar)"] + list(df.columns),
                                    index=(0 if age_col_g is None else list(df.columns).index(age_col_g)+1))
pos_col     = st.sidebar.selectbox("Posição (texto)", options=df.columns, index=list(df.columns).index(pos_col))
minutes_col = st.sidebar.selectbox("Minutos", options=df.columns, index=list(df.columns).index(minutes_col))
value_col = st.sidebar.selectbox(
    "Valor de mercado (opcional)", options=["(não usar)"] + list(df.columns),
    index=(0 if value_col is None else list(df.columns).index(value_col)+1)
)
contract_col = st.sidebar.selectbox(
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

# ---- FIM DE CONTRATO (igual ao que tinhas) ----
if contract_col != "(não usar)":
    dfw["_contract_end"] = dfw[contract_col].apply(to_date_any)

# ----------------------- Controlo dos filtros (no mesmo grupo) -----------------------
# Valor de mercado
if "_market_value" in dfw.columns:
    mv_min = float(np.nanmin(dfw["_market_value"])) if np.isfinite(np.nanmin(dfw["_market_value"])) else 0.0
    mv_max = float(np.nanmax(dfw["_market_value"])) if np.isfinite(np.nanmax(dfw["_market_value"])) else 0.0
    val_range = st.sidebar.slider("Valor de mercado", min_value=float(mv_min), max_value=float(mv_max),
                                  value=(float(mv_min), float(mv_max)))
else:
    val_range = None

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
if "_age" in dfw.columns and dfw["_age"].notna().any():
    a_min = int(np.nanmin(dfw["_age"]))
    a_max = int(np.nanmax(dfw["_age"]))
    age_range = st.sidebar.slider("Idade", min_value=a_min, max_value=a_max, value=(a_min, a_max))
else:
    age_range = None

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
profile_labels = st.sidebar.multiselect(
    f"Etiquetas (posições) associadas a '{profile}'",
    options=unique_pos_vals,
    default=unique_pos_vals[:1]
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
mask_pos = dfw[pos_col].astype(str).isin(profile_labels) if profile_labels else pd.Series(False, index=dfw.index)
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

# ----------------------- Output (único, dedup robusto) -----------------------
# Ordem base: Nome, Equipa, Posição, Divisão, Idade, Minutos, extras, Scores, Métricas(+pct)
show_cols = [name_col]
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

    # gradiente no score 0-100
    if "score_0_100" in df_.columns:
        sty = sty.background_gradient(subset=["score_0_100"], cmap="Greens")

    # gradiente em percentis (_pct)
    pct_cols = [c for c in df_.columns if str(c).endswith("_pct")]
    if pct_cols:
        sty = sty.background_gradient(subset=pct_cols, cmap="Blues")

    # realce contratos a expirar (<= 12 meses) com tom #bd0003 leve
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

st.dataframe(_style_df(out), use_container_width=True)

st.caption("Score bruto = soma(peso × z‑score). Score (0–100) = percentil do score dentro do conjunto filtrado.")
st.dataframe(out, use_container_width=True)

# ----------------------- Exportações -----------------------
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


