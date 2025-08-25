# app_nextgen.py
import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# -------------------- PAGE SETUP & BRANDING --------------------
st.set_page_config(
    page_title="LSC Scouting — NextGen",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS leve para sidebar mais compacta
st.markdown("""
<style>
section[data-testid="stSidebar"] div[data-testid="stSidebarContent"]{ padding-top:0 !important; }
[data-testid="stSidebar"][aria-expanded="true"]{ min-width:260px; max-width:260px; }
section[data-testid="stSidebar"] hr{ border:none; border-top:1px solid #e7e9ee; margin:12px 0; }
</style>
""", unsafe_allow_html=True)

# ---- Password opcional via st.secrets["password"]
def _password_gate():
    if "password_checked" in st.session_state:
        return
    if "password" in st.secrets:
        with st.sidebar:
            st.markdown("#### Acesso")
            pwd = st.text_input("Password", type="password", key="_pwd_try")
            if st.button("Entrar", use_container_width=True):
                st.session_state["password_checked"] = (pwd == st.secrets["password"])
        if not st.session_state.get("password_checked", False):
            st.stop()
    else:
        st.session_state["password_checked"] = True

_password_gate()

# ---- Logo (usa logo.png na raiz; ajusta width conforme precisares)
with st.sidebar:
    p = Path(__file__).with_name("logo.png")
    if p.exists():
        st.image(str(p), width=80)
        st.markdown("<h3 style='text-align:center; color:#bd0003; margin-top:6px;'>Leixões SC</h3>",
                    unsafe_allow_html=True)
        st.markdown("---")

# -------------------- HELPERS --------------------
def _make_unique(cols):
    seen, out = {}, []
    for c in map(str, cols):
        if c in seen:
            seen[c]+=1; out.append(f"{c}.{seen[c]}")
        else:
            seen[c]=0; out.append(c)
    return out

@st.cache_data(show_spinner=False)
def _read_csv_bytes(file_bytes: bytes):
    for sep in [",",";","\t","|"]:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
            if 1 < df.shape[1] <= 500:
                df.columns = _make_unique(df.columns)
                return df
        except Exception:
            pass
    return None

def _upload_or_remember():
    """Carrega CSV e guarda em st.session_state['df_raw'] para reuso."""
    df = st.session_state.get("df_raw")
    with st.expander("📁 Carregar CSV", expanded=(df is None)):
        up = st.file_uploader("Seleciona o CSV", type=["csv"], label_visibility="visible")
        if up is not None:
            content = up.read()
            t = _read_csv_bytes(content)
            if t is None:
                st.error("Não consegui ler o CSV (tenta outro separador).")
            else:
                st.session_state["df_raw"] = t
                df = t
                st.success("CSV carregado.")
    return df

# -------------------- UI (HOME NEXTGEN) --------------------
st.title("⚽ LSC Scouting — NextGen")
st.write("Base nova para evoluir para navegação multi‑página, IA e gráficos avançados — sem tocar na app atual.")

df = _upload_or_remember()
if df is not None:
    # KPIs simples
    c1, c2, c3, c4 = st.columns(4)
    total = len(df)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    c1.metric("Registos no CSV", f"{total:,}")
    c2.metric("Colunas numéricas", f"{len(num_cols)}")
    c3.metric("Colunas totais", f"{df.shape[1]}")
    c4.metric("Estado", "Dados prontos ✅")

    st.markdown("---")
    st.subheader("Pré‑visualização")
    st.dataframe(df.head(20), use_container_width=True, height=600)

st.markdown("---")
st.subheader("Próximos passos")
st.write(
    "• Criar páginas: **Perfis & Ranking**, **Exploração de Métricas**, **Descoberta & IA**.\n"
    "• Partilhar sessão de dados (df) entre páginas.\n"
    "• Migrar funcionalidades gradualmente sem interromper a app existente."
)
