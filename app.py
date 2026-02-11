import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Epic Store Analytics | Neural Core",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. CYBERPUNK VISUAL ENGINE (CSS)
# ==============================================================================
def load_css():
    st.markdown("""
    <style>
        /* --- GLOBAL --- */
        .stApp {
            background-color: #050505;
        }
        
        /* --- REMOVE TOP PADDING --- */
        .block-container {
            padding-top: 1rem !important; /* Default is usually 6rem */
            padding-bottom: 1rem !important;
        }
        
        /* 1. Make the header transparent so it blends with the background */
        header[data-testid="stHeader"] {
            background-color: transparent !important;
        }

        /* 2. Hide the colored decorative line at the top */
        div[data-testid="stDecoration"] {
            display: none;
        }

        /* 3. Style the Sidebar Toggle Button to match your Cyberpunk theme */
        button[kind="header"] {
            background-color: transparent !important;
            color: #00ffcc !important; /* Teal arrow */
            border: 1px solid #00ffcc;
        }
        
        /* --- SCROLLBARS --- */
        ::-webkit-scrollbar { width: 10px; }
        ::-webkit-scrollbar-track { background: #111; }
        ::-webkit-scrollbar-thumb { background: #333; border-radius: 5px; }
        ::-webkit-scrollbar-thumb:hover { background: #00ffcc; }

        /* --- TEXT STYLES --- */
        h1, h2, h3 { font-family: 'Courier New', monospace; letter-spacing: -1px; }
        h1 { text-shadow: 0 0 10px rgba(0, 255, 204, 0.3); color: #fff; }
        
        /* --- CUSTOM CLASSES --- */
        .neon-text-cyan { 
            color: #00ffcc; 
            text-shadow: 0 0 8px rgba(0, 255, 204, 0.6); 
            font-weight: bold; font-family: 'Courier New'; 
        }
        .neon-text-magenta { 
            color: #ff00ff; 
            text-shadow: 0 0 8px rgba(255, 0, 255, 0.6); 
            font-weight: bold; font-family: 'Courier New'; 
        }

        /* --- CARDS --- */
        .glass-card {
            background: rgba(20, 20, 20, 0.7);
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease, border-color 0.3s ease;
        }
        .glass-card:hover {
            border-color: #00ffcc;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 255, 204, 0.1);
        }

        /* --- SIDEBAR --- */
        section[data-testid="stSidebar"] {
            background-color: #080808;
            border-right: 1px solid #222;
        }
        
        /* --- BUTTONS (Teal Background + Dark Text) --- */
        .stButton > button {
            background-color: #00ffcc !important; /* Teal Background */
            color: #000000 !important;            /* Black Text */
            border: 1px solid #00ffcc !important;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-weight: 800; /* Extra Bold for readability */
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #00bfff !important; /* Slightly Blue-ish on Hover */
            color: #000000 !important;            
            box-shadow: 0 0 15px #00ffcc;         /* Glow Effect */
            border-color: #00bfff !important;
        }
        .stButton > button:active {
            background-color: #ffffff !important; /* White on Click */
            box-shadow: 0 0 20px #ffffff;
        }

        /* --- METRICS & PROGRESS --- */
        [data-testid="stMetricValue"] {
            color: #00ffcc !important;
            font-family: 'Courier New';
            text-shadow: 0 0 5px rgba(0, 255, 204, 0.4);
        }
        .confidence-bar {
            background: #222; height: 8px; border-radius: 4px; width: 100%; margin-top: 5px;
        }
        .confidence-fill {
            height: 100%; border-radius: 4px; 
            box-shadow: 0 0 10px currentColor; 
            transition: width 1s ease;
        }
                
        /* --- MULTISELECT TAGS (Fix White Text on Teal) --- */
        span[data-baseweb="tag"] {
            background-color: #00ffcc !important; /* Teal Background */
        }
        span[data-baseweb="tag"] span {
            color: #000000 !important;            /* Black Text */
            font-weight: bold;
        }
        
        /* Fix the 'X' close icon color */
        span[data-baseweb="tag"] svg {
            fill: #000000 !important;
            color: #000000 !important;
        }
                
    </style>
    """, unsafe_allow_html=True)

load_css()

# LOADING ANIMATION

# ==============================================================================
# 2.5 BOOT SEQUENCE ANIMATION
# ==============================================================================
def run_boot_sequence():
    """
    Renders a full-screen Cyberpunk boot sequence.
    Runs only once per session.
    """
    if 'boot_completed' in st.session_state:
        return

    # -- ANIMATION CSS --
    st.markdown("""
    <style>
        @keyframes flicker {
            0% { opacity: 0.8; } 5% { opacity: 1; } 10% { opacity: 0.8; }
            15% { opacity: 1; } 20% { opacity: 1; } 25% { opacity: 0.8; }
            30% { opacity: 1; } 100% { opacity: 1; }
        }
        @keyframes scanline {
            0% { transform: translateY(-100%); }
            100% { transform: translateY(100%); }
        }
        .boot-container {
            position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
            background-color: #000000;
            z-index: 99999;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            font-family: 'Courier New', monospace;
            overflow: hidden;
        }
        .boot-text {
            color: #00ffcc; font-size: 1.5rem; font-weight: bold;
            text-shadow: 0 0 10px #00ffcc;
            margin-bottom: 20px;
            animation: flicker 0.2s infinite;
        }
        .boot-log {
            color: #ff00ff; font-size: 0.9rem;
            text-align: left; width: 60%;
            border-left: 2px solid #ff00ff;
            padding-left: 15px;
            height: 150px; overflow: hidden;
            font-family: 'Courier New';
            opacity: 0.8;
        }
        .scan-line {
            position: absolute; top: 0; left: 0; width: 100%; height: 20px;
            background: rgba(0, 255, 204, 0.2);
            box-shadow: 0 0 10px rgba(0, 255, 204, 0.5);
            animation: scanline 2s linear infinite;
            pointer-events: none;
        }
        .progress-border {
            width: 60%; height: 20px; border: 1px solid #333;
            margin-top: 20px; position: relative;
        }
        .progress-fill {
            height: 100%; background-color: #00ffcc;
            box-shadow: 0 0 15px #00ffcc;
            transition: width 0.1s ease;
        }
    </style>
    """, unsafe_allow_html=True)

    # -- ANIMATION LOGIC --
    placeholder = st.empty()
    
    # Fake "Boot steps"
    steps = [
        "INITIALIZING NEURAL LINK...",
        "BYPASSING EPIC STORE FIREWALL...",
        "DECRYPTING GAME MANIFEST...",
        "LOADING TENSORFLOW KERNELS...",
        "OPTIMIZING GPU SHADERS...",
        "SYNCHRONIZING REVIEWS...",
        "SYSTEM READY."
    ]
    
    logs = []
    
    # Run the sequence
    with placeholder.container():
        for i, step in enumerate(steps):
            # Generate fake hex code scrolling
            hex_noise = " ".join([hex(np.random.randint(0, 255))[2:].upper().zfill(2) for _ in range(8)])
            logs.append(f"> {hex_noise} :: {step}")
            if len(logs) > 6: logs.pop(0) # Keep log short
            
            # Progress calculation
            progress = int((i + 1) / len(steps) * 100)
            
            # Render HTML manually
            st.markdown(f"""
                <div class="boot-container">
                    <div class="scan-line"></div>
                    <div class="boot-text">EPIC ANALYTICS PROTOCOL v3.0</div>
                    <div class="boot-log">
                        {"<br>".join(logs)}
                    </div>
                    <div class="progress-border">
                        <div class="progress-fill" style="width: {progress}%;"></div>
                    </div>
                    <p style="color: #666; margin-top: 10px;">LOADING MODULES... {progress}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Pause for effect (fast enough not to be annoying)
            time.sleep(0.3 if i < len(steps)-1 else 0.8)

    # Clear screen and set state
    placeholder.empty()
    st.session_state.boot_completed = True


# ==============================================================================
# 3. REAL DATA LOADING & MERGING
# ==============================================================================
@st.cache_data
def load_data():
    """
    Final Robust Loader: Integrated Notebook Logic + CPU/GPU Brand Extraction
    """
    try:
        # 1. Load CSVs
        games = pd.read_csv('data/games.csv', index_col=False)
        critic = pd.read_csv('data/open_critic.csv', index_col=False)
        social = pd.read_csv('data/social_networks.csv', index_col=False)
        
        # 2. Hardware: 7-column fix
        hw_cols = ['id_raw', 'os', 'cpu', 'ram_raw', 'gpu', 'storage', 'fk_game_id']
        hardware = pd.read_csv('data/necessary_hardware.csv', names=hw_cols, header=0, index_col=False)

        # 3. Standardize IDs
        games['id'] = games['id'].astype(str).str.strip().str.lower()
        critic['game_id'] = critic['game_id'].astype(str).str.strip().str.lower()
        social['fk_game_id'] = social['fk_game_id'].astype(str).str.strip().str.lower()
        hardware['id_link'] = hardware['fk_game_id'].astype(str).str.strip().str.lower().str[:32]

        # 4. Aggregate Critic Scores
        critic_agg = critic.groupby('game_id').agg({
            'rating': 'mean',
            'comment': lambda x: ' ||| '.join(str(v) for v in x if pd.notna(v)), 
            'company': 'count'
        }).reset_index().rename(columns={'game_id': 'id', 'rating': 'critic_score', 'comment': 'top_comments', 'company': 'review_count'})

        # 5. Extract Hardware Specs (RAM, CPU, GPU)
        import re
        
        def extract_ram_streamlit(row):
            text = f"{row['ram_raw']} {row['storage']} {row['cpu']} {row['gpu']}".upper()
            match_gb = re.search(r'(\d+(?:\.\d+)?)\s?GB', text)
            if match_gb: return float(match_gb.group(1))
            match_mb = re.search(r'(\d+(?:\.\d+)?)\s?MB', text)
            if match_mb: return float(match_mb.group(1)) / 1024
            return 0

        def extract_cpu_brand(text):
            text = str(text).lower()
            if 'intel' in text or 'i3' in text or 'i5' in text or 'i7' in text: return 'Intel'
            if 'amd' in text or 'ryzen' in text or 'athlon' in text: return 'AMD'
            return 'Generic/Other'

        def extract_gpu_brand(text):
            text = str(text).lower()
            if any(x in text for x in ['nvidia', 'geforce', 'gtx', 'rtx']): return 'Nvidia'
            if any(x in text for x in ['amd', 'radeon', 'rx']): return 'AMD'
            return 'Integrated/Other'

        # Apply extraction to hardware dataframe
        hardware['min_ram_gb'] = hardware.apply(extract_ram_streamlit, axis=1)
        hardware['cpu_brand'] = hardware['cpu'].apply(extract_cpu_brand)
        hardware['gpu_brand'] = hardware['gpu'].apply(extract_gpu_brand)
        
        # Group hardware to get 1 row per game (take max RAM and most common CPU/GPU)
        hw_agg = hardware.groupby('id_link').agg({
            'min_ram_gb': 'max',
            'cpu_brand': lambda x: x.mode()[0] if not x.mode().empty else 'Generic/Other',
            'gpu_brand': lambda x: x.mode()[0] if not x.mode().empty else 'Integrated/Other'
        }).reset_index().rename(columns={'id_link': 'id'})

        # 6. Aggregate Social
        social_agg = social.groupby('fk_game_id').size().reset_index(name='social_score').rename(columns={'fk_game_id': 'id'})

        # 7. Merge Everything
        df = pd.merge(games, critic_agg, on='id', how='left')
        df = pd.merge(df, hw_agg, on='id', how='left')
        df = pd.merge(df, social_agg, on='id', how='left')

        # 8. Final Cleaning
        df['critic_score'] = df['critic_score'].fillna(0)
        df['min_ram_gb'] = df['min_ram_gb'].fillna(0)
        df['social_score'] = df['social_score'].fillna(0)
        df['cpu_brand'] = df['cpu_brand'].fillna('Generic/Other')
        df['gpu_brand'] = df['gpu_brand'].fillna('Integrated/Other')
        
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
            if df['price'].mean() > 200: df['price'] /= 100.0

        if 'genres' in df.columns:
            df['primary_genre'] = df['genres'].astype(str).str.replace(r"[\[\]']", "", regex=True).str.split(',').str[0].str.strip()
        
        df['full_text'] = df['description'].fillna('') + " " + df['primary_genre'].fillna('')
        
        return df.dropna(subset=['name'])

    except Exception as e:
        st.error(f"FATAL ERROR LOADING DATA: {e}")
        return pd.DataFrame()

# ==============================================================================
# 4. AI ENGINE SETUP (Using scikit-learn)
# ==============================================================================
@st.cache_resource
def setup_neural_engine(_data):
    if _data.empty: return None, None, None, None
    
    from sklearn.feature_extraction import text 
    
    # 1. EXPANDED BANNED LIST (Cleaning the pollution)
    my_banned_words = [
        'game', 'games', 'play', 'player', 'players', 
        'world', 'new', 'experience', 'best', 
        'adventure', 'action', 'story', 'time', 'use',
        'based', 'set', 'includes', 'features', 'developed', 'content', # NEW BANS
        'explore', 'different', 'way', 'make', 'like'                   # MORE BANS
    ]
    
    # 2. MERGE
    final_stop_words = list(text.ENGLISH_STOP_WORDS.union(my_banned_words))
    
    # 3. VECTORIZE
    tfidf_lda = TfidfVectorizer(
        max_df=0.5, 
        min_df=2, 
        stop_words=final_stop_words
    )
    dtm_lda = tfidf_lda.fit_transform(_data['description'].astype(str))
    
    # 4. LDA MODEL (Keeping random_state fixed helps, but data changes affect it)
    lda_model = LatentDirichletAllocation(
        n_components=5, 
        random_state=42, 
        learning_method='online',
        n_jobs=-1
    )
    lda_model.fit(dtm_lda)
    
    # 5. REC ENGINE
    tfidf_rec = TfidfVectorizer(stop_words='english', max_features=5000)
    dtm_rec = tfidf_rec.fit_transform(_data['full_text'])
    cosine_sim = cosine_similarity(dtm_rec, dtm_rec)
    
    return tfidf_lda, lda_model, dtm_rec, cosine_sim

# ==============================================================================
# 5. UI COMPONENTS
# ==============================================================================
def render_metric_card(label, value, delta=None, color="#00ffcc"):
    st.markdown(f"""
    <div class="glass-card" style="border-left: 4px solid {color}; padding: 15px;">
        <h4 style="margin:0; color:#888; font-size: 0.9rem;">{label}</h4>
        <h2 style="margin:0; color:{color}; font-size: 2rem;">{value}</h2>
        {f'<p style="margin:0; color:{color}; font-size: 0.8rem;">{delta}</p>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 6. MAIN APP EXECUTION
# ==============================================================================

# --- A. BOOT & DATA ---

# 1. Run the Animation (It will check session_state internally to only run once)
run_boot_sequence() 

if 'booted' not in st.session_state: 
    st.session_state.booted = True

# 2. Now load the actual data
df = load_data()
if df.empty: 
    st.stop()

# --- B. NAVIGATION (ONLY ONE INSTANCE HERE) ---
st.sidebar.markdown("## ⚡ SYSTEM NAV")
page = st.sidebar.radio(
    "Select Page", 
    ["Dashboard Overview", "Visual Analytics", "Neural Discovery Lab", "Data Inspector"], 
    label_visibility="collapsed",
    key="main_nav_radio" # Unique key prevents duplicate ID errors
)

# --- C. TRAIN AI ---
tfidf_lda, lda_model, dtm_rec, sim_matrix = setup_neural_engine(df)

# --- D. GLOBAL FILTER ENGINE (PERSISTENT) ---

# 1. Reset Logic
def reset_callbacks():
    st.session_state.search_term = []
    st.session_state.price_range = (0, int(df['price'].max()))
    st.session_state.score_range = (0.0, 5.0) 
    st.session_state.sel_ram = "All Systems"
    st.session_state.sel_cpu = []
    st.session_state.sel_gpu = []

# 2. Global Filter UI (Visibility Logic)
if page != "Neural Discovery Lab":
    st.markdown("## EPIC STORE TELEMETRY")
    
    # Global Search bar
    c_search, c_reset = st.columns([4, 1])
    with c_search:
        all_names = df['name'].dropna().unique().tolist()
        all_genres = df['primary_genre'].dropna().unique().tolist()
        search_options = [f"Genre: {g}" for g in all_genres] + all_names
        
        selected_tags = st.multiselect(
            "🔍 Smart Search:", 
            options=search_options, 
            key="search_term", 
            label_visibility="collapsed"
        )

    with c_reset:
        st.button("🔄 RESET", on_click=reset_callbacks, use_container_width=True)

    # Advanced Filters Expander
    with st.expander("🎛️ GLOBAL MISSION CONTROL (FILTERS)", expanded=False):
        f1, f2, f3 = st.columns(3)
        
        with f1:
            st.markdown("**💰 ECONOMICS & QUALITY**")
            max_p = int(df['price'].max()) if not df.empty else 100
            price_range = st.slider("Price ($)", 0, max_p, (0, max_p), key="price_range")
            score_range_5 = st.slider("User Rating (0-5)", 0.0, 5.0, (0.0, 5.0), step=0.5, key="score_range")

        with f2:
            st.markdown("**⚙️ CORE SPECS**")
            ram_opts = ["All Systems", "Low Spec (<4GB)", "Standard (4-8GB)", "High Spec (>8GB)"]
            sel_ram = st.selectbox("RAM Tier", ram_opts, index=0, key="sel_ram")
            
            cpu_opts = sorted(df['cpu_brand'].unique())
            sel_cpu = st.multiselect("CPU Type", cpu_opts, default=[], key="sel_cpu")

        with f3:
            st.markdown("**⚙️ GRAPHICS**")
            gpu_opts = sorted(df['gpu_brand'].unique())
            sel_gpu = st.multiselect("GPU Brand", gpu_opts, default=[], key="sel_gpu")
else:
    # If we are on Neural Lab, fetch values from session state to keep variables defined
    selected_tags = st.session_state.get('search_term', [])
    price_range = st.session_state.get('price_range', (0, 100))
    score_range_5 = st.session_state.get('score_range', (0.0, 5.0))
    sel_ram = st.session_state.get('sel_ram', "All Systems")
    sel_cpu = st.session_state.get('sel_cpu', [])
    sel_gpu = st.session_state.get('sel_gpu', [])

# --- E. APPLY FILTERS GLOBALLY ---
dff = df.copy()

# 1. Identity Logic
if selected_tags:
    sel_names = [t for t in selected_tags if not t.startswith("Genre: ")]
    sel_genres_search = [t.replace("Genre: ", "") for t in selected_tags if t.startswith("Genre: ")]
    mask_name = dff['name'].isin(sel_names) if sel_names else pd.Series(False, index=dff.index)
    mask_genre = dff['primary_genre'].isin(sel_genres_search) if sel_genres_search else pd.Series(False, index=dff.index)
    dff = dff[mask_name | mask_genre]

# 2. Economics logic
dff = dff[(dff['price'] >= price_range[0]) & (dff['price'] <= price_range[1])]
min_s = score_range_5[0] * 20
max_s = score_range_5[1] * 20
dff = dff[(dff['critic_score'] >= min_s) & (dff['critic_score'] <= max_s)]

# 3. Hardware logic (FIXED COLUMN NAME)
if sel_ram == "Low Spec (<4GB)": 
    dff = dff[dff['min_ram_gb'] < 4]
elif sel_ram == "Standard (4-8GB)": 
    dff = dff[(dff['min_ram_gb'] >= 4) & (dff['min_ram_gb'] <= 8)]
elif sel_ram == "High Spec (>8GB)": 
    dff = dff[dff['min_ram_gb'] > 8]

if sel_gpu: dff = dff[dff['gpu_brand'].isin(sel_gpu)]
if sel_cpu: dff = dff[dff['cpu_brand'].isin(sel_cpu)]

# --- F. SIDEBAR STATUS & METRICS ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 SYSTEM MONITOR")

# Data Logic
total_count = len(df)
filtered_count = len(dff)
percent_visible = (filtered_count / total_count) if total_count > 0 else 0

# Metric 1: Total DB Size
st.sidebar.metric(
    label="Master Catalog", 
    value=f"{total_count:,}", 
    help="Total games in the local Epic Store database"
)

# Metric 2: Filtered Size (with dynamic delta)
# The 'delta' shows how many games were hidden by the current filters
diff = filtered_count - total_count
st.sidebar.metric(
    label="Active Stream", 
    value=f"{filtered_count:,}", 
    delta=f"{diff:,}" if diff != 0 else "MASTER LIST",
    delta_color="normal" if diff != 0 else "off"
)

# Visual Capacity Bar (Cyberpunk style)
st.sidebar.markdown(f"""
    <div style="margin-top: -10px; margin-bottom: 5px;">
        <small style="color: #888; font-family: 'Courier New';">STREAM UTILIZATION: {percent_visible:.1%}</small>
    </div>
""", unsafe_allow_html=True)
st.sidebar.progress(percent_visible)

st.sidebar.markdown("---")
st.sidebar.markdown("<small style='color:#444;'>v3.0.2 | CONNECTION: SECURE</small>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: DASHBOARD OVERVIEW
# ==============================================================================
if page == "Dashboard Overview":
    if dff.empty:
        st.warning("No games match the global filters.")
        st.stop()

    # Metrics Calculation
    avg_p = dff['price'].mean()
    
    # FIX: Calculate Avg Score excluding 0s
    scored_games = dff[dff['critic_score'] > 0]
    if not scored_games.empty:
        avg_s_100 = scored_games['critic_score'].mean()
        avg_s_5 = avg_s_100 / 20.0 
    else:
        avg_s_5 = 0.0

    # --- 1. METRICS (The "Big BANs") ---
    m1, m2, m3, m4 = st.columns(4)
    with m1: render_metric_card("Active Games", f"{len(dff)}", f"{(len(dff)/len(df)):.1%} of DB")
    with m2: render_metric_card("Avg Price", f"${avg_p:.2f}", "Filtered", "#ff00ff")
    with m3: render_metric_card("Avg Rating", f"⭐ {avg_s_5:.1f}/5", "Rated Games Only", "#00ffcc")
    with m4: render_metric_card("Top Genre", dff['primary_genre'].mode()[0] if not dff.empty else "N/A", "Dominant", "#ffff00")
   
    # --- 2. LEADERBOARD (Moved Here) ---
    st.markdown("### 🏆 Top Rated Leaders")
    
    # Get top 5 sorted by score
    leaders = dff.nlargest(5, 'critic_score')[['name', 'primary_genre', 'price', 'critic_score', 'developer']]
    
    st.dataframe(
        leaders,
        column_config={
            "name": st.column_config.TextColumn("Game Title", width="medium"),
            "primary_genre": "Genre",
            "developer": "Developer",
            "price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "critic_score": st.column_config.ProgressColumn(
                "User Score", 
                format="%.1f", # <--- CHANGED HERE (was %d)
                min_value=0, 
                max_value=100
            ),
        },
        hide_index=True,
        use_container_width=True
    )

    # --- 3. CHARTS ---
    st.markdown("---")

    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("### Performance Matrix")
        dff['plot_size'] = dff['review_count'].fillna(1).replace(0, 1)
        
        # 1. CREATE TABS FOR SCATTER PLOTS
        tab_price, tab_ram = st.tabs(["💰 Price Impact", "💾 RAM Specs"])
        
        # --- TAB 1: PRICE VS RATING (Existing) ---
        with tab_price:
            fig = px.scatter(
                dff, 
                x='price', 
                y='critic_score', 
                color='primary_genre',
                size='plot_size', 
                hover_data=['name', 'min_ram_gb', 'cpu_brand', 'gpu_brand'],
                template='plotly_dark', 
                opacity=0.8,
                title=f"Price vs Quality (n={len(dff)})",
                labels={
                    "price": "Price ($USD)",
                    "critic_score": "User Score (0-100)",
                    "primary_genre": "Genre",
                    "min_ram_gb": "Min RAM (GB)"
                },
                color_discrete_sequence=['#00ffcc', '#ff00ff', '#ffff00', '#00bfff']
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=30, l=0, r=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- TAB 2: RAM VS RATING (New) ---
        with tab_ram:
            fig_ram = px.scatter(
                dff, 
                x='min_ram_gb', 
                y='critic_score', 
                color='primary_genre',
                size='plot_size',
                hover_data=['name', 'price', 'cpu_brand'],
                template='plotly_dark', 
                opacity=0.8,
                title=f"Hardware Requirements vs Quality",
                labels={
                    "min_ram_gb": "Minimum RAM Required (GB)",
                    "critic_score": "User Score (0-100)",
                    "primary_genre": "Genre"
                },
                color_discrete_sequence=['#ff00ff', '#00ffcc', '#ffff00', '#00bfff'] # Swapped colors for variety
            )
            fig_ram.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=30, l=0, r=0, b=0),
                xaxis=dict(tickmode='linear', tick0=0, dtick=4) # Clean 4GB steps
            )
            st.plotly_chart(fig_ram, use_container_width=True)

    with c2:
        st.markdown("### Market Share")
        counts = dff['primary_genre'].value_counts()
        if len(counts) > 6:
            top_5 = counts.head(5)
            other = pd.Series({'Other': counts.iloc[5:].sum()})
            counts = pd.concat([top_5, other])
            
        fig = px.pie(
            values=counts.values, names=counts.index, hole=0.6,
            template='plotly_dark', color_discrete_sequence=px.colors.sequential.RdPu_r
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', showlegend=False, margin=dict(t=0, b=0, l=0, r=0),
            annotations=[dict(text=f"{len(dff)}", x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True) 

# ==============================================================================
# PAGE 2: VISUAL ANALYTICS (Notebook Graphs)
# ==============================================================================
elif page == "Visual Analytics":
    st.markdown("## DEEP DIVE ANALYTICS")
    st.markdown("Advanced visualization modules extracted from the Analysis Notebook.")
    st.markdown("---")

    # --- MAIN TABS ---
    tab1, tab2, tab3 = st.tabs(["💰 Economics & Distributions", "⭐ Sentiment Analysis", "📝 Text Analytics"])

    # ==========================================================================
    # TAB 1: ECONOMICS, HARDWARE & GENRES
    # ==========================================================================
    with tab1:
        st.markdown("### Global Distributions")
        
        # 1. Global Price Histogram
        if 'price' in df.columns:
            fig_hist = px.histogram(df, x='price', nbins=40, 
                                   template='plotly_dark', 
                                   title="Global Price Distribution",
                                   color_discrete_sequence=['#00ffcc'])
            fig_hist.update_layout(bargap=0.1, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")
        st.markdown("### Distribution by Genre")
        
        # PREPARE DATA: Clean Genre Data
        # Filter out NaN and literal 'nan' strings
        valid_genres_df = df[
            df['primary_genre'].notna() & 
            (df['primary_genre'].astype(str).str.lower() != 'nan')
        ]
        
        # Identify Top 12 Genres to keep the chart readable
        top_genres_list = valid_genres_df['primary_genre'].value_counts().head(12).index
        df_filtered = valid_genres_df[valid_genres_df['primary_genre'].isin(top_genres_list)]

        # --- SUB-TABS FOR GENRE ANALYSIS ---
        sub_t1, sub_t2, sub_t3 = st.tabs(["💰 Price by Genre", "💾 RAM by Genre", "⭐ Score by Genre"])

        # 1. PRICE BY GENRE (Sorted by Median -> Q3)
        with sub_t1:
            # Calculate Median and Q3 (75th percentile)
            price_stats = df_filtered.groupby('primary_genre')['price'].agg(
                median='median', 
                q3=lambda x: x.quantile(0.75)
            )
            # Sort by Median first, then Q3 to break ties
            price_order = price_stats.sort_values(by=['median', 'q3'], ascending=False).index
            
            fig_box_p = px.box(
                df_filtered, 
                x='primary_genre', 
                y='price', 
                template='plotly_dark',
                title="Pricing Strategy (Sorted by Median + Q3)",
                color='primary_genre',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_box_p.update_traces(yhoverformat="$.2f") # Format price hover
            fig_box_p.update_layout(
                showlegend=False, 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis={'categoryorder':'array', 'categoryarray': price_order} # Apply Advanced Sorting
            )
            st.plotly_chart(fig_box_p, use_container_width=True)

        # 2. RAM BY GENRE (Sorted by Median -> Q3)
        with sub_t2:
            # --- NEW: OUTLIER CONTROL ---
            # Toggle to hide extreme values to make the chart more readable
            hide_extreme = st.toggle("🚫 Hide Extreme Outliers (> 20GB)", value=True)
            
            # Create a plotting dataframe based on the toggle
            if hide_extreme:
                df_ram_plot = df_filtered[df_filtered['min_ram_gb'] <= 20]
            else:
                df_ram_plot = df_filtered

            # Calculate Median and Q3 (Based on the Visible Data)
            ram_stats = df_ram_plot.groupby('primary_genre')['min_ram_gb'].agg(
                median='median', 
                q3=lambda x: x.quantile(0.75)
            )
            # Sort order logic
            ram_order = ram_stats.sort_values(by=['median', 'q3'], ascending=False).index
            
            fig_box_r = px.box(
                df_ram_plot, 
                x='primary_genre', 
                y='min_ram_gb', 
                template='plotly_dark',
                title=f"Hardware Demands {'(Standard Range)' if hide_extreme else '(Full Range)'}",
                color='primary_genre',
                labels={'min_ram_gb': 'Min RAM (GB)'},
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            fig_box_r.update_layout(
                showlegend=False, 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis={'categoryorder':'array', 'categoryarray': ram_order}, # Apply Sorting
                yaxis=dict(dtick=2 if hide_extreme else 4) # Tighter grid lines if zoomed in
            )
            st.plotly_chart(fig_box_r, use_container_width=True)

        # 3. SCORE BY GENRE (Sorted by Median -> Q3)
        with sub_t3:
            # Filter for rated games only
            df_scored = df_filtered[df_filtered['critic_score'] > 0]
            
            # Calculate Median and Q3
            score_stats = df_scored.groupby('primary_genre')['critic_score'].agg(
                median='median', 
                q3=lambda x: x.quantile(0.75)
            )
            score_order = score_stats.sort_values(by=['median', 'q3'], ascending=False).index
            
            fig_box_s = px.box(
                df_scored, 
                x='primary_genre', 
                y='critic_score', 
                template='plotly_dark',
                title="Critic Quality (Sorted by Performance)",
                color='primary_genre',
                labels={'critic_score': 'User Score (0-100)'},
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            
            # Format hover to 1 decimal place
            fig_box_s.update_traces(yhoverformat=".1f") 
            
            fig_box_s.update_layout(
                showlegend=False, 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis={'categoryorder':'array', 'categoryarray': score_order}, # Apply Advanced Sorting
                yaxis=dict(tickformat=".1f")
            )
            st.plotly_chart(fig_box_s, use_container_width=True)

    # ==========================================================================
    # TAB 2: SENTIMENT (Fixed Column Names)
    # ==========================================================================
    with tab2:
        st.markdown("### Rating & Quality Metrics")
        if 'critic_score' in df.columns and 'primary_genre' in df.columns:
            rated_games = df[df['critic_score'] > 0]
            
            if not rated_games.empty:
                avg_rating_genre = rated_games.groupby('primary_genre')['critic_score'].mean().sort_values(ascending=False).head(15)
                
                fig_bar = px.bar(x=avg_rating_genre.values, y=avg_rating_genre.index, orientation='h',
                                template='plotly_dark',
                                title="Average User Rating by Genre (Rated Games Only)",
                                labels={'x': 'Average Score (0-100)', 'y': 'Genre'},
                                color=avg_rating_genre.values,
                                color_continuous_scale='Viridis')
                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("No rated games found in the dataset.")

    # ==========================================================================
    # TAB 3: TEXT ANALYTICS (Fixed Column Names)
    # ==========================================================================
    with tab3:
        st.markdown("### 📝 Description Text Analysis")
        st.markdown("Length of description vs User Rating.")
        
        if 'description' in df.columns and 'critic_score' in df.columns:
            text_df = df.copy()
            text_df['desc_len'] = text_df['description'].astype(str).str.len()
            text_df = text_df[text_df['critic_score'] > 0]
            
            if not text_df.empty:
                fig_scatter = px.scatter(text_df, x='desc_len', y='critic_score', 
                                        template='plotly_dark',
                                        opacity=0.5,
                                        title="Does Description Length Correlate with Rating?",
                                        labels={'desc_len': 'Description Character Count', 'critic_score': 'Score'},
                                        color_discrete_sequence=['#ff00ff'])
                fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("No rated games available for text analysis.")

# ==============================================================================
# PAGE 3: NEURAL DISCOVERY LAB (FINAL LAYOUT)
# ==============================================================================
elif page == "Neural Discovery Lab":
    st.markdown("## 🤖 NEURAL DISCOVERY LAB")
    st.markdown("NLP engine decoding game DNA to find semantic matches.")
    st.markdown("---")

    if 'name' not in df.columns:
        st.error("Dataset missing 'name' column.")
        st.stop()

    # --- BLOCK 1: SELECTION & MATCHES ---
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("### 📡 TARGET SELECTION")

        # --- FILTER LOGIC ---
        # 1. Create a subset where 'primary_genre' is NOT NaN
        # We check both .notna() (for real missing values) AND string 'nan' (for converted missing values)
        valid_genre_df = df[
            df['primary_genre'].notna() & 
            (df['primary_genre'].astype(str).str.lower() != 'nan')
        ]
        
        # 2. Generate the game list from this specific subset
        game_list = sorted(valid_genre_df['name'].dropna().astype(str).unique())
        
        # 3. Render Selectbox
        selected_game = st.selectbox("Select Subject:", game_list, label_visibility="collapsed")
        
        if selected_game:
            idx = df[df['name'] == selected_game].index[0]
            row = df.iloc[idx]
            
            # 1. Logic for RAM formatting
            ram_val = row.get('min_ram_gb', 0)
            if ram_val >= 1:
                ram_display = f"{ram_val:.0f} GB"
            elif ram_val > 0:
                ram_display = f"{int(ram_val*1024)} MB"
            else:
                ram_display = "N/A"

            # 2. Score and Social Logic
            raw_social = row.get('social_score', 0)
            s_score = 0 if pd.isna(raw_social) else int(raw_social)
            social_display = "🟢" * min(s_score, 5) if s_score > 0 else "⚫ No Links"
            
            c_val = row.get('critic_score', 0)
            c_score = f"{c_val:.0f}" if pd.notna(c_val) and c_val > 0 else "N/A"

            # 3. THE CARD RENDERING
            st.markdown(f"""
            <div class="glass-card" style="min-height: 220px; height: auto; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <h2 style="color:#00ffcc; margin:0; font-size:1.6rem; line-height:1.2;">{selected_game}</h2>
                    <p style="color:#888; font-size:0.9rem; margin-top:5px;">{row.get('primary_genre', 'Unknown')}</p>
                </div>
                <div style="margin-top: 20px;"> <!-- Added margin top to ensure spacing on grow -->
                    <hr style="border-color:#333; margin: 10px 0;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="text-align:left;">
                            <b style="color:#ff00ff; font-size:0.8rem;">CRITIC</b><br>
                            <span style="font-size:1.2rem; color:white; font-weight:bold;">{c_score}</span>
                        </div>
                        <div style="text-align:center;">
                            <b style="color:#00ffcc; font-size:0.8rem;">MEMORY</b><br>
                            <span style="font-size:1.2rem; color:white; font-weight:bold;">{ram_display}</span>
                        </div>
                        <div style="text-align:right;">
                            <b style="color:#ff00ff; font-size:0.8rem;">SOCIAL</b><br>
                            <span style="font-size:1rem; color:white;">{social_display}</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("### 🧬 SEMANTIC MATCHES")
        if selected_game:
            sim_scores = list(enumerate(sim_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_recs = sim_scores[1:4]
            
            # Create a horizontal grid for 3 recommendations
            r1, r2, r3 = st.columns(3)
            for rank, (r_idx, score) in enumerate(top_recs):
                rec_row = df.iloc[r_idx]
                r_genre = rec_row.get('primary_genre', 'N/A')
                
                with [r1, r2, r3][rank]:
                    st.markdown(f"""
                    <div class="glass-card" style="height: 308px; border-top: 3px solid #ff00ff;">
                        <h4 style="margin:0; color:white; font-size:1rem; height:50px; overflow:hidden;">{rec_row['name']}</h4>
                        <p style="color:#888; font-size:0.8rem; margin-bottom:10px;">{r_genre}</p>
                        <h2 style="margin:0; color:#ff00ff;">{score:.1%}</h2>
                        <p style="margin:0; color:#888; font-size:0.7rem;">MATCH RATE</p>
                    </div>
                    """, unsafe_allow_html=True)

    # --- BLOCK 2: AI INSIGHTS & CHATTER ---
    st.markdown("<br>", unsafe_allow_html=True)
    col_ai, col_chat = st.columns([1, 2])

    if selected_game:
        # 1. AI GENOME CARD (Bottom Left)
        with col_ai:
            # Safe text extraction
            raw_desc = row.get('description', '')
            desc_text = str(raw_desc) if pd.notna(raw_desc) and len(str(raw_desc)) > 5 else "No description available."
            
            # Predict Topic
            input_vec = tfidf_lda.transform([desc_text])
            topic_dist = lda_model.transform(input_vec)
            top_topic = topic_dist.argmax()
            conf = topic_dist[0][top_topic]
            
            # Get actual keywords (The "DNA")
            feature_names = tfidf_lda.get_feature_names_out()
            top_words_idx = lda_model.components_[top_topic].argsort()[:-6:-1]
            topic_keywords = ", ".join([feature_names[i] for i in top_words_idx])

            # Labels
            labels = {
                0: "Creation & World",
                1: "Combat & Survival",
                2: "Discovery & Mystery", 
                3: "Action & Speed",
                4: "Narrative & Story"
            }
            label_text = labels.get(top_topic, f"Topic {top_topic}")

            # RENDER CARD (Comments removed to fix raw code display)
            st.markdown(f"""
            <div class="glass-card" style="border: 1px solid #00ffcc; min-height: 260px;">
                <p style="color:#888; font-size:0.8rem; margin:0;">AI GENOME DECODE</p>
                <h2 style="color:#00ffcc; margin:10px 0; font-size:1.6rem;">{label_text}</h2>
                <div class="confidence-bar" style="height: 15px; background-color: #222; border-radius: 10px; margin-top: 10px;">
                    <div class="confidence-fill" style="width: {conf*100}%; background-color: #00ffcc; height: 100%; border-radius: 10px;"></div>
                </div>
                <p style="text-align:right; color:#00ffcc; font-size:0.8rem; margin-top:5px;">CONFIDENCE: {conf:.1%}</p>
                <hr style="border-color:#333; margin:15px 0;">
                <p style="color:#ff00ff; font-size:0.8rem; font-weight:bold;">ACTUAL DNA (KEYWORDS):</p>
                <p style="color:#aaa; font-family:'Courier New'; font-size:0.8rem;">[{topic_keywords}]</p>
            </div>
            """, unsafe_allow_html=True)
        

        # 2. CRITIC CHATTER CONSOLE (Bottom Right)
        with col_chat:
            # Robust extraction of comments
            raw_comments = str(row.get('top_comments', ''))
            valid_comments = [c.strip() for c in raw_comments.split('|||') if len(c.strip()) > 10]
            
            if valid_comments:
                # Initialize carousel state
                if 'comment_idx' not in st.session_state:
                    st.session_state.comment_idx = 0
                
                # Boundary check (in case we switch games and new game has fewer comments)
                if st.session_state.comment_idx >= len(valid_comments):
                    st.session_state.comment_idx = 0
                
                # Header with Controls
                h1, h2, h3 = st.columns([6, 1, 1])
                with h1: 
                    st.markdown(f"<h4 style='color:#ff00ff; margin:0; font-family:Courier New;'>🗣️ CRITIC CHATTER ({st.session_state.comment_idx + 1}/{len(valid_comments)})</h4>", unsafe_allow_html=True)
                with h2: 
                    if st.button("◀", key="prev_comment"):
                        st.session_state.comment_idx = (st.session_state.comment_idx - 1) % len(valid_comments)
                with h3: 
                    if st.button("▶", key="next_comment"):
                        st.session_state.comment_idx = (st.session_state.comment_idx + 1) % len(valid_comments)
                
                # Comment Display Box
                curr_comment = valid_comments[st.session_state.comment_idx]
                st.markdown(f"""
                <div class="glass-card" style="
                    height: 200px; 
                    overflow-y: auto; 
                    background: rgba(255,0,255,0.03); 
                    border-left: 4px solid #ff00ff;
                    margin-top: 5px;
                    padding: 20px;">
                    <span style="font-size:2rem; color:#ff00ff; line-height:0; vertical-align:middle;">“</span>
                    <span style="color:#ddd; font-style:italic; font-size:1.1rem; line-height:1.6;">{curr_comment}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="glass-card" style="height: 250px; display:flex; align-items:center; justify-content:center; border: 1px dashed #444;">
                    <span style="color:#666; font-family: Courier New;">[ NO CRITIC DATA INTERCEPTED ]</span>
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 4: DATA INSPECTOR
# ==============================================================================
elif page == "Data Inspector":
    st.markdown("## 📁 DATABASE ACCESS")
    
    # --- CONTROLS ---
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"**Found {len(dff)} matches**")
    with c2:
        # Toggle to show raw data including descriptions
        show_raw = st.toggle("🔓 Unlock Full Data", value=False)

    # --- DATA TABLE ---
    if show_raw:
        # VIEW 1: RAW FULL DATA (Shows EVERYTHING including description)
        st.markdown("### Raw Data View")
        st.dataframe(dff, use_container_width=True, height=600)
    else:
        # VIEW 2: CURATED VIEW + DESCRIPTION
        st.dataframe(
            dff,
            # Added 'description' to this list
            column_order=("name", "primary_genre", "price", "critic_score", "description"), 
            column_config={
                "name": st.column_config.TextColumn("Game Title", width="medium"),
                "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "critic_score": st.column_config.ProgressColumn("Critic Score", format="%d", min_value=0, max_value=100),
                "primary_genre": st.column_config.TextColumn("Genre"),
                # This displays the "Narrative" text you wanted
                "description": st.column_config.TextColumn("Narrative / Description", width="large"), 
            },
            use_container_width=True,
            height=600,
            hide_index=True
        )

# --- DEBUG: INSPECT TOPICS ---
if st.sidebar.checkbox("Show Topic Keywords"):
    st.sidebar.markdown("### 🧠 LDA Topic Keywords")
    feature_names = tfidf_lda.get_feature_names_out()
    for topic_idx, topic in enumerate(lda_model.components_):
        top_features_ind = topic.argsort()[:-6:-1] # Top 5 words
        top_features = [feature_names[i] for i in top_features_ind]
        st.sidebar.write(f"**Topic {topic_idx}:** {', '.join(top_features)}")