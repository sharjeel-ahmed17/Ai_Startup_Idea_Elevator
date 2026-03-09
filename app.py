import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Startup Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Bebas+Neue&family=JetBrains+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg:       #03080f;
    --card:     #080f1a;
    --border:   #0d1f35;
    --rose:     #ff2d6b;
    --cyan:     #00f5d4;
    --violet:   #bf5af2;
    --gold:     #ffd60a;
    --text:     #e2eaf5;
    --muted:    #4a6280;
    --font-head: 'Bebas Neue', sans-serif;
    --font-sans: 'Space Grotesk', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-sans) !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 65% 55% at -5% 5%,  rgba(255,45,107,.18) 0%, transparent 55%),
        radial-gradient(ellipse 55% 45% at 105% 95%, rgba(0,245,212,.14) 0%, transparent 50%),
        radial-gradient(ellipse 45% 35% at 55% 50%,  rgba(191,90,242,.08) 0%, transparent 45%),
        var(--bg) !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

.main .block-container {
    max-width: 1100px !important;
    padding: 2rem 2rem 4rem !important;
}

/* ── Hero ── */
.hero { text-align: center; padding: 4rem 0 2.8rem; position: relative; }

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: .5rem;
    font-family: var(--font-mono);
    font-size: .65rem;
    letter-spacing: .3em;
    color: var(--cyan);
    text-transform: uppercase;
    margin-bottom: 1.4rem;
    padding: .35rem 1.1rem;
    border: 1px solid rgba(0,245,212,.3);
    border-radius: 3px;
    background: rgba(0,245,212,.05);
}
.hero-eyebrow::before { content: ''; width: 6px; height: 6px; border-radius: 50%; background: var(--cyan); animation: blink 1.4s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

.hero h1 {
    font-family: var(--font-head) !important;
    font-size: clamp(4rem, 9vw, 8rem) !important;
    font-weight: 400 !important;
    letter-spacing: .06em !important;
    line-height: .95 !important;
    color: var(--text) !important;
    margin-bottom: .5rem !important;
}
.hero h1 .hl-rose   { color: var(--rose); }
.hero h1 .hl-cyan   { color: var(--cyan); }

.hero-sub {
    color: var(--muted);
    font-size: .95rem;
    font-weight: 300;
    max-width: 460px;
    margin: 1rem auto 0;
    line-height: 1.8;
    letter-spacing: .02em;
}

/* ── Divider line ── */
.hline {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), rgba(255,45,107,.4), var(--border), transparent);
    margin: 2rem 0;
}

/* ── Input card ── */
.input-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2.2rem 2.4rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.input-card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 12px;
    padding: 1px;
    background: linear-gradient(135deg, rgba(255,45,107,.5), rgba(0,245,212,.4), rgba(191,90,242,.3));
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events: none;
}
.input-card::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(255,45,107,.06) 0%, transparent 65%);
    pointer-events: none;
}

.section-label {
    font-family: var(--font-mono);
    font-size: .62rem;
    letter-spacing: .28em;
    color: var(--rose);
    text-transform: uppercase;
    margin-bottom: 1rem;
    opacity: .9;
}

/* ── Inputs ── */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    background: #040b14 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--font-sans) !important;
    font-size: .92rem !important;
    font-weight: 300 !important;
    padding: .85rem 1.1rem !important;
    transition: border-color .2s, box-shadow .2s !important;
    caret-color: var(--cyan) !important;
}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 3px rgba(0,245,212,.1), 0 0 20px rgba(0,245,212,.06) !important;
}
[data-testid="stTextArea"] label,
[data-testid="stTextInput"] label {
    color: var(--muted) !important;
    font-family: var(--font-mono) !important;
    font-size: .67rem !important;
    letter-spacing: .16em !important;
    text-transform: uppercase !important;
}

/* ── Button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--rose) 0%, #c4006a 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-head) !important;
    font-size: 1.15rem !important;
    font-weight: 400 !important;
    letter-spacing: .12em !important;
    padding: .85rem 2.4rem !important;
    cursor: pointer !important;
    transition: transform .15s, box-shadow .15s, filter .15s !important;
    box-shadow: 0 4px 24px rgba(255,45,107,.35), 0 0 0 0 rgba(255,45,107,0) !important;
    width: 100% !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    filter: brightness(1.1) !important;
    box-shadow: 0 8px 36px rgba(255,45,107,.5), 0 0 60px rgba(255,45,107,.15) !important;
}
[data-testid="stButton"] > button:active { transform: translateY(0) !important; }

/* ── Results header ── */
.results-header {
    font-family: var(--font-head);
    font-size: 2.4rem;
    letter-spacing: .1em;
    color: var(--text);
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    gap: .7rem;
}
.results-header .dot { color: var(--rose); }
.results-header .sub {
    font-family: var(--font-mono);
    font-size: .65rem;
    letter-spacing: .2em;
    color: var(--muted);
    font-weight: 300;
    margin-left: .4rem;
    align-self: flex-end;
    margin-bottom: .4rem;
}

/* ── Result cards ── */
.result-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem 1.7rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: transform .2s, box-shadow .2s;
}
.result-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,0,0,.4);
}
.result-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 3px;
    border-radius: 10px 0 0 10px;
}
.rc-rose::after   { background: linear-gradient(180deg, var(--rose), transparent); }
.rc-cyan::after   { background: linear-gradient(180deg, var(--cyan), transparent); }
.rc-violet::after { background: linear-gradient(180deg, var(--violet), transparent); }
.rc-gold::after   { background: linear-gradient(180deg, var(--gold), transparent); }

/* card inner glow */
.rc-rose   { box-shadow: inset 0 0 40px rgba(255,45,107,.04); }
.rc-cyan   { box-shadow: inset 0 0 40px rgba(0,245,212,.04); }
.rc-violet { box-shadow: inset 0 0 40px rgba(191,90,242,.04); }
.rc-gold   { box-shadow: inset 0 0 40px rgba(255,214,10,.04); }

.rc-title {
    font-family: var(--font-mono);
    font-size: .6rem;
    letter-spacing: .22em;
    text-transform: uppercase;
    margin-bottom: .7rem;
    padding-left: .15rem;
}
.rct-rose   { color: var(--rose); }
.rct-cyan   { color: var(--cyan); }
.rct-violet { color: var(--violet); }
.rct-gold   { color: var(--gold); }

.rc-body {
    font-family: var(--font-sans);
    font-size: .9rem;
    line-height: 1.8;
    color: var(--text);
    font-weight: 300;
    padding-left: .15rem;
}

/* ── Pills ── */
.pill-row { display: flex; flex-wrap: wrap; gap: .4rem; padding-left: .15rem; }
.pill {
    display: inline-block;
    padding: .28rem .8rem;
    border-radius: 4px;
    font-size: .72rem;
    font-family: var(--font-mono);
    letter-spacing: .05em;
    font-weight: 400;
}
.p-rose   { background: rgba(255,45,107,.1);  color: #ff6b92;  border: 1px solid rgba(255,45,107,.25); }
.p-cyan   { background: rgba(0,245,212,.08);  color: var(--cyan);    border: 1px solid rgba(0,245,212,.2); }
.p-violet { background: rgba(191,90,242,.1);  color: #d48af7;  border: 1px solid rgba(191,90,242,.25); }
.p-gold   { background: rgba(255,214,10,.08); color: var(--gold);    border: 1px solid rgba(255,214,10,.22); }

/* ── Info strip ── */
.info-strip {
    display: flex;
    align-items: center;
    gap: .6rem;
    background: rgba(0,245,212,.04);
    border: 1px solid rgba(0,245,212,.18);
    border-radius: 8px;
    padding: .85rem 1.2rem;
    font-family: var(--font-mono);
    font-size: .72rem;
    color: var(--cyan);
    letter-spacing: .08em;
    margin-bottom: 2rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #0d1f35; border-radius: 3px; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--font-mono) !important;
    font-size: .75rem !important;
    color: var(--muted) !important;
    letter-spacing: .1em !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Pydantic model & parser
# ─────────────────────────────────────────────
class Analyzer(BaseModel):
    market_potential: str
    competitors: List[str]
    monetization_strategy: str
    mvp_features: List[str]
    risks: List[str]


parser = PydanticOutputParser(pydantic_object=Analyzer)
format_instructions = parser.get_format_instructions()

prompt_template = """
You are an expert startup analyst.

Analyze the following startup concept.

Startup Idea: {startup_idea}
Target Users: {target_user}
Country: {country}

Your task is to evaluate the startup and produce a structured analysis.

Focus on:
- realistic market opportunities
- known competitors
- practical monetization
- MVP features for first launch
- major startup risks

{format_instructions}

Return ONLY valid JSON.
"""

prompt = PromptTemplate(
    input_variables=["startup_idea", "target_user", "country"],
    partial_variables={"format_instructions": format_instructions},
    template=prompt_template,
)


# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI-POWERED ANALYSIS</div>
    <h1><span class="hl-rose">STARTUP</span><br><span class="hl-cyan">ANALYZER</span></h1>
    <p class="hero-sub">Validate your idea in seconds — market potential, competition, risks & MVP strategy.</p>
</div>
<div class="hline"></div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Input card
# ─────────────────────────────────────────────
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">▸ Your Startup Brief</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    startup_idea = st.text_area(
        "Startup Idea",
        placeholder="e.g. An AI-powered personal finance app that auto-categorises expenses and suggests savings goals…",
        height=130,
    )
with col2:
    target_users = st.text_input("Target Audience", placeholder="e.g. Small business owners, social media managers")
    country = st.text_input("Country / Market", placeholder="e.g. Pakistan")

analyze_btn = st.button("⚡  ANALYZE MY STARTUP", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Analysis logic
# ─────────────────────────────────────────────
if analyze_btn:
    if not startup_idea.strip():
        st.warning("Please enter your startup idea first.")
    elif not target_users.strip():
        st.warning("Please enter your target audience.")
    elif not country.strip():
        st.warning("Please enter a country / market.")
    else:
        st.markdown("""
        <div class="info-strip">
            <span style="width:7px;height:7px;border-radius:50%;background:#00f5d4;flex-shrink:0;animation:blink 1.4s ease-in-out infinite;display:inline-block"></span>
            ANALYZING YOUR STARTUP WITH MISTRAL LARGE — THIS MAY TAKE A FEW SECONDS…
        </div>
        """, unsafe_allow_html=True)

        with st.spinner(""):
            try:
                model = ChatMistralAI(
                    model="mistral-large-latest",
                    temperature=0.7,
                )

                filled_prompt = prompt.format(
                    startup_idea=startup_idea,
                    target_user=target_users,
                    country=country,
                )

                response = model.invoke(filled_prompt)
                raw = response.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[-1]
                    raw = raw.rsplit("```", 1)[0].strip()

                result: Analyzer = parser.parse(raw)

                st.markdown('<div class="hline"></div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="results-header">
                    <span class="dot">◆</span> ANALYSIS COMPLETE
                    <span class="sub">// 5 INSIGHTS GENERATED</span>
                </div>""", unsafe_allow_html=True)

                # Row 1 — Market + Monetization
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class="result-card rc-rose">
                        <div class="rc-title rct-rose">Market Potential</div>
                        <div class="rc-body">{result.market_potential}</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="result-card rc-cyan">
                        <div class="rc-title rct-cyan">Monetization Strategy</div>
                        <div class="rc-body">{result.monetization_strategy}</div>
                    </div>""", unsafe_allow_html=True)

                # Competitors — full width
                comp_pills = "".join(f'<span class="pill p-violet">{c}</span>' for c in result.competitors)
                st.markdown(f"""
                <div class="result-card rc-violet">
                    <div class="rc-title rct-violet">Key Competitors</div>
                    <div class="pill-row">{comp_pills}</div>
                </div>""", unsafe_allow_html=True)

                # Row 3 — MVP + Risks
                c3, c4 = st.columns(2)
                with c3:
                    mvp_pills = "".join(f'<span class="pill p-cyan">✓ {f}</span>' for f in result.mvp_features)
                    st.markdown(f"""
                    <div class="result-card rc-cyan">
                        <div class="rc-title rct-cyan">MVP Features</div>
                        <div class="pill-row">{mvp_pills}</div>
                    </div>""", unsafe_allow_html=True)
                with c4:
                    risk_pills = "".join(f'<span class="pill p-rose">⚠ {r}</span>' for r in result.risks)
                    st.markdown(f"""
                    <div class="result-card rc-rose">
                        <div class="rc-title rct-rose">Key Risks</div>
                        <div class="pill-row">{risk_pills}</div>
                    </div>""", unsafe_allow_html=True)

                with st.expander("View raw JSON output"):
                    st.code(result.model_dump_json(indent=2), language="json")

            except Exception as e:
                st.error(f"Error: {e}")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:3.5rem 0 1rem;font-family:'JetBrains Mono',monospace;
     font-size:.62rem;color:#0d1f35;letter-spacing:.2em;">
    POWERED BY MISTRAL LARGE &nbsp;·&nbsp; BUILT WITH STREAMLIT
</div>
""", unsafe_allow_html=True)