import re
import emoji
import joblib
import pandas as pd
import streamlit as st
import google.generativeai as genai
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize
from dotenv import load_dotenv
import os

load_dotenv()

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TriageAI — Seller Dashboard",
    page_icon="🎯",
    layout="centered",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model      = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# ── GEMINI SETUP ──────────────────────────────────────────────────────────────
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

gemini = genai.GenerativeModel(
    model_name='models/gemini-2.5-flash',
    system_instruction="""
    คุณคือผู้ช่วยวิเคราะห์รีวิวสินค้าออนไลน์สำหรับเจ้าของร้านค้า
    ตอบเป็นภาษาไทยเสมอ
    วิเคราะห์เชิงธุรกิจ กระชับ และให้คำแนะนำที่นำไปปฏิบัติได้จริง
    """
)

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
def clean_thai(text):
    text = normalize(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[@#]\S+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s*ๆ', 'ๆ', text)
    text = text.replace('<3', 'รัก')
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── FUNCTIONS ─────────────────────────────────────────────────────────────────
def predict(text):
    text   = clean_thai(text)
    tokens = ' '.join(word_tokenize(text))
    tfidf  = vectorizer.transform([tokens])
    return model.predict(tfidf)[0]

def analyze_reviews(df):
    results = []
    for _, row in df.iterrows():
        comment = str(row.get("comment", "")).strip()
        stars   = row.get("rating_star", 0)
        if not comment:
            continue
        label = predict(comment)
        results.append({
            "comment":   comment,
            "stars":     stars,
            "sentiment": label,
        })
    return pd.DataFrame(results)

def call_gemini_dashboard(neg_comments, all_summary):
    neg_sample = [c[:100] for c in neg_comments[:30]]
    neg_text   = "\n".join(neg_sample)

    themes_prompt = f"""
    นี่คือรีวิวเชิงลบจากลูกค้า:
    {neg_text}

    สรุปเป็นหัวข้อหลัก 3-5 หัวข้อว่าลูกค้าบ่นเรื่องอะไรบ้าง
    ตอบเป็นภาษาไทย กระชับ ใช้ bullet point
    """

    conclusion_prompt = f"""
    ข้อมูลสรุปรีวิวสินค้า:
    {all_summary}

    เขียนสรุปภาพรวมและคำแนะนำสำหรับเจ้าของร้าน 3-4 ประโยค
    ตอบเป็นภาษาไทย เชิงบวกแต่ตรงไปตรงมา
    """

    themes     = gemini.generate_content(themes_prompt).text
    conclusion = gemini.generate_content(conclusion_prompt).text
    return themes, conclusion

# ── UI: HERO ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">🎯 Seller Intelligence</div>
    <div class="hero-title">Review<br><span>Dashboard</span></div>
    <div class="hero-sub">
        วิเคราะห์รีวิวสินค้าอัตโนมัติด้วย AI
        เพื่อให้เจ้าของร้านเข้าใจความรู้สึกลูกค้าและปรับปรุงร้านได้ทันที
    </div>
    <div class="hero-badges">
        <div class="badge">🤖 Gemini 2.5 Flash</div>
        <div class="badge">🇹🇭 PyThaiNLP</div>
        <div class="badge">📊 Sentiment Analysis</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── UI: UPLOAD ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">อัปโหลดไฟล์รีวิว</div>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:var(--text-dim);font-size:.88rem;margin-bottom:1rem;">'
    'อัปโหลดไฟล์ CSV ที่มีคอลัมน์ <strong style="color:var(--accent)">comment</strong> '
    'และ <strong style="color:var(--accent)">rating_star</strong>'
    '</p>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("อัปโหลด CSV", type=["csv"], label_visibility="collapsed")

col_btn, col_hint = st.columns([1, 4])
with col_btn:
    run = st.button("🔍 วิเคราะห์", type="primary", use_container_width=True)
with col_hint:
    st.markdown(
        '<p style="color:var(--muted);font-size:.82rem;margin-top:.7rem;">'
        'รองรับไฟล์ .csv เท่านั้น</p>',
        unsafe_allow_html=True,
    )

# ── UI: DASHBOARD ─────────────────────────────────────────────────────────────
if run:
    if uploaded is None:
        st.markdown("""
        <div class="result-card neutral">
            <div class="result-icon">📂</div>
            <div class="result-title">กรุณาอัปโหลดไฟล์ก่อน</div>
            <div class="result-body">อัปโหลดไฟล์ CSV ที่มีรีวิวสินค้า แล้วกดวิเคราะห์อีกครั้ง</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        try:
            raw_df = pd.read_csv(uploaded)

            if "comment" not in raw_df.columns or "rating_star" not in raw_df.columns:
                st.markdown("""
                <div class="result-card negative">
                    <div class="result-icon">⚠️</div>
                    <div class="result-title">ไฟล์ไม่ถูกต้อง</div>
                    <div class="result-body">ไฟล์ CSV ต้องมีคอลัมน์ชื่อ comment และ rating_star</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.spinner("กำลังวิเคราะห์รีวิว..."):
                    df = analyze_reviews(raw_df)

                total = len(df)
                pos   = (df['sentiment'] == 'pos').sum()
                neu   = (df['sentiment'] == 'neu').sum()
                neg   = (df['sentiment'] == 'neg').sum()
                score = round(df['stars'].mean(), 1) if total > 0 else 0.0

                st.markdown('<hr>', unsafe_allow_html=True)

                # ── SCORE ──
                card_class = 'positive' if score >= 4 else 'neutral' if score >= 3 else 'negative'
                st.markdown(f"""
                <div class="result-card {card_class}">
                    <div class="result-icon">⭐</div>
                    <div class="result-title">คะแนนรวม: {score} / 5</div>
                    <div class="result-body">จากรีวิวทั้งหมด {total} รายการ</div>
                </div>
                """, unsafe_allow_html=True)

                # ── BAR CHART ──
                st.markdown('<div class="section-label" style="margin-top:1.8rem;">สัดส่วนความรู้สึก</div>', unsafe_allow_html=True)
                chart_df = pd.DataFrame({
                    "ความรู้สึก": ["เชิงบวก 😊", "กลางๆ 😐", "เชิงลบ 😤"],
                    "จำนวน":      [int(pos), int(neu), int(neg)],
                })
                st.bar_chart(chart_df.set_index("ความรู้สึก"))

                # ── METRICS ──
                m1, m2, m3 = st.columns(3)
                m1.metric("✅ เชิงบวก", int(pos), f"{round(pos/total*100)}%")
                m2.metric("💬 กลางๆ",   int(neu), f"{round(neu/total*100)}%")
                m3.metric("⚠️ เชิงลบ",  int(neg), f"{round(neg/total*100)}%")

                # ── SAMPLE TABLE ──
                st.markdown('<div class="section-label" style="margin-top:1.8rem;">ตัวอย่างรีวิว</div>', unsafe_allow_html=True)
                sample = df[['comment', 'stars', 'sentiment']].head(10).copy()
                sample.columns = ["ความคิดเห็น", "ดาว", "ความรู้สึก"]
                sample["ความรู้สึก"] = sample["ความรู้สึก"].map({
                    "pos": "😊 บวก", "neu": "😐 กลาง", "neg": "😤 ลบ"
                })
                st.dataframe(sample, use_container_width=True, hide_index=True)

                # ── GEMINI ──
                neg_comments = df[df['sentiment'] == 'neg']['comment'].tolist()
                summary_text = f"""
                คะแนนเฉลี่ย: {score}/5
                รีวิวทั้งหมด: {total}
                เชิงบวก: {pos} ({round(pos/total*100)}%)
                กลางๆ: {neu} ({round(neu/total*100)}%)
                เชิงลบ: {neg} ({round(neg/total*100)}%)
                """

                with st.spinner("AI กำลังวิเคราะห์..."):
                    themes, conclusion = call_gemini_dashboard(neg_comments, summary_text)

                if neg_comments:
                    st.markdown('<div class="section-label" style="margin-top:1.8rem;">สิ่งที่ลูกค้าบ่น (ควรปรับปรุง)</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="ai-response">
                        <div class="ai-label">⚠️ ประเด็นเชิงลบที่พบบ่อย</div>
                        <div class="ai-text">{themes}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('<div class="section-label" style="margin-top:1.8rem;">สรุปภาพรวมและคำแนะนำ</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="ai-response">
                    <div class="ai-label">✨ บทสรุปจาก AI</div>
                    <div class="ai-text">{conclusion}</div>
                </div>
                """, unsafe_allow_html=True)

                # ── DOWNLOAD REPORT ──
                st.markdown('<div class="section-label" style="margin-top:1.8rem;">ดาวน์โหลดรายงาน</div>', unsafe_allow_html=True)

                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")

                report_text = f"""TriageAI — รายงานวิเคราะห์รีวิวสินค้า
สร้างเมื่อ: {pd.Timestamp.now().strftime("%d/%m/%Y %H:%M")}
{'='*50}

📊 สรุปภาพรวม
คะแนนเฉลี่ย: {score} / 5
รีวิวทั้งหมด: {total} รายการ
✅ เชิงบวก: {int(pos)} ({round(pos/total*100)}%)
💬 กลางๆ:   {int(neu)} ({round(neu/total*100)}%)
⚠️ เชิงลบ:  {int(neg)} ({round(neg/total*100)}%)

{'='*50}
⚠️ ประเด็นเชิงลบที่พบบ่อย
{'='*50}
{themes if neg_comments else "ไม่มีรีวิวเชิงลบ"}

{'='*50}
✨ บทสรุปและคำแนะนำจาก AI
{'='*50}
{conclusion}
"""

                st.download_button(
                    label="⬇️ ดาวน์โหลดรายงาน (.txt)",
                    data=report_text.encode("utf-8-sig"),
                    file_name=f"triage_report_{timestamp}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        except Exception as e:
            st.markdown(f"""
            <div class="result-card negative">
                <div class="result-icon">⚠️</div>
                <div class="result-title">เกิดข้อผิดพลาด</div>
                <div class="result-body">{str(e)}</div>
            </div>
            """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    TriageAI Seller Dashboard &nbsp;·&nbsp; Powered by Gemini 2.5 Flash & PyThaiNLP<br>
    <span style="opacity:.5">built with Streamlit</span>
</div>
""", unsafe_allow_html=True)