
import streamlit as st
import torch
import re
import joblib
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModel

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)

st.set_page_config(page_title="IT Ticket Classifier", page_icon="🎫", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [data-testid="stApp"] {
    background: #0d1117 !important; color: #e6edf3 !important;
    font-family: 'Inter', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.hero { font-size: 2.2rem; font-weight: 800;
         background: linear-gradient(135deg, #58a6ff, #bc8cff);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.sub  { font-size: 0.8rem; color: #8b949e; letter-spacing: 2px;
         text-transform: uppercase; margin-bottom: 28px; }
.card { background: #161b22; border: 1px solid #30363d;
         border-left: 4px solid #58a6ff;
         border-radius: 10px; padding: 20px 24px; margin-bottom: 12px; }
.label { font-size: 0.65rem; color: #8b949e; letter-spacing: 2px;
          text-transform: uppercase; margin-bottom: 4px; }
.value { font-size: 1.6rem; font-weight: 800; color: #58a6ff; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class MultiTaskTransformerNetwork(torch.nn.Module):
        def __init__(self, num_d, num_p):
            super().__init__()
            self.encoder = AutoModel.from_pretrained("distilbert-base-uncased")
            self.dropout = torch.nn.Dropout(0.2)
            self.d_head  = torch.nn.Linear(self.encoder.config.hidden_size, num_d)
            self.p_head  = torch.nn.Linear(self.encoder.config.hidden_size, num_p)
        def forward(self, ids, mask):
            cls = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]
            return self.d_head(self.dropout(cls)), self.p_head(self.dropout(cls))
    model = MultiTaskTransformerNetwork(4, 4).to(device)
    model.load_state_dict(torch.load("multitask_it_transformer.pt", map_location=device), strict=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return model, tokenizer, device

model, tokenizer, device = load_model()

class_names  = ['Application', 'Network', 'OS', 'Security']
priority_map = {0: 'Critical', 1: 'High', 2: 'Low', 3: 'Medium'}
routing_map  = {'Network': 'Network Infrastructure Team', 'OS': 'Operating Systems Support Team', 'Application': 'Application Support Team', 'Security': 'Security Incident Response Team'}
stop_words   = set(stopwords.words("english"))
lemmatizer   = WordNetLemmatizer()

all_kw = {"vpn","wi-fi","wifi","dropped packets","dns","packet","router","latency","dhcp",
           "switch","lan","connectivity","ping","network","windows","linux","ubuntu",
           "blue screen","bsod","driver","kernel","boot","crash","macos","reboot",
           "software","password reset","access denied","login","sap","outlook","bug",
           "excel","teams","app","malware","phishing","unauthorized","suspicious",
           "virus","ransomware","breach","compromised"}

def decontaminate(text):
    for kw in sorted(all_kw, key=len, reverse=True):
        pattern = r"(?<!\w)" + re.escape(kw) + r"(?!\w)"
        text = re.compile(pattern, re.IGNORECASE).sub("[DOMAIN_KEYWORD]", text)
    return text

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 2
    ]
    return " ".join(tokens)

def classify(raw_text):
    processed = preprocess_text(raw_text)
    model_text = decontaminate(processed)
    enc     = tokenizer(model_text, max_length=128, padding="max_length",
                        truncation=True, return_tensors="pt")
    with torch.no_grad():
        d_l, p_l = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    d_probs = torch.softmax(d_l, dim=1)[0].cpu().numpy()
    p_idx   = torch.argmax(p_l, dim=1).item()
    d_idx   = int(np.argmax(d_probs))
    domain  = class_names[d_idx]
    return {
        "domain":     domain,
        "priority":   priority_map.get(p_idx, "Medium"),
        "team":       routing_map.get(domain, "IT Support Team"),
        "confidence": round(float(d_probs[d_idx]) * 100, 1),
        "all_probs":  {class_names[i]: round(float(p)*100,1)
                        for i, p in enumerate(d_probs)}
    }

st.markdown("<div class='hero'>IT Ticket Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Multi-Task DistilBERT • Domain + Priority Routing</div>",
            unsafe_allow_html=True)

samples = [
    "Our remote VPN tunnel is dropping packets intermittently since 9am.",
    "System crashes with a memory dump on every reboot since the update.",
    "Users cannot log into the SAP portal — access denied error on all accounts.",
    "An employee received a phishing email and may have clicked the link.",
]

if "ticket_text" not in st.session_state:
    st.session_state.ticket_text = ""

col1, col2 = st.columns([1.2, 1], gap="large")
with col1:
    ticket = st.text_area("Paste raw ticket text:", height=160,
                          placeholder="Describe the IT issue here...",
                          key="ticket_text")
    run    = st.button("🚀 Classify Ticket", type="primary")
    st.markdown("<div style='margin-top:16px;font-size:0.65rem;color:#8b949e;"
                "letter-spacing:2px;text-transform:uppercase;'>Quick Examples</div>",
                unsafe_allow_html=True)
    for s in samples:
        if st.button(s[:58] + "...", key=s):
            st.session_state.ticket_text = s
            st.rerun()

with col2:
    if run and ticket.strip():
        r = classify(ticket)
        pcol = {"Critical":"#f85149","High":"#d29922","Medium":"#58a6ff","Low":"#3fb950"}.get(r["priority"],"#58a6ff")
        st.markdown(f"""
        <div class="card"><div class="label">Domain</div>
          <div class="value">📌 {r["domain"]}</div></div>
        <div class="card"><div class="label">Routed To</div>
          <div style="font-size:1rem;font-weight:600;color:#e6edf3;">📤 {r["team"]}</div></div>
        <div class="card"><div class="label">Priority</div>
          <div style="font-size:1.4rem;font-weight:800;color:{pcol};">⚡ {r["priority"]}</div></div>
        <div class="card"><div class="label">Confidence</div>
          <div class="value">{r["confidence"]}%</div></div>
        """, unsafe_allow_html=True)
        st.markdown("**Domain Probability Breakdown:**")
        for dom, prob in sorted(r["all_probs"].items(), key=lambda x: -x[1]):
            st.progress(prob / 100, text=f"{dom}: {prob}%")
    elif run:
        st.warning("Please enter a ticket description first.")
    else:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;height:260px;border:1px dashed #30363d;
                    border-radius:10px;color:#8b949e;">
            <div style="font-size:2.5rem">🎫</div>
            <div style="margin-top:8px">Awaiting ticket input</div>
        </div>""", unsafe_allow_html=True)
