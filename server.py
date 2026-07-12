import os
import re
import tempfile
import joblib
import numpy as np
from flask import Flask, jsonify, render_template_string, request

# ---------------------------------------------------------------------------
# Whisper (optional — loaded lazily on first audio request)
# ---------------------------------------------------------------------------
_whisper_model = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            ffmpeg_path = r"C:\Users\LOQ\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
            os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")
            _whisper_model = whisper.load_model("base", device="cpu")
        except ImportError:
            raise RuntimeError("openai-whisper не установлен. Запусти: pip install openai-whisper")
    return _whisper_model


# ---------------------------------------------------------------------------
# Model loader — results_new2 қалтасына бейімделген
# ---------------------------------------------------------------------------
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results_new2")

def load_models():
    """results_new2/ қалтасынан барлық модельдерді жүктейді."""
    loaded = {}

    # ── 1. CNN-LSTM (негізгі үздік модель) ──────────────────────────────────
    cnn_path = os.path.join(RESULTS_DIR, "best_model.keras")
    if not os.path.exists(cnn_path):
        # fallback: model_CNN_LSTM_optimized.keras
        cnn_path = os.path.join(RESULTS_DIR, "model_CNN_LSTM_optimized.keras")

    if os.path.exists(cnn_path):
        try:
            import tensorflow as tf
            loaded["cnn_lstm"] = tf.keras.models.load_model(cnn_path)
            print(f"✅ CNN-LSTM жүктелді: {cnn_path}")
        except Exception as e:
            print(f"⚠️  CNN-LSTM жүктелмеді: {e}")
    else:
        raise FileNotFoundError(f"CNN-LSTM модель табылмады: {cnn_path}")

    # ── 2. Keras Tokenizer ───────────────────────────────────────────────────
    for tok_name in ["best_model_tokenizer.pkl", "keras_tokenizer.pkl"]:
        tok_path = os.path.join(RESULTS_DIR, tok_name)
        if os.path.exists(tok_path):
            loaded["keras_tokenizer"] = joblib.load(tok_path)
            print(f"✅ Keras Tokenizer жүктелді: {tok_path}")
            break
    if "keras_tokenizer" not in loaded:
        raise FileNotFoundError("Keras tokenizer табылмады (best_model_tokenizer.pkl / keras_tokenizer.pkl)")

    # ── 3. NB + ruBERT ───────────────────────────────────────────────────────
    nb_path = os.path.join(RESULTS_DIR, "model_NB_BERT.pkl")
    if os.path.exists(nb_path):
        loaded["nb_rubert"] = joblib.load(nb_path)
        print(f"✅ NB+ruBERT жүктелді: {nb_path}")

    # ── 4. ruBERT Encoder ────────────────────────────────────────────────────
    enc_path = os.path.join(RESULTS_DIR, "rubert_encoder.pkl")
    if os.path.exists(enc_path):
        loaded["rubert_encoder"] = joblib.load(enc_path)
        print(f"✅ ruBERT Encoder жүктелді: {enc_path}")

    # ── 5. ruBERT + Ensemble ─────────────────────────────────────────────────
    ens_path = os.path.join(RESULTS_DIR, "model_SBERT_Ensemble.pkl")
    if os.path.exists(ens_path):
        loaded["rubert_ensemble"] = joblib.load(ens_path)
        print(f"✅ ruBERT+Ensemble жүктелді: {ens_path}")

    return loaded


print(f"📂 Модельдер қалтасы: {RESULTS_DIR}")
MODELS = load_models()
MODEL_NAME = "CNN-LSTM"   # негізгі үздік модель
print(f"🚀 Негізгі модель: {MODEL_NAME}")


# ---------------------------------------------------------------------------
# Fraud маркерлері
# ---------------------------------------------------------------------------
FRAUD_KEYWORDS = [
    "банк", "карта", "счет", "заблокирован", "блокировка", "подозрительн",
    "операция", "перевод", "код", "пароль", "cvv", "pin", "пин",
    "верификация", "подтвердит", "звоним", "служба безопасности",
    "мошенник", "взлом", "немедленно", "срочно", "полиция",
    "следователь", "прокурат", "арест", "задержан", "штраф",
    "выигр", "приз", "лотерея", "акци", "бесплатно",
    "кредит", "долг", "займ", "просроч",
    "ваш аккаунт", "ваша карта", "личный кабинет",
    "картаңыз", "блокталды", "верификация", "аударым",
]

# ---------------------------------------------------------------------------
# Утилиталар
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ\s]", "", text)
    return text.strip()

def find_markers(text: str) -> list:
    text_lower = text.lower()
    found = [kw for kw in FRAUD_KEYWORDS if kw in text_lower]
    return list(set(found))[:10]

def get_suspicious_segments(text: str) -> list:
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 10]
    if not sentences:
        return []
    segments = []
    for sent in sentences:
        score = sum(1 for kw in FRAUD_KEYWORDS if kw in sent.lower())
        if score > 0:
            segments.append({"text": sent, "score": score})
    segments.sort(key=lambda x: x["score"], reverse=True)
    return segments[:3]


# ---------------------------------------------------------------------------
# Болжау — CNN-LSTM негізгі, fallback: ruBERT+Ensemble → NB+ruBERT
# ---------------------------------------------------------------------------
MAX_LEN = 128   # CNN-LSTM үшін max sequence length

def predict_cnn_lstm(text: str) -> dict:
    """CNN-LSTM арқылы болжау."""
    tokenizer = MODELS["keras_tokenizer"]
    model     = MODELS["cnn_lstm"]

    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = tokenizer.texts_to_sequences([text])
    X   = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    proba = model.predict(X, verbose=0)[0]

    # Binary: shape (1,) немесе (2,)
    if proba.shape[0] == 1:
        fraud_prob = float(proba[0])
        pred       = 1 if fraud_prob >= 0.5 else 0
    else:
        fraud_prob = float(proba[1])
        pred       = int(np.argmax(proba))

    return {
        "predicted_class_raw": pred,
        "fraud_probability":   fraud_prob,
        "normal_probability":  1.0 - fraud_prob,
        "model_used":          "CNN-LSTM",
    }


def predict_rubert_ensemble(text: str) -> dict:
    """ruBERT + Ensemble арқылы болжау."""
    encoder = MODELS["rubert_encoder"]
    model   = MODELS["rubert_ensemble"]
    emb     = encoder.encode([text], convert_to_numpy=True)
    pred    = model.predict(emb)[0]
    proba   = model.predict_proba(emb)[0]
    fraud_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
    return {
        "predicted_class_raw": int(pred),
        "fraud_probability":   fraud_prob,
        "normal_probability":  1.0 - fraud_prob,
        "model_used":          "ruBERT + Ensemble",
    }


def predict_nb_rubert(text: str) -> dict:
    """NB + ruBERT арқылы болжау."""
    encoder = MODELS.get("rubert_encoder")
    model   = MODELS["nb_rubert"]
    if encoder is not None:
        X = encoder.encode([text], convert_to_numpy=True)
    else:
        # fallback: TF-IDF style sparse (жоқ болса қате шығады)
        raise RuntimeError("rubert_encoder жүктелмеген, NB+ruBERT қолдана алмайды.")
    pred    = model.predict(X)[0]
    proba   = model.predict_proba(X)[0]
    fraud_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
    return {
        "predicted_class_raw": int(pred),
        "fraud_probability":   fraud_prob,
        "normal_probability":  1.0 - fraud_prob,
        "model_used":          "NB + ruBERT",
    }


def predict(text: str) -> dict:
    """Негізгі модель: CNN-LSTM → ruBERT+Ensemble → NB+ruBERT."""
    cleaned = clean_text(text)

    if "cnn_lstm" in MODELS and "keras_tokenizer" in MODELS:
        try:
            return predict_cnn_lstm(cleaned)
        except Exception as e:
            print(f"⚠️ CNN-LSTM қате: {e}. ruBERT+Ensemble-ге ауысамыз.")

    if "rubert_encoder" in MODELS and "rubert_ensemble" in MODELS:
        try:
            return predict_rubert_ensemble(cleaned)
        except Exception as e:
            print(f"⚠️ ruBERT+Ensemble қате: {e}. NB+ruBERT-ке ауысамыз.")

    if "nb_rubert" in MODELS:
        return predict_nb_rubert(cleaned)

    raise RuntimeError("Ешқандай модель болжау жасай алмады.")


# ---------------------------------------------------------------------------
# Толық анализ
# ---------------------------------------------------------------------------
def analyze_call(transcript: str) -> dict:
    result     = predict(transcript)
    fraud_prob = result["fraud_probability"]

    if fraud_prob >= 0.75:
        predicted_class = "fraud"
        risk_level      = "HIGH"
        recommendation  = "⛔ Дереу қоңырауды тоқтатыңыз! Бұл алаяқтық белгілері бар. Ешқандай деректер бермеңіз."
    elif fraud_prob >= 0.45:
        predicted_class = "suspicious"
        risk_level      = "MEDIUM"
        recommendation  = "⚠️ Абай болыңыз. Жеке деректер пен кодтарды бермеңіз. Банкке тікелей хабарласыңыз."
    else:
        predicted_class = "normal"
        risk_level      = "LOW"
        recommendation  = "✅ Қоңырау қалыпты болып көрінеді. Дегенмен, жеке деректерді бермеуге тырысыңыз."

    markers  = find_markers(transcript)
    segments = get_suspicious_segments(transcript)

    reasons = []
    if fraud_prob >= 0.5:
        if markers:
            reasons.append(f"Алаяқтыққа тән сөздер табылды: {', '.join(markers[:5])}")
        if len(transcript) < 50:
            reasons.append("Мәтін өте қысқа — типтік SMS алаяқтығы")
        reasons.append(f"Модель fraud ықтималдығын {fraud_prob*100:.1f}% деп бағалады")
    else:
        reasons.append(f"Алаяқтыққа тән белгілер аз табылды ({len(markers)} сөз)")
        reasons.append(f"Модель normal ықтималдығын {(1-fraud_prob)*100:.1f}% деп бағалады")

    return {
        "predicted_class":     predicted_class,
        "fraud_probability":   round(fraud_prob, 4),
        "normal_probability":  round(1.0 - fraud_prob, 4),
        "risk_level":          risk_level,
        "recommendation":      recommendation,
        "markers":             markers,
        "decision_reasons":    reasons,
        "suspicious_segments": segments,
        "model_used":          result["model_used"],
        "features": {
            "scenario_type": "fraud_call" if predicted_class == "fraud" else "normal",
            "channel":       "call/sms",
        }
    }


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

INDEX_HTML = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fraud Call Analyzer</title>
  <style>
    :root {
      --bg: #f4efe6; --panel: #fffaf2; --ink: #1e1a16;
      --accent: #1f6f78; --accent-2: #d96c3f;
      --border: #d9ccb8; --ok: #2d7d46; --bad: #b63c31;
      --warn: #b17a16; --muted: #6f655c;
    }
    * { box-sizing: border-box; }
    body { margin:0; font-family:Georgia,"Times New Roman",serif; color:var(--ink);
      background: radial-gradient(circle at top right,#f8d9b8 0,transparent 22%),
                  radial-gradient(circle at left bottom,#d6ebe4 0,transparent 24%), var(--bg); }
    .wrap { max-width:980px; margin:32px auto; padding:24px; }
    .card { background:var(--panel); border:1px solid var(--border); border-radius:20px;
      padding:24px; box-shadow:0 18px 60px rgba(80,57,36,.08); }
    h1 { margin:0 0 8px; font-size:42px; }
    p  { color:var(--muted); margin:0 0 20px; font-size:18px; }
    .tabs { display:flex; gap:8px; margin-bottom:18px; }
    .tab-btn { border:1px solid var(--border); border-radius:999px; padding:8px 20px;
      font:inherit; font-size:15px; background:transparent; color:var(--muted); cursor:pointer; }
    .tab-btn.active { background:var(--accent); color:#fff; border-color:var(--accent); font-weight:700; }
    .tab-pane { display:none; } .tab-pane.active { display:block; }
    textarea { width:100%; min-height:240px; resize:vertical; border-radius:16px;
      border:1px solid var(--border); padding:16px; font:inherit; font-size:16px; }
    .drop-zone { border:2px dashed var(--border); border-radius:16px; padding:40px 24px;
      text-align:center; background:#fff; cursor:pointer; transition:.2s; }
    .drop-zone:hover,.drop-zone.drag-over { border-color:var(--accent); background:#f0fafa; }
    .drop-zone input[type=file] { display:none; }
    .drop-zone .dz-icon { font-size:42px; } .drop-zone .dz-label { font-size:17px; }
    .drop-zone .dz-sub { font-size:14px; color:var(--muted); }
    .drop-zone .dz-chosen { margin-top:12px; font-size:14px; font-weight:700; color:var(--accent); }
    .whisper-note { font-size:13px; color:var(--muted); margin-top:10px; }
    .row { display:flex; gap:12px; align-items:center; margin-top:16px; flex-wrap:wrap; }
    button.primary { border:0; border-radius:999px; padding:14px 22px; font-size:16px;
      font-weight:700; background:linear-gradient(135deg,var(--accent),#2d8d97);
      color:white; cursor:pointer; }
    button.primary:disabled { opacity:.6; cursor:default; }
    .hint { font-size:14px; color:var(--muted); }
    .transcript-preview { display:none; margin-top:14px; }
    .transcript-preview label { font-size:13px; text-transform:uppercase; color:var(--muted); }
    .transcript-preview textarea { min-height:120px; margin-top:6px; }
    .result { margin-top:24px; display:none; border-top:1px solid var(--border); padding-top:20px; }
    .pill { display:inline-block; padding:8px 14px; border-radius:999px; font-weight:700; font-size:15px; margin-right:8px; }
    .fraud { background:#fde2df; color:var(--bad); }
    .normal { background:#def3e3; color:var(--ok); }
    .suspicious { background:#fff0cf; color:var(--warn); }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; margin-top:16px; }
    .box { padding:14px; border:1px solid var(--border); border-radius:14px; background:#fff; }
    .label { font-size:13px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }
    .value { margin-top:6px; font-size:18px; font-weight:700; word-break:break-word; }
    ul { margin:10px 0 0; padding-left:20px; }
    pre { white-space:pre-wrap; font-size:14px; background:#fff; border:1px solid var(--border);
      border-radius:14px; padding:14px; margin-top:14px; }
    .progress-wrap { display:none; margin-top:12px; }
    .progress-bar-bg { background:var(--border); border-radius:999px; height:8px; overflow:hidden; }
    .progress-bar-fill { height:100%; width:0%; background:linear-gradient(90deg,var(--accent),#2d8d97);
      border-radius:999px; transition:width .3s; }
    .progress-label { font-size:13px; color:var(--muted); margin-top:6px; }
    .model-badge { display:inline-block; background:#e8f5e9; color:#2d7d46;
      border-radius:8px; padding:4px 10px; font-size:13px; margin-bottom:16px; }
  </style>
</head>
<body>
<div class="wrap"><div class="card">
  <h1>Fraud Call Analyzer</h1>
  <p>Алаяқтық қоңырауларды анықтау — мәтін немесе аудио жүктеңіз.</p>
  <div class="model-badge">🤖 Негізгі модель: {{ model_name }} &nbsp;|&nbsp; 📂 {{ results_dir }}</div>

  <div class="tabs">
    <button class="tab-btn active" data-tab="text">✏️ Транскрипт</button>
    <button class="tab-btn"        data-tab="audio">🎙 Аудио (Whisper)</button>
  </div>

  <div class="tab-pane active" id="tab-text">
    <textarea id="transcript" placeholder="Например: Здравствуйте, я звоню из банка..."></textarea>
    <div class="row">
      <button class="primary" id="analyzeTextBtn">Анализировать</button>
    </div>
  </div>

  <div class="tab-pane" id="tab-audio">
    <div class="drop-zone" id="dropZone">
      <input type="file" id="audioFile" accept="audio/*,video/mp4,.mp3,.wav,.ogg,.m4a,.flac,.mp4,.webm">
      <div class="dz-icon">🎧</div>
      <div class="dz-label">Перетащи аудиофайл сюда или нажми для выбора</div>
      <div class="dz-sub">MP3, WAV, M4A, OGG, FLAC, MP4 · до 50 МБ</div>
      <div class="dz-chosen" id="chosenFile"></div>
    </div>
    <p class="whisper-note">Файл Whisper арқылы транскрибацияланады (тіл: орыс). Анализден бұрын мәтінді өзгертуге болады.</p>
    <div class="progress-wrap" id="progressWrap">
      <div class="progress-bar-bg"><div class="progress-bar-fill" id="progressFill"></div></div>
      <div class="progress-label" id="progressLabel">Жүктелуде...</div>
    </div>
    <div class="transcript-preview" id="transcriptPreview">
      <label>Транскрипт (редактировать можно)</label>
      <textarea id="whisperText"></textarea>
    </div>
    <div class="row">
      <button class="primary" id="transcribeBtn">🎙 Транскрибировать</button>
      <button class="primary" id="analyzeAudioBtn" style="display:none; background:linear-gradient(135deg,#a0522d,var(--accent-2));">
        Анализировать
      </button>
    </div>
  </div>

  <div class="result" id="result">
    <div id="headline"></div>
    <div class="grid">
      <div class="box"><div class="label">Fraud Probability</div><div class="value" id="fraudProbability">-</div></div>
      <div class="box"><div class="label">Risk Level</div>      <div class="value" id="riskLevel">-</div></div>
      <div class="box"><div class="label">Scenario</div>        <div class="value" id="scenarioType">-</div></div>
      <div class="box"><div class="label">Model Used</div>      <div class="value" id="modelUsed">-</div></div>
    </div>
    <div class="box" style="margin-top:16px;"><div class="label">Маркерлер</div><ul id="markers"></ul></div>
    <div class="box" style="margin-top:16px;">
      <div class="label">Ұсыныс</div>
      <div class="value" id="recommendation" style="font-size:16px;font-weight:500;"></div>
    </div>
    <div class="box" style="margin-top:16px;"><div class="label">Неге солай шешті?</div><ul id="reasons"></ul></div>
    <div class="box" style="margin-top:16px;">
      <div class="label">Ең күдікті үзінді</div>
      <pre id="segmentText">-</pre>
    </div>
  </div>
</div></div>

<script>
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  });
});

const dropZone=document.getElementById("dropZone"),audioFile=document.getElementById("audioFile"),chosenFile=document.getElementById("chosenFile");
dropZone.addEventListener("click",()=>audioFile.click());
dropZone.addEventListener("dragover",e=>{e.preventDefault();dropZone.classList.add("drag-over");});
dropZone.addEventListener("dragleave",()=>dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop",e=>{e.preventDefault();dropZone.classList.remove("drag-over");if(e.dataTransfer.files[0])setFile(e.dataTransfer.files[0]);});
audioFile.addEventListener("change",()=>{if(audioFile.files[0])setFile(audioFile.files[0]);});
let selectedFile=null;
function setFile(f){selectedFile=f;chosenFile.textContent=f.name+" ("+(f.size/1024/1024).toFixed(2)+" МБ)";document.getElementById("analyzeAudioBtn").style.display="none";document.getElementById("transcriptPreview").style.display="none";}
function setProgress(pct,label){document.getElementById("progressWrap").style.display="block";document.getElementById("progressFill").style.width=pct+"%";document.getElementById("progressLabel").textContent=label;}
function hideProgress(){document.getElementById("progressWrap").style.display="none";}

document.getElementById("transcribeBtn").addEventListener("click",()=>{
  if(!selectedFile){alert("Сначала выбери аудиофайл.");return;}
  const btn=document.getElementById("transcribeBtn");
  btn.disabled=true;btn.textContent="Транскрибируем...";
  setProgress(10,"Загрузка файла...");
  const fd=new FormData();fd.append("audio",selectedFile);
  const xhr=new XMLHttpRequest();xhr.open("POST","/transcribe");
  xhr.upload.onprogress=e=>{if(e.lengthComputable){const pct=Math.round((e.loaded/e.total)*50);setProgress(pct,"Загрузка: "+(pct*2)+"%");}};
  xhr.onload=()=>{
    setProgress(90,"Whisper обрабатывает...");
    let data;try{data=JSON.parse(xhr.responseText);}catch{alert("Ошибка парсинга");btn.disabled=false;btn.textContent="🎙 Транскрибировать";hideProgress();return;}
    if(xhr.status!==200){alert(data.error||"Ошибка транскрипции");btn.disabled=false;btn.textContent="🎙 Транскрибировать";hideProgress();return;}
    document.getElementById("whisperText").value=data.transcript;
    document.getElementById("transcriptPreview").style.display="block";
    document.getElementById("analyzeAudioBtn").style.display="inline-block";
    setProgress(100,"Готово!");setTimeout(hideProgress,1200);
    btn.disabled=false;btn.textContent="🎙 Транскрибировать";
  };
  xhr.onerror=()=>{alert("Сетевая ошибка.");btn.disabled=false;btn.textContent="🎙 Транскрибировать";hideProgress();};
  xhr.send(fd);
});

async function runAnalysis(transcript){
  const r=await fetch("/analyze-call",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({transcript})});
  const data=await r.json();
  if(!r.ok)throw new Error(data.error||"Ошибка сервера");
  return data;
}

function renderResult(data){
  const pc=data.predicted_class||"unknown";
  const pillCls=pc==="fraud"?"fraud":pc==="suspicious"?"suspicious":"normal";
  document.getElementById("headline").innerHTML='<span class="pill '+pillCls+'">'+pc.toUpperCase()+'</span>';
  document.getElementById("fraudProbability").textContent=((data.fraud_probability||0)*100).toFixed(2)+"%";
  document.getElementById("riskLevel").textContent=data.risk_level||"-";
  document.getElementById("scenarioType").textContent=(data.features&&data.features.scenario_type)||"-";
  document.getElementById("modelUsed").textContent=data.model_used||"-";
  document.getElementById("recommendation").textContent=data.recommendation||"-";
  const markersEl=document.getElementById("markers");markersEl.innerHTML="";
  (data.markers||[]).forEach(t=>{const li=document.createElement("li");li.textContent=t;markersEl.appendChild(li);});
  const reasonsEl=document.getElementById("reasons");reasonsEl.innerHTML="";
  (data.decision_reasons||[]).forEach(t=>{const li=document.createElement("li");li.textContent=t;reasonsEl.appendChild(li);});
  document.getElementById("segmentText").textContent=(data.suspicious_segments&&data.suspicious_segments[0]&&data.suspicious_segments[0].text)||"Подозрительный сегмент не найден";
  document.getElementById("result").style.display="block";
}

document.getElementById("analyzeTextBtn").addEventListener("click",async()=>{
  const t=document.getElementById("transcript").value.trim();
  if(!t){alert("Вставь транскрипт.");return;}
  const btn=document.getElementById("analyzeTextBtn");
  btn.disabled=true;btn.textContent="Анализ...";
  try{renderResult(await runAnalysis(t));}catch(e){alert(e.message);}
  finally{btn.disabled=false;btn.textContent="Анализировать";}
});

document.getElementById("analyzeAudioBtn").addEventListener("click",async()=>{
  const t=document.getElementById("whisperText").value.trim();
  if(!t){alert("Транскрипт пуст.");return;}
  const btn=document.getElementById("analyzeAudioBtn");
  btn.disabled=true;btn.textContent="Анализ...";
  try{renderResult(await runAnalysis(t));}catch(e){alert(e.message);}
  finally{btn.disabled=false;btn.textContent="Анализировать";}
});
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/health")
def health():
    loaded = {k: type(v).__name__ for k, v in MODELS.items() if not isinstance(v, str)}
    return jsonify({"status": "ok", "model": MODEL_NAME, "results_dir": RESULTS_DIR, "loaded": loaded})

@app.route("/")
def index():
    return render_template_string(INDEX_HTML, model_name=MODEL_NAME, results_dir=RESULTS_DIR)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "audio file is required"}), 400
    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "empty filename"}), 400
    suffix = os.path.splitext(audio_file.filename)[1] or ".audio"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)
    try:
        model  = get_whisper()
        result = model.transcribe(tmp_path, language="ru", fp16=False)
        return jsonify({"transcript": result.get("text", "").strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try: os.unlink(tmp_path)
        except: pass

@app.route("/analyze-call", methods=["GET", "POST"])
def analyze_call_route():
    if request.method == "GET":
        return render_template_string(INDEX_HTML, model_name=MODEL_NAME, results_dir=RESULTS_DIR)
    payload    = request.get_json(force=True)
    transcript = payload.get("transcript", "")
    if not transcript.strip():
        return jsonify({"error": "transcript is required"}), 400
    try:
        result = analyze_call(transcript)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print(f"\n🌐 Сервер іске қосылды: http://localhost:8000\n")
    app.run(host="0.0.0.0", port=8000, debug=False)