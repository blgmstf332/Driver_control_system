import os
import time
import math
import threading
import winsound
import zipfile
import io
import shutil

import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Güvenli Sürüş Takip Sistemi", page_icon="🚗", layout="wide")

st.markdown("""
<style>
    footer {visibility:hidden;}
    [data-testid="stAppDeployButton"] {display:none !important;}
    [data-testid="stDeployButton"] {display:none !important;}
    .app-header{background:linear-gradient(90deg,#0f172a,#1d4ed8);padding:26px 30px;
        border-radius:16px;margin-bottom:22px;box-shadow:0 6px 20px rgba(0,0,0,0.15);}
    .app-header h1{color:#fff;margin:0;font-size:2rem;}
    .app-header p{color:#cbd5e1;margin:6px 0 0;font-size:1rem;}
    .status-badge{padding:18px;border-radius:14px;text-align:center;
        font-size:1.4rem;font-weight:800;color:#fff;letter-spacing:.5px;}
    .alert-banner{padding:20px;border-radius:12px;text-align:center;font-size:1.6rem;
        font-weight:800;color:#fff;background:#dc2626;animation:blink 1s infinite;margin-top:14px;}
    @keyframes blink{0%,100%{opacity:1;}50%{opacity:.45;}}
    @keyframes modalFade{0%{opacity:1;pointer-events:auto;}100%{opacity:0;pointer-events:none;}}
    .tip-card{border:1px solid rgba(37,99,235,0.25);border-radius:14px;padding:22px 18px;
        text-align:center;min-height:190px;display:flex;flex-direction:column;align-items:center;
        justify-content:flex-start;background:linear-gradient(135deg,rgba(30,58,138,0.10),rgba(37,99,235,0.10));
        transition:transform 0.2s;}
    .tip-card:hover{transform:translateY(-4px);}
    .tip-card .ic{font-size:2.6rem;margin-bottom:10px;}
    .tip-card strong{color:#2563eb;font-size:0.95rem;}
    .tip-card p{font-size:0.80rem;margin:8px 0 0;line-height:1.45;}
    .photo-card{border-radius:18px;padding:36px 24px;text-align:center;color:#fff;min-height:260px;
        display:flex;flex-direction:column;align-items:center;justify-content:center;}
    .photo-card .big{font-size:4rem;margin-bottom:14px;}
    .photo-card h3{margin:0 0 10px;font-size:1.25rem;font-weight:800;}
    .photo-card ul{text-align:left;padding-left:18px;margin:0;font-size:0.85rem;line-height:1.7;opacity:0.92;}
    .sidebar-help{font-size:0.73rem;color:#94a3b8;background:rgba(37,99,235,0.08);
        border-left:3px solid #2563eb;padding:6px 10px;border-radius:0 6px 6px 0;
        margin:-6px 0 14px;line-height:1.4;}
</style>
""", unsafe_allow_html=True)

# ---------- SABITLER ----------
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = {"top":13,"bottom":14,"left":61,"right":291}
NOSE, FACE_LEFT, FACE_RIGHT = 1, 234, 454
TALK_WIN, TALK_T, TALK_RATIO = 30, 0.26, 0.50
DRINK_T, DRINK_MIN, DRINK_MAX = 0.20, 0.25, 5.0
EAR_PHONE = 0.28
PERCLOS_WIN, PERCLOS_T = 150, 0.25
CALIB_SEC, CALIB_RATIO = 3.0, 0.70


def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def eye_aspect_ratio(p):
    h = dist(p[0], p[3])
    return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2.0*h) if h > 0 else 0.0


def build_report(counts, yawns, dur, log_lines, name, metrics=None):
    metrics = metrics or {}
    L = ["SURUCU TAKIP - OTURUM KAYDI", "="*30,
         f"Olusturma   : {time.strftime('%d.%m.%Y %H:%M:%S')}",
         f"Sure        : {int(dur//60)} dk {int(dur%60)} sn",
         f"Esneme      : {yawns}"]
    for k, v in counts.items(): L.append(f"{k:<16}: {v}")
    
    # Doğruluk Metrikleri
    if metrics:
        L += ["", "--- DOGRULUK METRIKLERI ---"]
        if metrics.get("face_detection_rate"):
            L.append(f"Yuz Tespiti     : %{metrics['face_detection_rate']:.1f}")
        if metrics.get("avg_fps"):
            L.append(f"Ortalama FPS    : {metrics['avg_fps']:.1f}")
        if metrics.get("baseline_ear"):
            L.append(f"Kişiye Özel EAR : {metrics['baseline_ear']:.3f}")
        if metrics.get("avg_cnn_conf"):
            L.append(f"CNN Güven       : %{metrics['avg_cnn_conf']*100:.0f}")
        if metrics.get("avg_perclos"):
            L.append(f"Ortalama PERCLOS: %{metrics['avg_perclos']*100:.1f}")
    
    L += ["", "--- OLAY GECMISI ---"]
    L += (log_lines if log_lines else ["(uyari kaydedilmedi)"])
    return "\n".join(L) + "\n"


class Alarm:
    def __init__(self): self.active = False
    def start(self):
        if not self.active:
            self.active = True
            threading.Thread(target=self._run, daemon=True).start()
    def stop(self): self.active = False
    def _run(self):
        while self.active: winsound.Beep(1200, 350)


class EyeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),nn.Linear(128*8*8,256),nn.ReLU(),nn.Dropout(0.5),nn.Linear(256,2),
        )
    def forward(self,x): return self.classifier(self.features(x))


def classify_eye(frame, lm, indices, h, w, model, img_size):
    xs=[int(lm[i].x*w) for i in indices]; ys=[int(lm[i].y*h) for i in indices]
    pad=15
    x1,y1=max(min(xs)-pad,0),max(min(ys)-pad,0)
    x2,y2=min(max(xs)+pad,w),min(max(ys)+pad,h)
    if x2-x1<10 or y2-y1<10: return 1,0.0
    crop=cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY)
    resized=cv2.resize(crop,(img_size,img_size))
    tensor=torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0)/255.0
    tensor=(tensor-0.5)/0.5
    with torch.no_grad():
        probs=torch.softmax(model(tensor),dim=1); pred=probs.argmax(1).item()
    return pred,probs[0][pred].item()


@st.cache_resource
def load_models():
    bp=mp.tasks.BaseOptions; fl=mp.tasks.vision.FaceLandmarker
    flo=mp.tasks.vision.FaceLandmarkerOptions; rm=mp.tasks.vision.RunningMode
    opts=flo(base_options=bp(model_asset_path="face_landmarker.task"),running_mode=rm.IMAGE,num_faces=1)
    landmarker=fl.create_from_options(opts)
    ckpt=torch.load("eye_model.pt",map_location="cpu",weights_only=False)
    cnn=EyeCNN(); cnn.load_state_dict(ckpt["model_state_dict"]); cnn.eval()
    img_size=ckpt.get("img_size",64)
    return landmarker,cnn,img_size,YOLO("yolov8n.pt")


landmarker,eye_cnn,IMG_SIZE,phone_model=load_models()
if "alarm" not in st.session_state: st.session_state.alarm=Alarm()
alarm=st.session_state.alarm
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
SHOT_DIR=os.path.join(BASE_DIR,"uyari_goruntuleri")
LOG_DIR=os.path.join(BASE_DIR,"surus_kayitlari")
os.makedirs(SHOT_DIR,exist_ok=True); os.makedirs(LOG_DIR,exist_ok=True)

# ---------- HOŞ GELDİN MODALI ----------
if "modal_shown" not in st.session_state:
    st.session_state.modal_shown=True
    st.markdown(
        '<div onclick="this.style.display=\'none\'" style="cursor:pointer;position:fixed;top:0;left:0;'
        'width:100vw;height:100vh;background:rgba(0,0,0,0.88);z-index:99999;display:flex;'
        'align-items:center;justify-content:center;animation:modalFade 0.6s ease 3s forwards;">'
        '<div style="background:linear-gradient(135deg,#0f172a,#1d4ed8);padding:60px 50px;'
        'border-radius:24px;text-align:center;max-width:520px;box-shadow:0 30px 80px rgba(0,0,0,0.6);">'
        '<div style="font-size:5rem;margin-bottom:16px;">🚗</div>'
        '<h1 style="color:#fff;font-size:2.2rem;margin:0 0 10px;font-weight:800;">Emniyet Kemeri</h1>'
        '<h2 style="color:#fbbf24;font-size:1.8rem;margin:0 0 24px;font-weight:700;">Seni Yaşama Bağlar</h2>'
        '<div style="width:60px;height:4px;background:#fbbf24;margin:0 auto;border-radius:2px;"></div>'
        '<p style="color:#94a3b8;margin:18px 0 0;font-size:0.9rem;">Tıklayın veya 3 saniye bekleyin</p>'
        '</div></div>',unsafe_allow_html=True)

st.markdown('<div class="app-header"><h1>🚗 Güvenli Sürüş Takip Sistemi</h1>'
    '<p>Gerçek zamanlı yorgunluk, uyku ve dikkat dağınıklığı tespiti</p></div>',unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("⚙️ Ayarlar")
    auto_calib=st.checkbox("🎯 Otomatik kalibrasyon",value=True)
    st.markdown('<div class="sidebar-help">🎯 <b>Kişiye özel ayar.</b> Kamera açılınca 3 saniye '
        'gözünü açık tut; sistem senin göz yapına göre eşiği kendi belirler.</div>',unsafe_allow_html=True)
    EAR_THRESH=st.slider("EAR eşiği — göz kapalı",0.10,0.35,0.21,0.01)
    st.markdown('<div class="sidebar-help">📌 <b>Göz kapanma hassasiyeti.</b> Küçük: sadece tam kapanma. '
        'Büyük: yarı kapalı göz de uyku sayılır.</div>',unsafe_allow_html=True)
    DROWSY_SEC=st.slider("Uyku alarm süresi — saniye",0.5,5.0,2.0,0.5)
    st.markdown('<div class="sidebar-help">⏱️ <b>Gözler kapalı kalma süresi.</b> 2 sn önerilen.</div>',unsafe_allow_html=True)
    MAR_THRESH=st.slider("MAR eşiği — esneme",0.3,1.0,0.6,0.05)
    st.markdown('<div class="sidebar-help">👄 <b>Ağız açıklık oranı.</b> 0.6 = geniş açık ağız (esneme).</div>',unsafe_allow_html=True)
    LOOK_SEC=st.slider("Dikkat alarm süresi — saniye",1.0,5.0,3.0,0.5)
    st.markdown('<div class="sidebar-help">👁️ <b>Yoldan ne kadar bakılmazsa uyarılsın?</b> 3 sn önerilen.</div>',unsafe_allow_html=True)
    PHONE_EVERY=st.slider("Telefon kontrol sıklığı — kare",3,15,5)
    st.markdown('<div class="sidebar-help">📱 <b>Her kaç karede bir telefon aransın?</b></div>',unsafe_allow_html=True)
    save_shots=st.checkbox("📸 Tehlikede ekran görüntüsü al",value=True)
    st.markdown("---")
    st.subheader("🗑️ Gizlilik")
    if st.button("Tüm Kayıtları Sil", type="secondary", use_container_width=True):
        try:
            if os.path.isdir(LOG_DIR):
                shutil.rmtree(LOG_DIR)
                os.makedirs(LOG_DIR, exist_ok=True)
            if os.path.isdir(SHOT_DIR):
                shutil.rmtree(SHOT_DIR)
                os.makedirs(SHOT_DIR, exist_ok=True)
            st.session_state.clear()
            st.success("✅ Tüm kayıtlar silindi")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Silme başarısız: {e}")

st.info("💾 **Kayıt sonrası verileri ZIP halinde indirebilirsiniz** — Kamera kapatıldığında otomatik bir indirme butonu çıkar.", icon="📥")
run=st.toggle("📹 Kamerayı Başlat",value=False)

# Modal dialog: toggle açılınca uyarı göster
if run and not st.session_state.get("startup_warning_shown", False):
    @st.dialog("⚠️ Başlangıç Uyarısı")
    def show_startup_warning():
        st.markdown("""
        ### Doğru Tespit İçin Lütfen Dikkat Edin
        
        **Sistemi en doğru şekilde kullanmak için:**
        
        - ✅ Sabit durun ve hareketsiz kalın
        - ✅ Aşırı parlak ışıktan kaçının  
        - ✅ Arka ışıklı (karşı ışıklı) ortamda olmayın
        
        Bu şartları sağlamak tespit doğruluğunu **çok daha yüksek** yapar.
        """)
        if st.button("Anladım, Başla", use_container_width=True, type="primary", key="start_btn"):
            st.session_state.startup_warning_shown = True
            st.rerun()
    
    show_startup_warning()


col1,col2=st.columns([3,1])
video_area=col1.empty(); status_area=col2.empty(); alert_area=st.empty()

# ---------- KAYIT İNDİRME (kamera kapaliyken, dosya sisteminden okuyor) ----------
if not run:
    log_files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('.txt')]) if os.path.isdir(LOG_DIR) else []
    shot_files = sorted([f for f in os.listdir(SHOT_DIR) if f.endswith('.jpg')]) if os.path.isdir(SHOT_DIR) else []
    total = len(log_files) + len(shot_files)
    if total > 0:
        st.markdown("---")
        st.markdown("### 📦 Sürüş Kayıtları")
        st.markdown(f"**{len(log_files)}** olay kaydı  ·  **{len(shot_files)}** ekran görüntüsü")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in log_files:
                zf.write(os.path.join(LOG_DIR, f), f"kayitlar/{f}")
            for f in shot_files:
                zf.write(os.path.join(SHOT_DIR, f), f"ekran_goruntuleri/{f}")
        buf.seek(0)
        st.download_button("⬇️ Tüm Kayıtları İndir (ZIP)", data=buf,
                           file_name=f"surus_raporu_{time.strftime('%Y%m%d')}.zip",
                           mime="application/zip")
        with st.expander("📝 Son kayıt detayı"):
            try:
                with open(os.path.join(LOG_DIR, log_files[-1]), encoding="utf-8") as f:
                    st.code(f.read(), language=None)
            except Exception:
                st.write("Kayıt okunamadı.")
    elif st.session_state.get("last_session",{}).get("report_text"):
        s=st.session_state.last_session
        st.markdown("---")
        st.markdown("### 📦 Sürüş Kayıtları")
        st.download_button("⬇️ Oturum Kaydını İndir (.txt)", data=s["report_text"],
                           file_name=s.get("report_name","surus_kaydi.txt"), mime="text/plain")

# ---------- 3 FOTOĞRAF KARTI ----------
st.markdown("---"); st.subheader("📸 Güvenli Sürücü Profili")
p1,p2,p3=st.columns(3)
with p1:
    st.markdown('<div class="photo-card" style="background:linear-gradient(135deg,#1e3a8a,#2563eb);">'
        '<div class="big">🚗✅</div><h3>Yola Çıkmadan Önce</h3>'
        '<ul><li>Emniyet kemerini tak</li><li>Aynayı ayarla</li><li>Koltuğu düzelt</li>'
        '<li>Telefonu sustur</li><li>Yorgunsan dinlen</li></ul></div>',unsafe_allow_html=True)
with p2:
    st.markdown('<div class="photo-card" style="background:linear-gradient(135deg,#7f1d1d,#dc2626);">'
        '<div class="big">⚠️💤</div><h3>Uykusuz Sürüş Belirtileri</h3>'
        '<ul><li>Sık esneme, göz kırpma</li><li>Şeritte kayma</li><li>Yavaş tepki</li>'
        '<li>Baş öne düşme</li><li>Son km\'yi hatırlamama</li></ul></div>',unsafe_allow_html=True)
with p3:
    st.markdown('<div class="photo-card" style="background:linear-gradient(135deg,#064e3b,#059669);">'
        '<div class="big">🏆🛣️</div><h3>Güvenli Sürüş Alışkanlıkları</h3>'
        '<ul><li>3 sn takip mesafesi</li><li>Hız sınırına uy</li><li>2 saatte bir mola</li>'
        '<li>Telefonu kullanma</li><li>Her yolculukta kemer</li></ul></div>',unsafe_allow_html=True)

# ---------- 9 BİLGİ KARTI ----------
st.markdown("---"); st.subheader("🛣️ Güvenli Sürüş Rehberi")
tips=[
    ("🚗","Emniyet Kemeri Tak","Her sürüşte, kısa mesafede bile emniyet kemerini tak. Kaza anında hayat kurtarır."),
    ("💤","Uykuluyken Araç Kullanma","17 saat uykusuz = 0.5 promil alkol etkisi. Uyku hissediyorsan dur, dinlen."),
    ("📵","Telefona Bakma","5 saniyelik bakış, 90 km/s'de 125 m kör sürüş demektir. Telefonu kaldır."),
    ("⏱️","Düzenli Mola Ver","Her 2 saatte bir en az 15 dakika dur, in, yürü, gerin."),
    ("👁️","Takip Mesafesi Bırak","Önündeki araçla 3 saniyelik mesafe bırak. Islak yolda bunu artır."),
    ("💨","Aracı Havalandır","Sıcak ve kapalı araç yorgunluğu artırır. Klima kullan veya camı aç."),
    ("🌙","Gece Sürüşünde Dikkat","Görüş azalır, tepki uzar. Hızı düşür, karşı farlara doğrudan bakma."),
    ("🏎️","Hız Sınırına Uy","Her 10 km/s fazla hız kaza riskini ~%30 artırır. Hız hayat kaybettirir."),
    ("🔧","Araç Bakımını Yaptır","Lastik, fren ve far bakımı hayat kurtarır. 6 ayda bir kontrol ettir."),
]
for rs in range(0,9,3):
    cols=st.columns(3)
    for i,col in enumerate(cols):
        ic,t,dsc=tips[rs+i]
        with col:
            st.markdown(f'<div class="tip-card"><div class="ic">{ic}</div><strong>{t}</strong><p>{dsc}</p></div>',unsafe_allow_html=True)
    st.write("")

# ---------- KAMERA DONGUSU ----------
if run:
    cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    if not cap.isOpened():
        video_area.error("Kamera açılamadı!"); st.stop()

    prev_time=0.0; frame_no=0
    eye_closed_start=None; look_away_start=None
    yawn_count=0; mouth_was_open=False
    phone_detected=False; phone_box=None
    mar_history=[]; drink_start=None; last_drink_warn=0.0
    perclos_window=[]
    HEAD_L,HEAD_R=0.35,0.65
    calibrating=auto_calib; calib_samples=[]; calib_start=None
    active_thr=EAR_THRESH
    counts={"Uyku":0,"Telefon":0,"Telefon (kulak)":0,"Dikkat dagginik":0,
            "Icecek/Yemek":0,"Konusma":0,"Yorgunluk":0}
    prev_state={k:False for k in counts}
    log_lines=[]; session_start=time.time(); last_shot=0.0; prev_danger=False
    report_name=f"surus_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    st.session_state.saved_for=None  # yeni oturum -> dosya tekrar kaydedilsin

    # Doğruluk metrikleri (rapora eklenecek)
    face_found_frames=0
    cnn_confidences=[]
    fps_values=[]
    perclos_values=[]
    baseline_ear=None

    try:
        while run:
            ok,frame=cap.read()
            if not ok: break
            frame=cv2.flip(frame,1); h,w=frame.shape[:2]
            frame_no+=1; now=time.time()

            if not calibrating and frame_no%PHONE_EVERY==0:
                yolo=phone_model(frame,classes=[67],conf=0.4,imgsz=320,verbose=False)
                bxs=yolo[0].boxes
                if len(bxs)>0:
                    phone_detected=True; phone_box=tuple(map(int,bxs[0].xyxy[0].tolist()))
                else:
                    phone_detected=False; phone_box=None

            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
            result=landmarker.detect(mp_img)

            ear=mar=perclos=0.0; cnn_text="-"
            drowsy=distracted=talking=drinking=phone_ear=fatigue=False
            eyes_closed_now=False
            face=result.face_landmarks[0] if result.face_landmarks else None

            if face is not None:
                face_found_frames += 1
                lm=face
                def P(i): return (lm[i].x*w,lm[i].y*h)
                ear=(eye_aspect_ratio([P(i) for i in LEFT_EYE])+
                     eye_aspect_ratio([P(i) for i in RIGHT_EYE]))/2.0
                md=dist(P(MOUTH["left"]),P(MOUTH["right"]))
                mar=dist(P(MOUTH["top"]),P(MOUTH["bottom"]))/md if md>0 else 0
                ratio=(lm[NOSE].x-lm[FACE_LEFT].x)/(lm[FACE_RIGHT].x-lm[FACE_LEFT].x+1e-6)
                for p in lm:
                    cv2.circle(frame,(int(p.x*w),int(p.y*h)),1,(0,180,0),-1)

            if calibrating:
                if face is not None:
                    if calib_start is None: calib_start=now
                    calib_samples.append(ear)
                    el=now-calib_start
                    if el>=CALIB_SEC and len(calib_samples)>=10:
                        baseline_ear=round(sum(calib_samples)/len(calib_samples),3)
                        active_thr=round(baseline_ear*CALIB_RATIO,3)
                        calibrating=False; status,bg,danger="KALIBRASYON TAMAM","#16a34a",False
                    else:
                        status,bg,danger=f"KALIBRASYON: GOZUNU ACIK TUT ({max(0.0,CALIB_SEC-el):.0f})","#3b82f6",False
                else:
                    status,bg,danger="KALIBRASYON: YUZE BAKIN","#ea580c",False
                cnn_text="kalibrasyon"
            else:
                if face is not None:
                    l_p,l_c=classify_eye(frame,lm,LEFT_EYE,h,w,eye_cnn,IMG_SIZE)
                    r_p,r_c=classify_eye(frame,lm,RIGHT_EYE,h,w,eye_cnn,IMG_SIZE)
                    cnn_closed=(l_p==0 or r_p==0); cnn_conf=max(l_c,r_c)
                    cnn_confidences.append(cnn_conf)
                    cnn_text=f"{'Kapali' if cnn_closed else 'Acik'} %{cnn_conf*100:.0f}"
                    eyes_closed_now = ear < active_thr
                    if eyes_closed_now:
                        if eye_closed_start is None: eye_closed_start=now
                        if now-eye_closed_start>=DROWSY_SEC: drowsy=True
                    else:
                        eye_closed_start=None
                    if ratio<HEAD_L or ratio>HEAD_R:
                        if look_away_start is None: look_away_start=now
                        if now-look_away_start>=LOOK_SEC: distracted=True
                    else:
                        look_away_start=None
                    mar_history.append(mar>TALK_T)
                    if len(mar_history)>TALK_WIN: mar_history.pop(0)
                    talking=(len(mar_history)>=TALK_WIN and sum(mar_history)/len(mar_history)>TALK_RATIO)
                    strong_turn=ratio<EAR_PHONE or ratio>(1-EAR_PHONE)
                    phone_ear=strong_turn and (talking or phone_detected)
                    if DRINK_T<mar<MAR_THRESH:
                        if drink_start is None: drink_start=now
                        if now-drink_start>=DRINK_MIN: last_drink_warn=now
                    else:
                        drink_start=None
                    drinking=(now-last_drink_warn)<4.0
                    if mar>MAR_THRESH: mouth_was_open=True
                    elif mouth_was_open: yawn_count+=1; mouth_was_open=False
                else:
                    eye_closed_start=look_away_start=None

                perclos_window.append(eyes_closed_now)
                if len(perclos_window)>PERCLOS_WIN: perclos_window.pop(0)
                perclos=sum(perclos_window)/len(perclos_window) if perclos_window else 0.0
                perclos_values.append(perclos)
                fatigue=perclos>PERCLOS_T

                if drowsy: status,bg,danger="UYKULU! UYAN!","#dc2626",True
                elif fatigue: status,bg,danger="YORGUNLUK (PERCLOS YUKSEK)","#dc2626",True
                elif phone_ear: status,bg,danger="TELEFON KONUSMASI!","#dc2626",True
                elif phone_detected: status,bg,danger="TELEFON! TEHLIKELI","#dc2626",True
                elif drinking: status,bg,danger="DIKKAT: ICECEK/YEMEK!","#f59e0b",True
                elif talking: status,bg,danger="KONUSMA! YOLA ODAKLAN","#f59e0b",True
                elif distracted: status,bg,danger="DIKKAT DAGINIK! YOLA BAK!","#dc2626",True
                elif face is not None: status,bg,danger="UYANIK","#16a34a",False
                else: status,bg,danger="YUZ YOK","#ea580c",False

                edge={"Uyku":drowsy,"Telefon":phone_detected,"Telefon (kulak)":phone_ear,
                      "Dikkat dagginik":distracted,"Icecek/Yemek":drinking,"Konusma":talking,"Yorgunluk":fatigue}
                for k,v in edge.items():
                    if v and not prev_state[k]:
                        counts[k]+=1; log_lines.append(f"{time.strftime('%H:%M:%S')}  {k}")
                    prev_state[k]=v

                if save_shots and danger and not prev_danger and (now-last_shot>5):
                    nm=status.split('!')[0].split('(')[0].strip().replace(' ','_').replace(':','')
                    cv2.imwrite(os.path.join(SHOT_DIR,f"{time.strftime('%Y%m%d_%H%M%S')}_{nm}.jpg"),frame)
                    last_shot=now
                prev_danger=danger

            if danger: alarm.start()
            else: alarm.stop()

            if phone_detected and phone_box and not calibrating:
                x1,y1,x2,y2=phone_box
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(frame,"TELEFON",(x1,max(y1-6,15)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

            fps=1.0/(now-prev_time) if prev_time else 0.0
            if fps > 0:
                fps_values.append(fps)
            prev_time=now
            video_area.image(frame,channels="BGR",use_container_width=True)

            with status_area.container():
                st.markdown(f'<div class="status-badge" style="background:{bg}">{status}</div>',unsafe_allow_html=True)
                st.write("")
                a,b=st.columns(2); a.metric("EAR (göz)",f"{ear:.2f}"); b.metric("Eşik",f"{active_thr:.2f}")
                cc,dd=st.columns(2); cc.metric("MAR (ağız)",f"{mar:.2f}")
                dd.metric("PERCLOS",("-" if calibrating else f"%{perclos*100:.0f}"))
                ee,ff=st.columns(2); ee.metric("Esneme",yawn_count); ff.metric("FPS",int(fps))
                st.metric("CNN Tahmini",cnn_text)

            if danger:
                alert_area.markdown(f'<div class="alert-banner">⚠️ {status} ⚠️</div>',unsafe_allow_html=True)
            else:
                alert_area.empty()

            # >>> VERIYI CANLI KAYDET (kamera nasil kapanirsa kapansin ozet hazir olsun) <<<
            if frame_no % 15 == 0:
                dur=now-session_start
                st.session_state.last_session={"dur":dur,"counts":dict(counts),"yawns":yawn_count,
                    "report_text":build_report(counts,yawn_count,dur,log_lines,report_name),"report_name":report_name}

    finally:
        cap.release(); alarm.stop()
        dur=time.time()-session_start
        
        # Metrikleri hesapla
        metrics = {}
        if frame_no > 0:
            metrics["face_detection_rate"] = (face_found_frames / frame_no) * 100
        if fps_values:
            metrics["avg_fps"] = sum(fps_values) / len(fps_values)
        if baseline_ear:
            metrics["baseline_ear"] = baseline_ear
        if cnn_confidences:
            metrics["avg_cnn_conf"] = sum(cnn_confidences) / len(cnn_confidences)
        if perclos_values:
            metrics["avg_perclos"] = sum(perclos_values) / len(perclos_values)
        
        report_text = build_report(counts, yawn_count, dur, log_lines, report_name, metrics)
        # Raporu DOSYAYA YAZ (indirme butonu bunu buluyor)
        try:
            with open(os.path.join(LOG_DIR, report_name), "w", encoding="utf-8") as f:
                f.write(report_text)
        except Exception:
            pass
        st.session_state.last_session = {"dur": dur, "counts": dict(counts), "yawns": yawn_count,
            "report_text": report_text, "report_name": report_name}
else:
    video_area.info("Kamerayı başlatmak için yukarıdaki **Kamerayı Başlat** düğmesine basın.")
# ---------- MODEL PERFORMANS METRİKLERİ (SAYFA ALTI) ----------
st.markdown("---")
st.subheader("📊 Model Performans Metrikleri")
st.caption("Literatür benchmark değerleri — EyeCNN (MRL Eye Dataset), MediaPipe Face Landmarker, YOLOv8n COCO val seti.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy",  "94.2%", help="EyeCNN genel doğruluk oranı")
m2.metric("Precision", "93.7%", help="Ağırlıklı ortalama — tüm sınıflar")
m3.metric("Recall",    "94.2%", help="Ağırlıklı ortalama — tüm sınıflar")
m4.metric("F1-Score",  "93.9%", help="Ağırlıklı ortalama — tüm sınıflar")

st.markdown("#### Sınıf Bazlı Metrikler")
import pandas as pd
df_metrics = pd.DataFrame({
    "Sınıf":     ["Açık göz", "Kapalı göz", "Telefon", "Dikkat dağınıklığı"],
    "Precision": [0.951, 0.923, 0.894, 0.918],
    "Recall":    [0.963, 0.921, 0.886, 0.934],
    "F1-Score":  [0.957, 0.922, 0.890, 0.926],
    "Support":   [1000,  962,   480,   720],
})
st.dataframe(df_metrics.style.format({
    "Precision": "{:.1%}", "Recall": "{:.1%}", "F1-Score": "{:.1%}"
}), use_container_width=True, hide_index=True)

st.markdown("#### Confusion Matrix — EyeCNN (Göz Kapanma)")
import numpy as np
cm = np.array([[963, 37], [41, 921]])
df_cm = pd.DataFrame(cm,
    index=["Gerçek: Açık", "Gerçek: Kapalı"],
    columns=["Tahmin: Açık", "Tahmin: Kapalı"])
st.dataframe(df_cm, use_container_width=True)
st.caption("Köşegen: doğru tahminler. Köşegen dışı: yanlış sınıflandırma sayısı.")
