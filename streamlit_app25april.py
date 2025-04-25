import streamlit as st, tensorflow as tf, numpy as np, cv2, tempfile, os, sys, subprocess, importlib

# ─── безопасный импорт OpenCV без libGL ───
def safe_import_cv2():
    try:
        return importlib.import_module("cv2")
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "--quiet", "opencv-python-headless==4.7.0.72"])
        return importlib.import_module("cv2")
cv2 = safe_import_cv2()

# ────────── параметры ──────────
IMG_SIZE = 150
STEP     = 10       # каждый 10-й кадр
THR_OBV  = 0.60     # порог уверенности «есть обвал»
COLLAPSE_CLASSES = ["No_Landslide", "Rockfall", "Earth_Flow"]
DANGER_CLASSES   = ["Safe", "Roads_Damaged", "Houses_Damaged"]

# ─── кэш-загрузка моделей ───
@st.cache_resource(show_spinner=False)
def load_models():
    coll = tf.keras.models.load_model("collapse_model (1).h5", compile=False) \
           if os.path.exists("collapse_model (1).h5") else None
    if not os.path.exists("danger_model.h5"):
        st.error("Файл **danger_model.h5** не найден."); st.stop()
    dang = tf.keras.models.load_model("danger_model (1).h5", compile=False)
    return coll, dang
collapse_model, danger_model = load_models()

# ─── классификация одного ролика ───
def classify(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    prog  = st.progress(0)
    # голоса
    v_coll = np.zeros(3,int) if collapse_model else None
    v_dan  = np.zeros(3,int)
    frame_id = 0; every = max(total//100,1)

    while True:
        ok, frame = cap.read()
        if not ok: break
        if frame_id % STEP == 0:
            fr = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            (IMG_SIZE, IMG_SIZE))
            if collapse_model:
                p = collapse_model.predict(np.expand_dims(fr,0)/255., verbose=0)[0]
                v_coll[np.argmax(p)] += 1
            q = danger_model.predict(np.expand_dims(fr,0)/255., verbose=0)[0]
            v_dan[np.argmax(q)] += 1
        if frame_id % every == 0:
            prog.progress(min(frame_id/max(total,1), 1.0))
        frame_id += 1

    cap.release(); prog.empty()

    # итог collapse
    if collapse_model:
        conf_c = v_coll.max()/v_coll.sum() if v_coll.sum() else 0
        cls_c  = COLLAPSE_CLASSES[v_coll.argmax()]
        if conf_c < THR_OBV:                         # порог
            cls_c, conf_c = "No_Landslide", 1-conf_c
    else:
        cls_c, conf_c = None, None

    # итог danger
    conf_d = v_dan.max()/v_dan.sum() if v_dan.sum() else 0
    cls_d  = DANGER_CLASSES[v_dan.argmax()]

    return cls_c, conf_c, cls_d, conf_d

# ─── UI ───
st.set_page_config("Landslide Detection", "🌋", layout="centered")
st.title("🌋 Landslide Detection")

st.markdown(
"Загрузите видео (*mp4 / avi / mov / mkv*) и нажмите **Анализировать**. "
"Модель определит вид обвала и уровень опасности. "
"Если рядом нет `collapse_model.h5`, будет показана только опасность."
)

video_file = st.file_uploader("Видео-файл", type=["mp4","avi","mov","mkv"])

if video_file:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(video_file.read()); tmp.close()
    st.video(tmp.name)   # превью

    if st.button("🔍 Анализировать"):
        with st.spinner("Обработка…"):
            res = classify(tmp.name)
            if res is None:
                st.error("Не удалось прочитать видео.")
            else:
                cls_c, conf_c, cls_d, conf_d = res
                if cls_c is not None:
                    st.info(f"**Тип обвала:** {cls_c}  ({conf_c*100:.1f} %)")
                st.success(f"**Опасность:** {cls_d}  ({conf_d*100:.1f} %)")
        os.unlink(tmp.name)  # удаляем временный файл
else:
    st.caption("⬆️  Выберите видео для анализа")
