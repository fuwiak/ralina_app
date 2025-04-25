# streamlit_app.py  —  совместим с .h5, в которых есть слой “Cast/TFOpLambda”
# --------------------------------------------------------------------------

import streamlit as st, tensorflow as tf, numpy as np, tempfile, os, sys, subprocess, importlib

# ─── безопасный импорт OpenCV (headless) ───
def safe_import_cv2():
    try:
        return importlib.import_module("cv2")
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "--quiet", "opencv-python-headless==4.7.0.72"])
        return importlib.import_module("cv2")
cv2 = safe_import_cv2()

# ─── константы ───
IMG = 150
STEP = 10
THR  = 0.60        # порог уверенности «есть обвал»
COLLAPSE = ["No_Landslide", "Rockfall", "Earth_Flow"]
DANGER   = ["Safe", "Roads_Damaged", "Houses_Damaged"]

# ─── регистрация «неизвестных» слоёв из .h5 ───
CUSTOM_OBJECTS = {
    "Cast":        tf.keras.layers.Lambda,   # старый сериализатор пишет Cast
    "TFOpLambda":  tf.keras.layers.Lambda,   # Keras-3 пишет TFOpLambda
}

@st.cache_resource(show_spinner=False)
def load_models():
    coll = tf.keras.models.load_model("collapse_model (1).h5",
                                      compile=False,
                                      custom_objects=CUSTOM_OBJECTS) \
           if os.path.exists("collapse_model (1).h5") else None

    if not os.path.exists("danger_model (1).h5"):
        st.error("Файл **danger_model (1).h5** не найден."); st.stop()

    dang = tf.keras.models.load_model("danger_model (1).h5",
                                      compile=False,
                                      custom_objects=CUSTOM_OBJECTS)
    return coll, dang
collapse_model, danger_model = load_models()

# ─── видео → (тип обвала, его conf, опасность, её conf) ───
def analyse_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step_prog = max(total//100, 1)
    prog = st.progress(0)

    v_coll = np.zeros(3,int) if collapse_model is not None else None
    v_dang = np.zeros(3,int)
    fid = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if fid % STEP == 0:
            fr = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (IMG, IMG))
            inp = np.expand_dims(fr,0)/255.
            if collapse_model is not None:
                p = collapse_model.predict(inp, verbose=0)[0]
                v_coll[np.argmax(p)] += 1
            q = danger_model.predict(inp, verbose=0)[0]
            v_dang[np.argmax(q)] += 1
        if fid % step_prog == 0:
            prog.progress(min(fid/max(total,1), 1.0))
        fid += 1
    cap.release(); prog.empty()

    # итог collapse
    if collapse_model is not None:
        conf_c = v_coll.max()/v_coll.sum() if v_coll.sum() else 0
        cls_c  = COLLAPSE[v_coll.argmax()]
        if conf_c < THR:
            cls_c, conf_c = "No_Landslide", 1-conf_c
    else:
        cls_c, conf_c = None, None

    conf_d = v_dang.max()/v_dang.sum() if v_dang.sum() else 0
    cls_d  = DANGER[v_dang.argmax()]

    return cls_c, conf_c, cls_d, conf_d

# ─── UI ───
st.set_page_config("Landslide Detection", "🌋", layout="centered")
st.title("🌋 Landslide Detection")

st.write(
"Загрузите видео (*.mp4, *.avi, *.mov, *.mkv*) и нажмите **Анализировать**. "
"Если `collapse_model.h5` отсутствует, приложение покажет только опасность."
)

file = st.file_uploader("Видео-файл", type=["mp4","avi","mov","mkv"])
if file:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(file.read()); tmp.close()
    st.video(tmp.name)

    if st.button("🔍 Анализировать"):
        with st.spinner("Обработка…"):
            res = analyse_video(tmp.name)
        os.unlink(tmp.name)            # очищаем tmp
        if res is None:
            st.error("Не удалось прочитать видео.")
        else:
            c_cls,c_conf,d_cls,d_conf = res
            if c_cls is not None:
                st.info(f"**Тип обвала:** {c_cls}  ({c_conf*100:.1f} %)")
            st.success(f"**Опасность:** {d_cls}  ({d_conf*100:.1f} %)")
else:
    st.caption("⬆️  Выберите видео для анализа")
