# streamlit_app.py  –  фиксим Unknown layer 'Cast'

import streamlit as st, tensorflow as tf, numpy as np, os, tempfile, sys, subprocess, importlib

# — headless OpenCV —
def safe_cv2():
    try:
        import cv2
        return cv2
    except Exception as first_err:
        try:
            import subprocess, sys, importlib
            # >=4.9.0.80 есть колёса под 3.12
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--quiet",
                "opencv-python-headless>=4.9.0.80"
            ])
            return importlib.import_module("cv2")
        except Exception as second_err:
            st.error(
                "Не удалось установить **opencv-python-headless**.\n\n"
                f"Первая ошибка: {first_err}\n\n"
                f"Повторная попытка: {second_err}\n\n"
                "Попробуйте вручную:  \n"
                "`pip install opencv-python-headless>=4.9.0.80`"
            )
            st.stop()
cv2 = safe_cv2()

IMG, STEP, THR = 150, 10, 0.60
COLLAPSE_CLASSES = ["No_Landslide", "Rockfall", "Earth_Flow"]
DANGER_CLASSES   = ["Safe", "Roads_Damaged", "Houses_Damaged"]

@st.cache_resource(show_spinner=False)
def load_models():
    coll = None
    try:
        if os.path.exists("collapse_model (1).h5"):
            # ① Keras-3: отключаем safe_mode → игнорируем незнакомые слои Cast/TFOpLambda
            coll = tf.keras.models.load_model("collapse_model (1).h5",
                                              compile=False, safe_mode=False)
    except Exception as e:
        st.warning(f"collapse_model (1).h5 не удалось открыть: {e}")
    if not os.path.exists("danger_model (1).h5"):
        st.error("Файл danger_model (1).h5 не найден."); st.stop()
    # ② аналогично для danger
    danger = tf.keras.models.load_model("danger_model (1).h5",
                                        compile=False, safe_mode=False)
    return coll, danger
collapse_model, danger_model = load_models()

# —–– единая функция инференса —––
def classify(path):
    cap = cv2.VideoCapture(path); tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    v_col = np.zeros(3,int) if collapse_model else None
    v_dan = np.zeros(3,int); prog = st.progress(0)
    i=0; every=max(tot//100,1)
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i % STEP == 0:
            fr = cv2.resize(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB),(IMG,IMG))
            if collapse_model:
                p = collapse_model.predict(fr[None]/255.,verbose=0)[0]
                v_col[p.argmax()] += 1
            q = danger_model.predict(fr[None]/255.,verbose=0)[0]
            v_dan[q.argmax()]  += 1
        if i % every == 0: prog.progress(min(i/tot,1.0))
        i += 1
    cap.release(); prog.empty()

    # итоги
    if collapse_model:
        conf_c = v_col.max()/v_col.sum() if v_col.sum() else 0
        cls_c  = COLLAPSE_CLASSES[v_col.argmax()]
        if conf_c < THR: cls_c, conf_c = "No_Landslide", 1-conf_c
    else:
        cls_c, conf_c = None, None
    conf_d = v_dan.max()/v_dan.sum() if v_dan.sum() else 0
    cls_d  = DANGER_CLASSES[v_dan.argmax()]
    return cls_c, conf_c, cls_d, conf_d

# —–– UI —––
st.set_page_config("🌋 Landslide Detection", "🌋"); st.title("🌋 Landslide Detection")
st.write("Загрузите видео (.mp4 / .avi / .mov / .mkv) и нажмите **Анализировать**.")

file = st.file_uploader("Видео-файл", type=["mp4","avi","mov","mkv"])
if file:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(file.read()); tmp.close(); st.video(tmp.name)
    if st.button("🔍 Анализировать"):
        with st.spinner("Обработка…"):
            c_cls,c_conf,d_cls,d_conf = classify(tmp.name)
            if c_cls is not None:
                st.info(f"**Тип обвала:** {c_cls} ({c_conf*100:.1f} %)")
            st.success(f"**Опасность:** {d_cls} ({d_conf*100:.1f} %)")
        os.unlink(tmp.name)
else:
    st.caption("⬆️  Сначала выберите видео.")
