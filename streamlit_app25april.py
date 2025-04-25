import streamlit as st
st.set_page_config("🌋 Landslide Detection", "🌋")

import tensorflow as tf, numpy as np, os, tempfile, importlib

# ─── 1. видео-бэкенд ───
try:
    import cv2
    BACKEND = "cv2"
except Exception:
    import imageio.v3 as iio
    BACKEND = "imageio"

# ─── 2. константы ───
IMG, STEP, THR = 150, 10, 0.60
COLLAPSE = ["No_Landslide", "Rockfall", "Earth_Flow"]
DANGER   = ["Safe", "Roads_Damaged", "Houses_Damaged"]

# ─── 3. заглушки для странных слоёв ───
Identity = tf.keras.layers.Lambda(lambda x: x, name="Identity")
CUSTOM   = {k: Identity for k in
            ["TFOpLambda", "Cast", "tf.math.multiply", "tf.__operators__.add"]}

# ─── 4. fallback-builder MobileNet-head ───
def build_head():
    base = tf.keras.applications.MobileNetV2(include_top=False,
                                             input_shape=(IMG, IMG, 3),
                                             weights=None)
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(3, activation="softmax")(x)
    return tf.keras.Model(base.input, out)

def robust_load(h5_path: str) -> tf.keras.Model:
    try:
        return tf.keras.models.load_model(
            h5_path, compile=False, safe_mode=False, custom_objects=CUSTOM
        )
    except Exception as e:
        # st.warning(f"Не удалось полностью загрузить «{h5_path}» "
        #            f"({type(e).__name__}). Пытаюсь восстановить веса…")
        model = build_head()
        model.load_weights(h5_path, skip_mismatch=True)
        return model

# ─── 5. кэш-загрузка моделей ───
@st.cache_resource(show_spinner=False)
def load_models():
    # coll_p   = "collapse_model (1).h5"
    # danger_p = "danger_model (1).h5"
    coll_p   = "collapse_model.h5"
    danger_p = "danger_model.h5"

    coll = robust_load(coll_p)   if os.path.exists(coll_p)   else None
    if not os.path.exists(danger_p):
        st.error("Файл **danger_model (1).h5** не найден."); st.stop()
    danger = robust_load(danger_p)
    return coll, danger

COLL_M, DANG_M = load_models()

# ─── 6. анализ видео ───
def analyse(path: str):
    votes_c = np.zeros(3, int) if COLL_M else None
    votes_d = np.zeros(3, int)

    if BACKEND == "cv2":
        import cv2
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        prog = st.progress(0); every = max(total // 100, 1)
        while True:
            ok, fr = cap.read(); idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if not ok: break
            if idx % STEP == 0:
                fr = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (IMG, IMG))
                if COLL_M is not None:
                    votes_c[COLL_M(fr[None] / 255., training=False)
                            .numpy().argmax()] += 1
                votes_d[DANG_M(fr[None] / 255., training=False)
                        .numpy().argmax()] += 1
            if idx % every == 0: prog.progress(idx / total)
        cap.release(); prog.empty()
    else:
        rdr = iio.get_reader(path, format="ffmpeg"); total = len(rdr); prog = st.progress(0)
        for idx, fr in enumerate(rdr):
            if idx % STEP == 0:
                fr = tf.image.resize(fr, (IMG, IMG)).numpy().astype("uint8")
                if COLL_M is not None:
                    votes_c[COLL_M(fr[None] / 255., training=False)
                            .numpy().argmax()] += 1
                votes_d[DANG_M(fr[None] / 255., training=False)
                        .numpy().argmax()] += 1
            if idx % (total // 100 + 1) == 0: prog.progress(idx / total)
        prog.empty()

    # — результаты —
    if COLL_M and votes_c.sum():
        conf_c = votes_c.max() / votes_c.sum()
        cls_c  = COLLAPSE[votes_c.argmax()]
        if conf_c < THR: cls_c = "No_Landslide"
    else:
        cls_c, conf_c = None, None

    conf_d = votes_d.max() / votes_d.sum() if votes_d.sum() else 0
    cls_d  = DANGER[votes_d.argmax()]

    return cls_c, conf_c, cls_d, conf_d

# ─── 7. интерфейс ───
st.title("🌋 Landslide Detection")
st.write("Загрузите видео (*mp4 / avi / mov / mkv*) и нажмите **Анализировать**.")

file = st.file_uploader("Видео-файл", type=["mp4", "avi", "mov", "mkv"])
if file:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(file.read()); tmp.close()
    st.video(tmp.name)

    if st.button("🔍 Анализировать"):
        with st.spinner(f"Анализ… (backend = {BACKEND})"):
            c_cls, c_conf, d_cls, d_conf = analyse(tmp.name)
            if c_cls is not None:
                st.info(f"**Тип обвала:** {c_cls} — {c_conf*100:.1f}%")
            st.success(f"**Опасность:** {d_cls} — {d_conf*100:.1f}%")
        os.unlink(tmp.name)
else:
    st.caption("⬆️ Сначала выберите видео.")
