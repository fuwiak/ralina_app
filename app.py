import streamlit as st
import requests
from io import BytesIO
import numpy as np
import cv2
from collections import Counter
from tensorflow.keras.models import load_model
from PIL import Image

# 1) Загружаем модель
model = load_model("landslide_model.h5")

# ВАЖНО: порядок классов должен совпадать с порядком при обучении
# Смотрите, что у вас в train_generator.class_indices, например:
# print(train_generator.class_indices)
# Если там {'high': 0, 'low': 1, 'medium': 2, 'none': 3}, то class_names = ["high","low","medium","none"]
class_names = ["none", "low", "medium", "high"]  
IMG_SIZE = (150, 150)

# 2) Функция обработки видео
def predict_video(video_path, model, sample_every=1):
    """
    Открываем видео (video_path) через OpenCV, идём по кадрам.
    - sample_every: берём каждый N-й кадр, чтобы уменьшить нагрузку (по умолчанию 1 = все кадры).
    - Для каждого кадра делаем предикт, сохраняем индекс предсказанного класса.
    - Возвращаем (название_класса, относительная_доля), соответствующие "чаще всего встречающийся" класс.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0.0

    frame_counter = 0
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        # Пропускаем кадры, если хотим брать каждый N-й
        if frame_counter % sample_every != 0:
            continue

        # Подготовим кадр для модели
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # BGR -> RGB
        pil_img = Image.fromarray(frame_rgb).resize(IMG_SIZE)  # PIL с ресайзом
        img_array = np.array(pil_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)          # (1, 150, 150, 3)

        # Предсказание
        preds = model.predict(img_array)  # (1, 4)
        class_idx = np.argmax(preds, axis=1)[0]
        predictions.append(class_idx)

    cap.release()
    if len(predictions) == 0:
        return None, 0.0

    # Считаем, какой класс встречается чаще
    counts = Counter(predictions)
    top_class_idx = max(counts, key=counts.get)
    top_class_count = counts[top_class_idx]

    # Относительная доля самого популярного класса среди всех кадров
    confidence = top_class_count / len(predictions)

    return class_names[top_class_idx], confidence


# 3) Интерфейс Streamlit
st.title("Landslide Risk Classifier (Video)")

tab1, tab2 = st.tabs(["Загрузить видеофайл", "Ссылка на видео (URL)"])

# --- СЦЕНАРИЙ 1: ФАЙЛ ---
with tab1:
    st.subheader("1) Загрузите видеофайл (mp4, mov, ...)")

    uploaded_file = st.file_uploader("Выберите видео", type=['mp4','avi','mov','mkv'])
    if uploaded_file is not None:
        # Сохраним байты во временный файл, чтобы OpenCV мог прочитать
        temp_video_path = "temp_uploaded_video.mp4"  # или .avi и т.п.
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())

        # Показываем само видео в плеере Streamlit
        st.video(temp_video_path)

        if st.button("Запустить классификацию (файл)"):
            with st.spinner("Анализируем видео. Подождите..."):
                pred_class, conf = predict_video(temp_video_path, model, sample_every=1)
                if pred_class is None:
                    st.error("Не удалось прочитать кадры из видео.")
                else:
                    st.success(f"Преобладающий класс: {pred_class}, доля = {conf:.2f}")

# --- СЦЕНАРИЙ 2: URL ---
with tab2:
    st.subheader("2) Укажите ссылку (URL) на видео")
    url_input = st.text_input("Введите ссылку (http/https):")
    if st.button("Загрузить и классифицировать (URL)"):
        url = url_input.strip()
        if not url:
            st.warning("Пожалуйста, введите непустой URL.")
        else:
            # Скачаем байты видео
            try:
                response = requests.get(url, stream=True)
                if response.status_code != 200:
                    st.error("Не удалось скачать видео. Код:", response.status_code)
                else:
                    temp_video_path = "temp_url_video.mp4"
                    with open(temp_video_path, "wb") as f:
                        f.write(response.content)

                    st.video(temp_video_path)

                    # Запускаем классификацию
                    with st.spinner("Анализируем видео. Подождите..."):
                        pred_class, conf = predict_video(temp_video_path, model, sample_every=1)
                        if pred_class is None:
                            st.error("Не удалось прочитать кадры из видео.")
                        else:
                            st.success(f"Преобладающий класс: {pred_class}, доля = {conf:.2f}")
            except Exception as e:
                st.error(f"Ошибка при скачивании: {e}")
