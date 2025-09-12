# Procesamiento y transcripción de audio
import torch
import whisperx
import pydub
from pydub import AudioSegment
import imageio_ffmpeg as iio_ffmpeg
import librosa

# NLP y análisis
import pandas as pd
import numpy as np
import re
import string
from collections import Counter
from pysentimiento import create_analyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from deep_translator import GoogleTranslator

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import base64
import pdfkit

# Web app
import gradio as gr
import uuid
import csv
import os
import tempfile
import time

# Seguridad
from dotenv import load_dotenv

#Funciones
import traceback

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
whisper_model = whisperx.load_model("small", device, compute_type=compute_type)
align_model, align_metadata = None, None  # se cargará dinámicamente según idioma
labels = ["Positive", "Negative", "Neutral"]

load_dotenv("secrets.env")
HF_TOKEN = os.getenv("HF_TOKEN")

# Cargar el modelo roberta-base-go_emotions
try:
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    print("✓ Modelo roberta-base-go_emotions cargado correctamente")
except Exception as e:
    print(f"Error cargando modelo roberta: {e}")

# Descargas de NLTK
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("Error descargando recursos NLTK, pero continuando...")
    
# Stop words
stop_words = set(stopwords.words('spanish'))
# Puntuación a eliminar (excepto signos de exclamación/interrogación)
punctuation_to_strip = set(string.punctuation) | {"«", "»", "…"}
punctuation_to_strip -= {"¡", "¿", "!", "?"}

# Variable global para almacenar los datos de la transcripción
current_audioDF = None
current_audio_path = None

# Variable global para el idioma
lenguaje = "es"  # Idioma del audio a analizar

def transcribe_with_whisperx_stream(audio_path, num_speakers=None, progress=gr.Progress()):
    """
    Transcripción con progress bar mejorado usando Gradio Progress
    """
    global align_model, align_metadata, current_audioDF, current_audio_path
    current_audio_path = audio_path
    progress(0, desc="🚀 Iniciando transcripción...")
    
    try:
        # --- 1. Cargar audio ---
        progress(0.05, desc="📁 Cargando archivo de audio...")
        audio = whisperx.load_audio(audio_path)
        duration = librosa.get_duration(path=audio_path)
        chunk_size = 30  # segundos
        num_chunks = int(duration // chunk_size) + 1
        partial_segments = []
        transcript_so_far = ""
        
        # --- 2. Transcribir en chunks ---
        progress(0.1, desc="🎤 Transcribiendo audio...")
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, duration)
            
            # Transcribir solo ese pedazo
            audio_chunk = audio[int(start * 16000):int(end * 16000)]
            result = whisper_model.transcribe(audio_chunk, batch_size=8)
            
            # Guardar segmentos
            for seg in result["segments"]:
                seg["start"] += start
                seg["end"] += start
                partial_segments.append(seg)
                transcript_so_far += f"{seg['text'].strip()} "
                
            # Actualizar progreso (0.1 a 0.6 para transcripción)
            chunk_progress = 0.1 + (0.5 * (i + 1) / num_chunks)
            progress(chunk_progress, desc=f"🎤 Transcribiendo segmento {i+1}/{num_chunks}...")
            
        # --- 3. Alineación ---
        progress(0.65, desc="🎯 Alineando transcripción...")
        
        if align_model is None or result["language"] != getattr(align_metadata, "language_code", None):
            align_model, align_metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result_aligned = whisperx.align(partial_segments, align_model, align_metadata, audio, device)
        
        # --- 4. Diarización ---
        progress(0.8, desc="👥 Identificando hablantes...")
        
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
        if num_speakers is not None and str(num_speakers).strip() != "":
            diarize_segments = diarize_model(audio, num_speakers=int(num_speakers))
        else:
            diarize_segments = diarize_model(audio)
        result_final = whisperx.assign_word_speakers(diarize_segments, result_aligned)
        
        # --- 5. Guardar transcripción con speakers en archivo ---
        progress(0.9, desc="💾 Guardando transcripción...")
        
        with open("transcription.txt", "w", encoding="utf-8") as file:
            for seg in result_final["segments"]:
                if "speaker" in seg:
                    speaker = seg["speaker"]
                    text = seg["text"].strip()
                    if text:
                        file.write(f"{speaker}: {text}\n")
        
        # --- 6. Guardar en DataFrame ---
        progress(0.95, desc="📊 Procesando resultados...")
        
        audioDF = pd.DataFrame(result_final["segments"])
        current_audioDF = audioDF.copy()
        groupedDF = group_consecutive_speakers(audioDF)
        
        progress(1.0, desc="✅ Transcripción completada exitosamente")
        return groupedDF
        
    except Exception as e:
        progress(0, desc=f"❌ Error en transcripción: {str(e)}")
        raise e

def group_consecutive_speakers(audioDF):
    """
    Groups consecutive speaker segments into single sentences, adjusting 'end' time and 'words'.
    """
    if not isinstance(audioDF, pd.DataFrame) or \
       not all(col in audioDF.columns for col in ['speaker', 'start', 'end', 'text', 'words']):
        raise ValueError("Input must be a pandas DataFrame with 'speaker', 'start', 'end', 'text', and 'words' columns.")

    grouped_segments = []
    current_speaker = None
    current_start = None
    current_end = None
    current_text = ""
    current_words = []

    for _, row in audioDF.iterrows():
        text = row['text'].strip()
        if row['speaker'] != current_speaker:
            if current_speaker is not None:
                grouped_segments.append({
                    'speaker': current_speaker,
                    'start': current_start,
                    'end': current_end,
                    'text': current_text.strip(),
                    'words': current_words
                })
            current_speaker = row['speaker']
            current_start = row['start']
            current_end = row['end']
            current_text = text
            current_words = row['words']
        else:
            current_end = row['end']
            current_text += " " + text
            current_words.extend(row['words'])

    # Agrega el último segmento
    if current_speaker is not None:
        grouped_segments.append({
            'speaker': current_speaker,
            'start': current_start,
            'end': current_end,
            'text': current_text.strip(),
            'words': current_words
        })

    return pd.DataFrame(grouped_segments)

def extract_speaker_samples(audioDF, audio_path, min_duration=2.0):
    """
    Extrae muestras de audio de cada speaker para identificación
    ORDENADOS POR PRIMERA APARICIÓN EN EL TIEMPO
    """
    # Cargar el archivo de audio completo
    audio = AudioSegment.from_file(audio_path)
    
    # Encontrar la primera aparición de cada speaker (ordenado por tiempo)
    first_appearances = {}
    for idx, row in audioDF.iterrows():
        speaker = row['speaker']
        if speaker not in first_appearances:
            first_appearances[speaker] = row['start']
    
    # Ordenar speakers por primera aparición temporal
    speakers_ordered = sorted(first_appearances.keys(), key=lambda x: first_appearances[x])
    
    # Crear lista temporal: índice, orador, inicio, fin, duración
    temp_list = [
        (idx, row['speaker'], row['start'], row['end'], row['end'] - row['start'])
        for idx, row in audioDF.iterrows()
    ]
    
    # Buscar el primer segmento > min_duration segundos por orador EN ORDEN TEMPORAL
    speaker_audio_files = {}
    orators = []
    
    for speaker in speakers_ordered:  # Procesar en orden temporal
        for idx, spk, start, end, duration in temp_list:
            if spk == speaker and duration > min_duration and speaker not in speaker_audio_files:
                start_ms = int(start * 1000)
                end_ms = int(end * 1000)
                segment = audio[start_ms:end_ms]
                
                # Crear archivo temporal
                temp_file = f"temp_segment_{speaker}.wav"
                segment.export(temp_file, format="wav")
                speaker_audio_files[speaker] = temp_file
                orators.append(speaker)
                break  # Encontramos una muestra válida para este speaker
    
    return orators, speaker_audio_files

def update_speaker_names(audioDF, orator_names):
    """
    Actualiza el DataFrame con los nuevos nombres de los oradores
    """
    updated_audioDF = audioDF.copy()
    
    for index, row in updated_audioDF.iterrows():
        speaker = row['speaker']
        if speaker in orator_names and orator_names[speaker].strip():
            updated_audioDF.at[index, 'speaker'] = orator_names[speaker]
    
    return updated_audioDF

def remove_stopwords_es(text):
    # Tokenizamos
    tokens = word_tokenize(text.lower(), language='spanish')
    cleaned_tokens = []
    
    for token in tokens:
        # Quitar puntuación innecesaria
        word_clean = token.strip("".join(punctuation_to_strip))
        # Conservar si no es stopword ni vacío
        if word_clean and word_clean not in stop_words:
            cleaned_tokens.append(word_clean)
    
    # Reconstruir texto y unir signos sueltos a la palabra anterior
    texto = ' '.join(cleaned_tokens)
    # Quitar espacios antes de los signos de admiración/interrogación
    texto = re.sub(r'\s+([?!¡¿])', r'\1', texto)
    return texto

def apply_text_filters(audioDF, remove_stopwords=False, min_tokens=4):
    """
    Aplica filtros de texto al DataFrame
    """
    # Calcular duración
    audioDF['duration'] = audioDF['end'] - audioDF['start']
    
    if remove_stopwords:
        # Aplicar limpieza de stopwords
        audioDF["text_clean"] = audioDF["text"].apply(remove_stopwords_es)
        # Filtrar filas vacías
        audioDF = audioDF[audioDF['text_clean'].str.strip().astype(bool)]
        # Filtrar por cantidad mínima de tokens
        audioDF = audioDF[audioDF['text_clean'].apply(lambda x: len(str(x).split()) >= min_tokens)]
    else:
        # Sin limpieza, solo aplicar filtro de tokens mínimos al texto original
        audioDF["text_clean"] = audioDF["text"]  # Mantener texto original
        audioDF = audioDF[audioDF['text'].apply(lambda x: len(str(x).split()) >= min_tokens)]
    
    return audioDF

def analyze_sentiments(audioDF, language="es", progress=None):
    """
    Analiza sentimientos y emociones del DataFrame usando pysentimiento con progress bar
    """
    global lenguaje
    lenguaje = language
    
    if progress:
        progress(0, desc="🧠 Iniciando análisis de sentimientos...")
    
    try:
        sentiment_analyzer = create_analyzer(task="sentiment", lang=lenguaje)
        emotion_analyzer = create_analyzer(task="emotion", lang=lenguaje)
        
        total_rows = len(audioDF)
        
        if progress:
            progress(0.1, desc="🧠 Analizando sentimientos...")
        
        # Analizar sentimientos con progress
        sentiment_results = []
        for i, text in enumerate(audioDF['text_clean']):
            sentiment_results.append(sentiment_analyzer.predict(text).output)
            if i % 10 == 0 and progress:  # Update every 10 items
                progress(0.1 + 0.4 * (i / total_rows), desc=f"🧠 Analizando sentimientos ({i+1}/{total_rows})...")
        
        audioDF['sentimiento'] = sentiment_results
        
        if progress:
            progress(0.5, desc="😊 Analizando emociones...")
        
        # Analizar emociones con progress
        emotion_results = []
        emotion_scores = []
        for i, text in enumerate(audioDF['text_clean']):
            result = emotion_analyzer.predict(text)
            emotion_results.append(result.output)
            emotion_scores.append(result.probas)
            if i % 10 == 0 and progress:  # Update every 10 items
                progress(0.5 + 0.4 * (i / total_rows), desc=f"😊 Analizando emociones ({i+1}/{total_rows})...")
        
        audioDF['emocion'] = emotion_results
        audioDF['emocion_score'] = emotion_scores
        
        # Análisis de scores de sentimientos
        if progress:
            progress(0.9, desc="📊 Finalizando análisis...")
        sentiment_scores = []
        for text in audioDF['text_clean']:
            sentiment_scores.append(sentiment_analyzer.predict(text).probas)
        audioDF['sentimiento_score'] = sentiment_scores
        
        if progress:
            progress(1.0, desc="✅ Análisis de sentimientos completado")
        
        return audioDF
    except Exception as e:
        if progress:
            progress(0, desc=f"❌ Error en análisis: {str(e)}")
        print(f"Error en análisis de sentimientos: {e}")
        return audioDF

def traducir(texto):
    try:
        return GoogleTranslator(source='auto', target='en').translate(texto)
    except Exception as e:
        print(f"Error traduciendo: {texto} -> {e}")
        return None

def remove_stopwords_en(text):
    # Definir stopwords en inglés
    stop_words_en = set(stopwords.words('english'))
    # Puntuación a eliminar (excepto signos de exclamación/interrogación)
    punctuation_to_strip = set(string.punctuation) | {"«", "»", "…"}
    punctuation_to_strip -= {"¡", "¿", "!", "?"}
    
    # Tokenizamos
    tokens = word_tokenize(text.lower(), language='english')
    cleaned_tokens = []
    
    for token in tokens:
        # Quitar puntuación innecesaria
        word_clean = token.strip("".join(punctuation_to_strip))
        # Conservar si no es stopword ni vacío
        if word_clean and word_clean not in stop_words_en:
            cleaned_tokens.append(word_clean)
    
    # Reconstruir texto y unir signos sueltos a la palabra anterior
    texto = ' '.join(cleaned_tokens)
    # Quitar espacios antes de los signos de admiración/interrogación
    texto = re.sub(r'\s+([?!¡¿])', r'\1', texto)
    return texto

def analyze_complementary_english(audioDF, progress=None):
    """
    Análisis complementario en inglés para frases con emoción 'others' o sentimiento 'NEU'
    """
    if progress:
        progress(0, desc="🌐 Iniciando análisis complementario en inglés...")
    
    try:
        # Hacer una copia para trabajar
        df_work = audioDF.copy()
        
        # Filtrar casos que necesitan análisis complementario
        mask = (df_work['emocion'] == 'others') | (df_work['sentimiento'] == 'NEU')
        
        if not mask.any():
            if progress:
                progress(1.0, desc="✅ No se requiere análisis complementario")
            print("No hay frases que requieran análisis complementario en inglés")
            return audioDF
        
        indices_to_analyze = df_work[mask].index.tolist()
        total_items = len(indices_to_analyze)
        
        if progress:
            progress(0.1, desc=f"🌐 Procesando {total_items} textos...")
        
        # Inicializar columnas nuevas si no existen
        for col in ['text_translated', 'text_clean_en', 'sentimiento_en', 'emocion_en', 'sentimiento_score_en', 'emocion_score_en']:
            if col not in df_work.columns:
                df_work[col] = None
        
        # Crear analizadores una sola vez
        sentiment_analyzer_en = create_analyzer(task="sentiment", lang="en")
        emotion_analyzer_en = create_analyzer(task="emotion", lang="en")
        
        # Procesar cada frase individualmente
        for count, idx in enumerate(indices_to_analyze):
            try:
                # Verificar que el índice existe
                if idx not in df_work.index:
                    print(f"Índice {idx} no existe en el DataFrame")
                    continue
                
                original_text = df_work.at[idx, 'text']
                
                # Traducir
                translated_text = traducir(original_text)
                if translated_text is None or translated_text.strip() == '':
                    continue
                
                df_work.at[idx, 'text_translated'] = translated_text
                
                # Limpiar stopwords en inglés
                clean_text_en = remove_stopwords_en(translated_text)
                
                # Verificar cantidad mínima de tokens
                if len(clean_text_en.split()) < 4:
                    continue
                
                df_work.at[idx, 'text_clean_en'] = clean_text_en
                
                # Predecir sentimientos y emociones
                sent_result = sentiment_analyzer_en.predict(clean_text_en)
                emo_result = emotion_analyzer_en.predict(clean_text_en)
                
                df_work.at[idx, 'sentimiento_en'] = sent_result.output
                df_work.at[idx, 'emocion_en'] = emo_result.output
                df_work.at[idx, 'sentimiento_score_en'] = sent_result.probas
                df_work.at[idx, 'emocion_score_en'] = emo_result.probas
                
                # Actualizar progreso
                if progress:
                    progress_val = 0.1 + 0.8 * (count + 1) / total_items
                    progress(progress_val, desc=f"🌐 Procesando texto {count+1}/{total_items}...")
                
            except Exception as e:
                traceback.print_exc()
                continue
        
        if progress:
            progress(1.0, desc="✅ Análisis complementario completado")
        
        return df_work
        
    except Exception as e:
        if progress:
            progress(0, desc=f"❌ Error en análisis complementario: {str(e)}")
        print(f"Error general en análisis complementario en inglés: {e}")
        traceback.print_exc()
        return audioDF

def analisis_baseGoEmotions_samlowe(text):
    try:
        if classifier is None:
            print("Modelo roberta no disponible, usando 'neutral'")
            return 'neutral'
        
        # El modelo devuelve una lista de diccionarios
        model_outputs = classifier(text)
        
        # Debug: ver qué devuelve el modelo
        print(f"DEBUG roberta output: {model_outputs}")
        
        # Manejar diferentes formatos de respuesta
        if isinstance(model_outputs, list) and len(model_outputs) > 0:
            if isinstance(model_outputs[0], dict) and 'label' in model_outputs[0]:
                return model_outputs[0]['label']
            elif isinstance(model_outputs[0], list) and len(model_outputs[0]) > 0:
                if isinstance(model_outputs[0][0], dict) and 'label' in model_outputs[0][0]:
                    return model_outputs[0][0]['label']
        
        print(f"Formato inesperado de roberta: {type(model_outputs)}")
        return 'neutral'
        
    except Exception as e:
        print(f"Error en análisis de emociones roberta: {e}")
        return 'neutral'

# Mapping para agrupar emociones
emotion_mapping = {
    # anger group
    'anger': 'anger',
    'annoyance': 'anger',
    'disapproval': 'anger',
    # disgust group
    'disgust': 'disgust',
    # fear group
    'fear': 'fear',
    'nervousness': 'fear',
    # joy group
    'admiration': 'joy',
    'amusement': 'joy',
    'approval': 'joy',
    'caring': 'joy',
    'desire': 'joy',
    'excitement': 'joy',
    'gratitude': 'joy',
    'joy': 'joy',
    'love': 'joy',
    'pride': 'joy',
    'optimism': 'joy',
    'relief': 'joy',
    # sadness group
    'sadness': 'sadness',
    'disappointment': 'sadness',
    'embarrassment': 'sadness',
    'grief': 'sadness',
    'remorse': 'sadness',
    # surprise group
    'confusion': 'surprise',
    'curiosity': 'surprise',
    'realization': 'surprise',
    'surprise': 'surprise',
    # neutral group
    'neutral': 'neutral'
}

def apply_roberta_analysis_and_replace(audioDF, progress=None):
    """
    Aplica análisis con roberta-base-go_emotions y reemplaza valores NEU y others
    """
    if progress:
        progress(0, desc="🤖 Iniciando análisis Roberta...")
    
    try:
        # Trabajar con una copia
        df_work = audioDF.copy()
        
        # Verificar si existe la columna text_translated
        if 'text_translated' not in df_work.columns:
            if progress:
                progress(0.5, desc="⚠️ Sin textos traducidos, aplicando cambios básicos...")
            print("No hay columna 'text_translated', saltando análisis roberta")
            # Solo aplicar reemplazo básico de 'others' -> 'neutral'
            df_work.loc[df_work["emocion"] == "others", "emocion"] = "neutral"
            if progress:
                progress(1.0, desc="✅ Cambios básicos aplicados")
            return df_work
        
        # Filtrar casos que tienen traducción (fueron procesados en inglés)
        mask_translated = df_work['text_translated'].notna()
        
        if not mask_translated.any():
            if progress:
                progress(0.5, desc="⚠️ Sin textos para análisis Roberta...")
            print("No hay textos traducidos para analizar con roberta")
            # Solo aplicar reemplazo básico de 'others' -> 'neutral'
            df_work.loc[df_work["emocion"] == "others", "emocion"] = "neutral"
            if progress:
                progress(1.0, desc="✅ Cambios básicos aplicados")
            return df_work
        
        indices_translated = df_work[mask_translated].index.tolist()
        total_items = len(indices_translated)
        
        if progress:
            progress(0.1, desc=f"🤖 Procesando {total_items} textos con Roberta...")
        
        print(f"Aplicando análisis roberta a {len(indices_translated)} textos traducidos...")
        
        # Inicializar columnas si no existen
        if 'emotion_roberta' not in df_work.columns:
            df_work['emotion_roberta'] = None
        if 'sentiment_en_pysentimiento' not in df_work.columns:
            df_work['sentiment_en_pysentimiento'] = None
        
        # Crear analizador de sentimientos en inglés
        sentiment_analyzer_en = create_analyzer(task="sentiment", lang="en")
        
        # Procesar cada texto traducido
        for count, idx in enumerate(indices_translated):
            try:
                translated_text = df_work.at[idx, 'text_translated']
                clean_text_en = df_work.at[idx, 'text_clean_en'] if 'text_clean_en' in df_work.columns else None
                
                if not translated_text or pd.isna(translated_text):
                    continue
                
                # Análisis de emociones con roberta
                emotion_roberta = analisis_baseGoEmotions_samlowe(translated_text)
                emotion_mapped = emotion_mapping.get(emotion_roberta, 'neutral')
                df_work.at[idx, 'emotion_roberta'] = emotion_mapped
                
                # Análisis de sentimiento con pysentimiento en inglés
                if clean_text_en and len(str(clean_text_en).strip()) > 0:
                    sentiment_en = sentiment_analyzer_en.predict(clean_text_en).output
                    df_work.at[idx, 'sentiment_en_pysentimiento'] = sentiment_en
                else:
                    # Si no hay texto limpio, usar el traducido
                    sentiment_en = sentiment_analyzer_en.predict(translated_text).output
                    df_work.at[idx, 'sentiment_en_pysentimiento'] = sentiment_en
                
                print(f"✓ Roberta análisis índice {idx}: emotion={emotion_mapped}, sentiment={sentiment_en}")
                
                # Actualizar progreso
                if progress:
                    progress_val = 0.1 + 0.7 * (count + 1) / total_items
                    progress(progress_val, desc=f"🤖 Procesando con Roberta {count+1}/{total_items}...")
                
            except Exception as e:
                print(f"Error procesando índice {idx} con roberta: {e}")
                continue
        
        # Aplicar reemplazos condicionales
        if progress:
            progress(0.9, desc="🔄 Aplicando reemplazos...")
        
        print("Aplicando reemplazos condicionales...")
        
        # Reemplazar sentimiento solo si es 'NEU' y hay análisis en inglés
        if 'sentiment_en_pysentimiento' in df_work.columns:
            mask_replace_sentiment = (
                (df_work["sentimiento"] == "NEU") & 
                df_work["sentiment_en_pysentimiento"].notna()
            )
            df_work.loc[mask_replace_sentiment, "sentimiento"] = df_work.loc[mask_replace_sentiment, "sentiment_en_pysentimiento"]
        
        # Reemplazar emoción solo si es 'others' y hay análisis roberta
        if 'emotion_roberta' in df_work.columns:
            mask_replace_emotion = (
                (df_work["emocion"] == "others") & 
                df_work["emotion_roberta"].notna()
            )
            df_work.loc[mask_replace_emotion, "emocion"] = df_work.loc[mask_replace_emotion, "emotion_roberta"]
        
        # Reemplazar 'others' restantes con 'neutral'
        df_work.loc[df_work["emocion"] == "others", "emocion"] = "neutral"
        
        print("✓ Reemplazos condicionales aplicados")
        
        if progress:
            progress(1.0, desc="✅ Análisis Roberta completado")
        
        return df_work
        
    except Exception as e:
        if progress:
            progress(0, desc=f"❌ Error en análisis Roberta: {str(e)}")
        print(f"Error en apply_roberta_analysis_and_replace: {e}")
        traceback.print_exc()
        # En caso de error, al menos aplicar el reemplazo básico
        audioDF.loc[audioDF["emocion"] == "others", "emocion"] = "neutral"
        return audioDF

def generate_analysis_plots(audioDF, progress=None):
    """
    Genera todos los gráficos de análisis con progress bar
    """
    if progress:
        progress(0, desc="📊 Iniciando generación de gráficos...")
    
    try:
        
        # Traducir las emociones
        df_plot = audioDF.copy()
        df_plot['emocion'] = df_plot['emocion'].replace({
            'joy': 'alegría',
            'surprise': 'sorpresa',
            'disgust': 'disgusto',
            'sadness': 'tristeza',
            'fear': 'miedo',
            'anger': 'enojo'
        })
        
        # Definir colores
        colores_sentimientos = {
            'POS': '#00A651',
            'NEG': '#F05A61',
            'NEU': '#FFCB08'
        }
        
        colores_emociones = {
            'enojo': '#F05A61',
            'alegría': '#FFCB08',
            'tristeza': '#2884C6',
            'miedo': '#00A651',
            'sorpresa': '#009ACE',
            'disgusto': '#8A73B3',
            'neutral': '#F7923D',
            'others': '#F7923D'
        }
        
        plot_files = []
        
        # 1. Duración por orador
        if progress:
            progress(0.05, desc="📊 Creando gráfico de duración por orador...")
        plt.figure(figsize=(10, 6))
        speaker_durations = df_plot.groupby('speaker')['duration'].sum()
        plt.bar(speaker_durations.index, speaker_durations.values)
        plt.xlabel('Orador')
        plt.ylabel('Duración total (segundos)')
        plt.title('Duración de la conversación por orador')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("duracion_orador.png", format="png", bbox_inches="tight")
        plot_files.append("duracion_orador.png")
        plt.close()
        
        # 2. Distribución de sentimientos por orador
        if progress:
            progress(0.15, desc="📊 Creando gráfico de sentimientos por orador...")
        sentiment_durations = df_plot.groupby(["speaker", "sentimiento"])["duration"].sum().unstack(fill_value=0)
        sentiments_percentages = sentiment_durations.div(sentiment_durations.sum(axis=1), axis=0) * 100
        
        plt.figure(figsize=(10, 6))
        ax = sentiments_percentages.plot(
            kind='bar',
            stacked=True,
            color=[colores_sentimientos.get(col, '#CCCCCC') for col in sentiment_durations.columns]
        )
        plt.xlabel('Orador')
        plt.ylabel('Porcentaje (%)')
        plt.title('Distribución de Sentimientos por Orador (según tiempo hablado)')
        plt.xticks(rotation=45)
        plt.legend(title="Sentimientos", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("sentimiento_orador.png", format="png", bbox_inches="tight")
        plot_files.append("sentimiento_orador.png")
        plt.close()
        
        # 3. Distribución de emociones por orador
        if progress:
            progress(0.25, desc="📊 Creando gráfico de emociones por orador...")
        emotions_durations = df_plot.groupby(["speaker", "emocion"])["duration"].sum().unstack(fill_value=0)
        emotions_percentages = emotions_durations.div(emotions_durations.sum(axis=1), axis=0) * 100
        
        plt.figure(figsize=(10, 6))
        ax = emotions_percentages.plot(
            kind='bar',
            stacked=True,
            color=[colores_emociones.get(col, '#CCCCCC') for col in emotions_durations.columns]
        )
        plt.xlabel('Orador')
        plt.ylabel('Porcentaje (%)')
        plt.title('Distribución de Emociones por Orador (según tiempo hablado)')
        plt.xticks(rotation=45)
        plt.legend(title="Emoción", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("emocion_orador.png", format="png", bbox_inches="tight")
        plot_files.append("emocion_orador.png")
        plt.close()
        
        # 4. Pie chart sentimientos
        if progress:
            progress(0.35, desc="📊 Creando gráfico circular de sentimientos...")
        sentiment_durations_total = df_plot.groupby("sentimiento")["duration"].sum()
        sentiment_percentages = (sentiment_durations_total / sentiment_durations_total.sum()) * 100
        colors = [colores_sentimientos.get(sent, "#CCCCCC") for sent in sentiment_percentages.index]
        
        plt.figure(figsize=(8, 8))
        plt.pie(sentiment_percentages, labels=sentiment_percentages.index, autopct='%1.1f%%', startangle=140, colors=colors)
        plt.title("Distribución de Sentimientos en toda la interacción")
        plt.tight_layout()
        plt.savefig("sentimiento_total.png", format="png", bbox_inches="tight")
        plot_files.append("sentimiento_total.png")
        plt.close()
        
        # 5. Pie chart emociones
        if progress:
            progress(0.45, desc="📊 Creando gráfico circular de emociones...")
        emotions_durations_total = df_plot.groupby("emocion")["duration"].sum()
        emotions_percentages = (emotions_durations_total / emotions_durations_total.sum()) * 100
        colors = [colores_emociones.get(emo, "#CCCCCC") for emo in emotions_percentages.index]
        
        plt.figure(figsize=(8, 8))
        plt.pie(emotions_percentages, labels=emotions_percentages.index, autopct='%1.1f%%', startangle=140, colors=colors)
        plt.title("Distribución de Emociones en toda la interacción")
        plt.tight_layout()
        plt.savefig("emocion_total.png", format="png", bbox_inches="tight")
        plot_files.append("emocion_total.png")
        plt.close()
        
        # 6. Sentimiento a lo largo del tiempo por speaker
        if progress:
            progress(0.55, desc="📊 Creando línea temporal de sentimientos por speaker...")
        sentimiento_map = {'NEG': -1, 'NEU': 0, 'POS': 1}
        df_plot['sentimiento_num'] = df_plot['sentimiento'].map(sentimiento_map)
        df_plot['mid_time'] = (df_plot['start'] + df_plot['end']) / 2
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_plot, x='mid_time', y='sentimiento_num', hue='speaker', marker='o')
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.title('Sentimiento a lo largo del tiempo por participante')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Sentimiento')
        plt.yticks([-1, 0, 1], ['Negativo', 'Neutral', 'Positivo'])
        plt.legend(title='Orador')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("sentimiento_tiempo_speakers.png", format="png", bbox_inches="tight")
        plot_files.append("sentimiento_tiempo_speakers.png")
        plt.close()
        
        # 7. Sentimiento a lo largo del tiempo general
        if progress:
            progress(0.65, desc="📊 Creando línea temporal general de sentimientos...")
        plt.figure(figsize=(12, 4))
        sns.lineplot(data=df_plot.sort_values('mid_time'), x='mid_time', y='sentimiento_num', marker='o', color='black')
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.title('Sentimiento a lo largo del tiempo (interacción completa)')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Sentimiento')
        plt.yticks([-1, 0, 1], ['Negativo', 'Neutral', 'Positivo'])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("sentimiento_tiempo.png", format="png", bbox_inches="tight")
        plot_files.append("sentimiento_tiempo.png")
        plt.close()
        
        # 8. WordCloud
        if progress:
            progress(0.75, desc="📊 Creando nube de palabras...")
        text = " ".join(df_plot['text_clean'].astype(str))
        if text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig("wordcloud.png", format="png", bbox_inches="tight")
            plot_files.append("wordcloud.png")
            plt.close()
        
        # 9. Palabras más frecuentes
        if progress:
            progress(0.85, desc="📊 Creando gráfico de palabras frecuentes...")
        text_lower = " ".join(df_plot['text_clean'].astype(str)).lower()
        words = re.findall(r'\b\w+\b', text_lower)
        word_counts = Counter(words)
        most_common_words = word_counts.most_common(20)
        
        if most_common_words:
            plt.figure(figsize=(10, 6))
            words_list, counts_list = zip(*most_common_words)
            plt.bar(words_list, counts_list)
            plt.xlabel("Palabras")
            plt.ylabel("Frecuencia")
            plt.title("Las 20 palabras más frecuentes")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig("palabras_frecuentes.png", format="png", bbox_inches="tight")
            plot_files.append("palabras_frecuentes.png")
            plt.close()
        
        if progress:
            progress(1.0, desc="✅ Todos los gráficos generados exitosamente")
        
        return plot_files
        
    except Exception as e:
        if progress:
            progress(0, desc=f"❌ Error generando gráficos: {str(e)}")
        print(f"Error generando gráficos: {e}")
        return []

def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# 🔹 Generador del reporte final
def generar_reporte_pdf_reportlab(audioDF, plot_files, output_path="Informe_AS-EC.pdf"):
    try:
        
        # Configurar documento con márgenes más amplios
        doc = SimpleDocTemplate(output_path, pagesize=A4, 
                              leftMargin=0.75*inch, rightMargin=0.75*inch,
                              topMargin=1*inch, bottomMargin=1*inch)
        styles = getSampleStyleSheet()
        
        # Crear estilos personalizados
        title_style = ParagraphStyle('CustomTitle',
                                   parent=styles['Title'],
                                   fontSize=24,
                                   spaceAfter=30,
                                   textColor=colors.HexColor('#2C3E50'),
                                   alignment=TA_CENTER)
        
        subtitle_style = ParagraphStyle('CustomSubtitle',
                                      parent=styles['Heading2'],
                                      fontSize=16,
                                      textColor=colors.HexColor('#34495E'),
                                      spaceAfter=20,
                                      spaceBefore=15)
        
        summary_style = ParagraphStyle('Summary',
                                     parent=styles['Normal'],
                                     fontSize=11,
                                     textColor=colors.HexColor('#2C3E50'),
                                     alignment=TA_JUSTIFY,
                                     spaceAfter=12)
        
        story = []
        
        # PORTADA
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph("INFORME DE ANÁLISIS", title_style))
        story.append(Paragraph("Sentimientos y Emociones en Audio", title_style))
        story.append(Spacer(1, 1*inch))
        
        # Información del reporte
        info_data = [
            ["Fecha de generación:", time.strftime('%d/%m/%Y')],
            ["Hora:", time.strftime('%H:%M')],
            ["Total de segmentos:", str(len(audioDF))],
            ["Participantes:", str(len(audioDF['speaker'].unique()))],
            ["Duración total:", f"{audioDF['duration'].sum():.1f} segundos"]
        ]
        
        info_table = Table(info_data, colWidths=[2.5*inch, 2.5*inch])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        story.append(info_table)
        story.append(PageBreak())
        
        # RESUMEN EJECUTIVO
        story.append(Paragraph("RESUMEN EJECUTIVO", subtitle_style))
        
        # Calcular estadísticas clave
        total_duracion = audioDF['duration'].sum()
        participantes = audioDF['speaker'].unique()
        sentiment_dist = audioDF.groupby('sentimiento')['duration'].sum()
        emotion_dist = audioDF.groupby('emocion')['duration'].sum()
        
        # Participante más activo
        speaker_time = audioDF.groupby('speaker')['duration'].sum()
        most_active = speaker_time.idxmax()
        most_active_pct = (speaker_time.max() / total_duracion) * 100
        
        # Sentimiento predominante
        main_sentiment = sentiment_dist.idxmax()
        main_sentiment_pct = (sentiment_dist.max() / total_duracion) * 100
        
        # Emoción predominante
        main_emotion = emotion_dist.idxmax()
        main_emotion_pct = (emotion_dist.max() / total_duracion) * 100
        
        resumen_text = f"""
        Este informe presenta el análisis de sentimientos y emociones de una conversación con {len(participantes)} participantes 
        y una duración total de {total_duracion:.1f} segundos ({total_duracion/60:.1f} minutos).
        
        <br/>
        <b>Hallazgos principales:</b><br/>
        • <b>Participante más activo:</b> {most_active} ({most_active_pct:.1f}% del tiempo total)<br/>
        • <b>Sentimiento predominante:</b> {main_sentiment} ({main_sentiment_pct:.1f}% del tiempo)<br/>
        • <b>Emoción predominante:</b> {main_emotion} ({main_emotion_pct:.1f}% del tiempo)<br/>
        • <b>Total de segmentos analizados:</b> {len(audioDF)}
        """
        
        story.append(Paragraph(resumen_text, summary_style))
        story.append(Spacer(1, 20))
        
        # TABLA DETALLADA DE PARTICIPANTES 
        story.append(Paragraph("ANÁLISIS POR PARTICIPANTE TOTAL", subtitle_style))
        
        participant_summary = audioDF.groupby('speaker').agg({
            'duration': ['sum', 'count'],
            'sentimiento': lambda x: x.value_counts().index[0],  # más frecuente
            'emocion': lambda x: x.value_counts().index[0]       # más frecuente
        }).round(1)
        
        participant_data = [["Participante", "Tiempo total (s)", "Intervenciones", "Sentimiento Principal", "Emoción Principal"]]
        
        for speaker in participant_summary.index:
            tiempo = participant_summary.loc[speaker, ('duration', 'sum')]
            intervenciones = participant_summary.loc[speaker, ('duration', 'count')]
            sentimiento = participant_summary.loc[speaker, ('sentimiento', '<lambda>')]
            emocion = participant_summary.loc[speaker, ('emocion', '<lambda>')]
            
            participant_data.append([
                str(speaker),
                f"{tiempo:.1f}",
                str(intervenciones),
                str(sentimiento),
                str(emocion)
            ])
        
        participant_table = Table(participant_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1.2*inch, 1.2*inch])
        participant_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ECF0F1')),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(participant_table)
        
        # TABLA DETALLADA DE INTERVENCIONES
        story.append(Paragraph("ANÁLISIS POR INTERVENCIÓN", subtitle_style))

        intervencion_data = [["# Intervención", "Participante", "Duración (s)", "Sentimiento", "Emoción"]]

        for idx, row in audioDF.iterrows():
            intervencion_data.append([
                str(idx + 1),   # número de intervención
                str(row['speaker']),
                f"{row['duration']:.1f}",
                str(row['sentimiento']),
                str(row['emocion'])
            ])

        intervencion_table = Table(intervencion_data, colWidths=[1*inch, 1.5*inch, 1*inch, 1.5*inch, 1.5*inch])
        intervencion_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27AE60')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F9F9F9')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        story.append(intervencion_table)
        story.append(PageBreak())

        # GRÁFICOS ORGANIZADOS
        story.append(Paragraph("ANÁLISIS VISUAL", subtitle_style))
        
        # Organizar gráficos por categorías
        graficos_organizados = [
            ("Análisis de Participación", ["duracion_orador.png"]),
            ("Distribución de Sentimientos", ["sentimiento_total.png", "sentimiento_orador.png"]),
            ("Distribución de Emociones", ["emocion_total.png", "emocion_orador.png"]),
            ("Evolución Temporal", ["sentimiento_tiempo.png", "sentimiento_tiempo_speakers.png"]),
            ("Análisis de Contenido", ["wordcloud.png", "palabras_frecuentes.png"])
        ]
        
        chart_titles = {
            "duracion_orador.png": "Duración por Participante",
            "sentimiento_orador.png": "Sentimientos por Participante", 
            "emocion_orador.png": "Emociones por Participante",
            "sentimiento_total.png": "Distribución General de Sentimientos",
            "emocion_total.png": "Distribución General de Emociones",
            "sentimiento_tiempo.png": "Evolución del Sentimiento",
            "sentimiento_tiempo_speakers.png": "Sentimientos por Participante en el Tiempo",
            "wordcloud.png": "Nube de Palabras",
            "palabras_frecuentes.png": "Palabras Más Frecuentes"
        }
        
        # Crear diccionario de archivos disponibles
        available_plots = {}
        for plot_file in plot_files:
            if os.path.exists(plot_file):
                filename = os.path.basename(plot_file)
                available_plots[filename] = plot_file
        
        for categoria, archivos in graficos_organizados:
            # Verificar si hay gráficos disponibles en esta categoría
            archivos_disponibles = [arch for arch in archivos if arch in available_plots]
            if not archivos_disponibles:
                continue
                
            story.append(Paragraph(categoria, ParagraphStyle('CategoryTitle',
                                                            parent=styles['Heading3'],
                                                            fontSize=14,
                                                            textColor=colors.HexColor('#2980B9'),
                                                            spaceAfter=15,
                                                            spaceBefore=20)))
            
            for archivo in archivos_disponibles:
                if archivo in available_plots:
                    title = chart_titles.get(archivo, archivo.replace('.png', '').replace('_', ' ').title())
                    
                    story.append(Paragraph(title, ParagraphStyle('ChartTitle',
                                                                parent=styles['Heading4'],
                                                                fontSize=12,
                                                                textColor=colors.HexColor('#34495E'),
                                                                spaceAfter=10)))
                    try:
                        img = Image(available_plots[archivo], width=5.5*inch, height=3.3*inch)
                        story.append(img)
                        story.append(Spacer(1, 20))
                    except Exception as e:
                        print(f"Error añadiendo {archivo}: {e}")
                        story.append(Paragraph(f"[Error cargando gráfico: {title}]", styles['Normal']))
                        story.append(Spacer(1, 10))
        
        # Construir el documento
        doc.build(story)
        return output_path
        
    except Exception as e:
        print(f"Error generando PDF: {e}")
        return None 

def generar_reporte(analysis_data):
    if analysis_data is None:
        return gr.update(visible=False)
    try:
        audioDF = analysis_data.get("filtered_audioDF")
        plot_files = analysis_data.get("plot_files", [])
        
        pdf_path = generar_reporte_pdf_reportlab(audioDF, plot_files)
        
        if pdf_path:
            return gr.update(value=pdf_path, visible=True)
        else:
            return gr.update(visible=False)
        
    except Exception as e:
        return gr.update(visible=False)


def generate_csv_from_analysis(analysis_data):
    """
    Genera un archivo CSV con los datos de análisis
    """
    if analysis_data is None:
        return None, gr.update(value="❌ No hay datos para exportar")
    
    try:
        
        sentiment_data = analysis_data.get("sentiment_data", [])
        if not sentiment_data:
            return None, gr.update(value="❌ No hay datos de sentimientos para exportar")
        
        # Ruta en la carpeta temporal con nombre fijo
        file_path = os.path.join(tempfile.gettempdir(), "Tabla_analisis_AS-EC.csv")
        
        # Escribir CSV
        with open(file_path, mode='w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
            
            # Headers
            headers = ["Participante", "Texto_Original", "Texto_Limpio", "Sentimiento", "Emocion", "Duracion", "Confianza"]
            csv_writer.writerow(headers)
            
            # Datos
            for row in sentiment_data:
                csv_writer.writerow(row)
        
        return file_path
        
    except Exception as e:
        return None, gr.update(value=f"❌ Error generando CSV: {str(e)}")

def create_audio_analyzer_app():
    with gr.Blocks(theme=gr.themes.Soft(), title="Analizador de Audio Multi-Página") as demo:
        
        # CSS personalizado para mejor apariencia
        gr.HTML("""
        <style>
                @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;700&display=swap');
                .main-title {
                    font-family: 'Open Sans', sans-serif;
                    font-size: 24px;
                    font-weight: bold;
                    text-transform: uppercase;
                    color: white;
                    text-align: center;
                    margin-bottom: 10px;
                }
                .subtitle {
                    font-family: 'Open Sans', sans-serif;
                    font-size: 14px;
                    color: rgba(255,255,255,0.9);
                    text-align: center;
                    margin-bottom: 30px;
                }
                .page-container {
                    min-height: 500px;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 10px;
                    margin: 10px 0;
                }
                .page-title {
                    color: white;
                    text-align: center;
                    font-size: 2.5em;
                    margin-bottom: 30px;
                }
                .progress-container {
                    background: rgba(255,255,255,0.1);
                    border-radius: 10px;
                    padding: 15px;
                    margin: 20px 0;
                }
                .step-indicator {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 30px;
                    padding: 0 20px;
                }
                .step {
                    padding: 12px 24px;
                    border-radius: 25px;
                    color: white;
                    font-weight: bold;
                    min-width: 150px;
                    text-align: center;
                }
                .step.active {
                    background-color: #4CAF50;
                    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4);
                }
                .step.completed {
                    background-color: #2196F3;
                }
                .step.pending {
                    background-color: rgba(255,255,255,0.2);
                }
                .btn-primary, button[role="button"].primary {
                    min-width: 200px !important;
                    height: 50px !important;
                    font-size: 16px !important;
                    font-weight: bold !important;
                }
                
                .tab-nav button, .tabs .tab-nav button, div[role="tablist"] button {
                    color: #fff !important;
                    background-color: rgba(255,255,255,0.2) !important;
                    border: 1px solid rgba(255,255,255,0.3) !important;
                    backdrop-filter: blur(10px);
                }

                .tab-nav button[aria-selected="true"], .tabs .tab-nav button[aria-selected="true"], 
                div[role="tablist"] button[aria-selected="true"] {
                    color: #fff !important;
                    background-color: rgba(255,255,255,0.4) !important;
                    border-color: rgba(255,255,255,0.6) !important;
                    box-shadow: 0 2px 10px rgba(255,255,255,0.3);
                }

                button[role="tab"] {
                    color: #fff !important;
                    background-color: rgba(255,255,255,0.2) !important;
                }

                button[role="tab"][aria-selected="true"] {
                    color: #fff !important;
                    background-color: rgba(255,255,255,0.4) !important;
                    font-weight: bold;
                }
        </style>
        """)
        
        # Estado para controlar la página actual
        current_page = gr.State(value=1)
        
        # Estados para almacenar datos entre páginas
        audio_data_state = gr.State()
        speakers_data_state = gr.State()
        final_analysis_state = gr.State()
        
        # Indicador de progreso visual
        step_indicator = gr.HTML("""
        <div class="main-title"> Analizador de Sentimientos y Emociones en Audio</div>
        <div class="step-indicator">
            <div class="step active">1. Carga de Audio</div>
            <div class="step pending">2. Transcripción e Identificación</div>
            <div class="step pending">3. Resultados</div>
        </div>
        """)
        
        # PÁGINA 1: Carga de Audio
        with gr.Column(visible=True, elem_classes=["page-container"]) as page1:
            gr.HTML('<h1 class="page-title">Carga y Procesamiento de Audio</h1>')
            
            with gr.Row():
                with gr.Column(scale=2):
                    audio_input = gr.Audio(
                        sources=["upload"], 
                        type="filepath", 
                        label="Seleccione su archivo de audio o video",
                    )
                
                with gr.Column(scale=1):
                    speakers_input = gr.Textbox(
                        label="Número de participantes (opcional)", 
                        placeholder="Ej: 2",
                        info="Déjelo vacío para detección automática"
                    )
                    language_input = gr.Dropdown(
                        choices=[("Español", "es"), ("Inglés", "en")], 
                        value="es", 
                        label="Idioma principal del audio"
                    )
                with gr.Column(scale=1):
                    with gr.Accordion("⚙️ Configuración Avanzada", open=False):
                        remove_stopwords_input = gr.Checkbox(
                            label="Filtrar palabras vacías", 
                            value=True,
                            info="Elimina palabras como 'el', 'la', 'de', etc."
                        )
                        min_tokens_input = gr.Number(
                            label="Mínimo de palabras por frase", 
                            value=4,
                            minimum=1,
                            maximum=20
                        )
            
            process_btn = gr.Button(
                "Procesar Audio", 
                variant="primary", 
                size="lg"
            )
            
            # Progress bar visible bajo el botón "Procesar Audio" - SOLO UNO
            with gr.Column(visible=False, elem_classes=["progress-container"]) as progress_container:
                gr.Markdown("### 📊 Progreso del Procesamiento")
                progress_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    label="🚀 Iniciando...",
                    interactive=False,
                    show_label=True
                )
        
        # PÁGINA 2: Identificación de Participantes
        with gr.Column(visible=False, elem_classes=["page-container"]) as page2:
            gr.HTML('<h1 class="page-title">Transcripción completada</h1>')
            
            with gr.Column():
                gr.Markdown("### Procesamiento Completado")
                transcript_preview = gr.Textbox(
                    label="Vista previa de transcripción",
                    lines=4,
                    interactive=False
                )
                speakers_detected = gr.Textbox(
                    label="Participantes detectados",
                    interactive=False
                )
            gr.HTML('<h1 class="page-title">Identificación de Participantes</h1>')
            gr.Markdown("""
            ### Escuche las muestras de voz y asigne nombres reales a cada participante
            Esto mejorará significativamente la legibilidad del análisis final.
            """)
            
            # Contenedor dinámico para speakers
            speaker_audio_1 = gr.Audio(label="Muestra Speaker 1", visible=False, interactive=False)
            speaker_name_1 = gr.Textbox(label="Nombre para Speaker 1", visible=False, interactive=True)
            speaker_audio_2 = gr.Audio(label="Muestra Speaker 2", visible=False, interactive=False)  
            speaker_name_2 = gr.Textbox(label="Nombre para Speaker 2", visible=False, interactive=True)
            speaker_audio_3 = gr.Audio(label="Muestra Speaker 3", visible=False, interactive=False)
            speaker_name_3 = gr.Textbox(label="Nombre para Speaker 3", visible=False, interactive=True)
            speaker_audio_4 = gr.Audio(label="Muestra Speaker 4", visible=False, interactive=False)
            speaker_name_4 = gr.Textbox(label="Nombre para Speaker 4", visible=False, interactive=True)
            
            with gr.Row():
                back_to_page1_btn = gr.Button(
                    "⬅️ Volver a Carga de Audio",
                    variant="secondary"
                )
                analyze_btn = gr.Button(
                    "Realizar Análisis Completo",
                    variant="primary",
                    size="lg"
                )
            
            # Progress bar para página 2 - SOLO UNO
            with gr.Column(visible=False, elem_classes=["progress-container"]) as progress2_container:
                gr.Markdown("### 📊 Progreso del Análisis")
                progress2_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    label="🔍 Esperando...",
                    interactive=False,
                    show_label=True
                )
        
        # PÁGINA 3: Resultados Finales
        with gr.Column(visible=False, elem_classes=["page-container"]) as page3:
            gr.HTML('<h1 class="page-title">Resultados del Análisis</h1>')
            
            with gr.Tabs():
                with gr.TabItem("Transcripción"):
                    final_transcript = gr.Textbox(
                        label="Transcripción completa con análisis",
                        lines=8,
                        interactive=False
                    )
                
                with gr.TabItem("Tabla de Análisis"):
                    gr.Markdown("### Datos detallados del análisis")
                    
                    # Botón para descargar CSV
                    download_csv_btn = gr.Button(
                        "Descargar Tabla como CSV",
                        variant="primary",
                        size="lg"
                    )
                    
                    # Archivo descargable
                    csv_download_file = gr.File(
                        label="Descargar CSV",
                        interactive=True,
                        visible=False
                    )
                    
                    sentiment_table = gr.DataFrame(
                        headers=["Participante", "Texto Original", "Texto Limpio", "Sentimiento", "Emoción", "Duración", "Confianza"],
                        interactive=False,
                        wrap=True
                    )
                
                with gr.TabItem("Gráficos"):
                    # Fila 1: Gráficos de duración y sentimientos por orador
                    with gr.Row(equal_height=True):
                        plot_duration = gr.Image(label="Duración por Orador", visible=False, height=400)
                        plot_sentiment_orador = gr.Image(label="Sentimientos por Orador", visible=False, height=400)
                        plot_emotion_orador = gr.Image(label="Emociones por Orador", visible=False, height=400)
                    
                    # Fila 2: Gráficos de torta (distribuciones totales)
                    with gr.Row(equal_height=True):
                        plot_sentiment_dist = gr.Image(label="Distribución Total de Sentimientos", visible=False, height=400)
                        plot_emotion_dist = gr.Image(label="Distribución Total de Emociones", visible=False, height=400)
                    
                    # Fila 3: Líneas de tiempo
                    with gr.Row(equal_height=True):
                        plot_timeline = gr.Image(label="Evolución del Sentimiento", visible=False, height=400)
                        plot_timeline_speakers = gr.Image(label="Sentimientos por Participante en el Tiempo", visible=False, height=400)
                    
                    # Fila 4: Análisis de texto
                    with gr.Row(equal_height=True):
                        plot_wordcloud = gr.Image(label="Nube de Palabras", visible=False, height=400)
                        plot_frequency = gr.Image(label="Palabras Más Frecuentes", visible=False, height=400)
                            
                with gr.TabItem("Reporte PDF"):
                    gr.Markdown("### Genere y descargue el reporte del análisis")
                    
                    generate_report_btn = gr.Button(
                        "Generar Reporte PDF",
                        variant="primary",
                        size="lg"
                    )
                    
                    report_download = gr.File(
                        label="Descargar Reporte",
                        interactive=True,
                        visible=False
                    )
            
            with gr.Row():
                back_to_page2_btn = gr.Button(
                    "⬅️ Volver a Identificación",
                    variant="secondary"
                )
                restart_btn = gr.Button(
                    "🔄 Nuevo Análisis",
                    variant="secondary"
                )
        
        # FUNCIONES DE NAVEGACIÓN Y PROCESAMIENTO
        
        def process_audio_with_progress(audio, speakers, language, remove_stopwords, min_tokens, progress=gr.Progress()):
            """Procesa el audio con barra de progreso visible usando el streaming function"""
            if audio is None:
                gr.Warning("Por favor, suba un archivo de audio")
                return [
                    gr.update(visible=False),  # progress_container
                    gr.update(value=0, label="❌ Error: No se subió audio", interactive=False),        
                    None, None, None           # states
                ]
            
            try:
                # Mostrar progress container
                yield [
                    gr.update(visible=True),   # progress_container visible
                    gr.update(value=5, label="🚀 Iniciando procesamiento...", interactive=False),
                    None, None, None
                ]
                
                # Usar la función de streaming mejorada para transcripción
                groupedDF = transcribe_with_whisperx_stream(audio, speakers, progress)
                
                yield [
                    gr.update(visible=True),
                    gr.update(value=80, label="🎤 Extrayendo muestras de voz...", interactive=False),
                    None, None, None
                ]
                
                # Paso 2: Extracción de muestras
                progress(0.8, desc="🎤 Extrayendo muestras de voz...")
                orators, speaker_files = extract_speaker_samples(current_audioDF, audio)
                
                # Paso 3: Preparación de resultados
                progress(0.9, desc="📝 Preparando resultados...")
                transcript_text = "\n".join([
                    f"{row['speaker']}: {row['text'][:100]}..."
                    for _, row in groupedDF.iterrows()
                    if row['text'].strip()
                ])
                
                speakers_text = f"Detectados {len(orators)} participantes: {', '.join(orators)}"
                
                progress(1.0, desc="✅ ¡Procesamiento completado!")
                
                # Resultado final
                yield [
                    gr.update(visible=False),  # hide progress
                    gr.update(value=100, label="✅ ¡Procesamiento completado!", interactive=False),
                    gr.update(value=transcript_text),
                    gr.update(value=speakers_text),
                    {"grouped_df": groupedDF, "speakers": orators, "speaker_files": speaker_files}
                ]
                
            except Exception as e:
                gr.Error(f"Error procesando audio: {str(e)}")
                yield [
                    gr.update(visible=False),
                    gr.update(value=0, label=f"❌ Error: {str(e)}", interactive=False),
                    gr.update(value=""),
                    gr.update(value=""),
                    None
                ]
        
        def go_to_page2(audio_data):
            """Navega a página 2 y configura speakers"""
            if audio_data is None:
                return [
                    gr.update(value=1),  # current_page
                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                    gr.update(value="<div>...step indicator...</div>"),
                    # Ocultar todos los speakers
                    gr.update(visible=False), gr.update(visible=False),  # speaker 1
                    gr.update(visible=False), gr.update(visible=False),  # speaker 2
                    gr.update(visible=False), gr.update(visible=False),  # speaker 3
                    gr.update(visible=False), gr.update(visible=False)   # speaker 4
                ]
            
            speakers = audio_data.get("speakers", [])
            speaker_files = audio_data.get("speaker_files", {})
            
            # Configurar updates para cada speaker (máximo 4)
            speaker_updates = []
            for i in range(4):
                if i < len(speakers):
                    speaker = speakers[i]
                    audio_file = speaker_files.get(speaker, None)
                    speaker_updates.extend([
                        gr.update(visible=True, value=audio_file, label=f"Muestra {speaker}"),  # audio
                        gr.update(visible=True, label=f"Nombre para {speaker}", interactive=True)  # textbox
                    ])
                else:
                    speaker_updates.extend([
                        gr.update(visible=False),  # audio oculto
                        gr.update(visible=False, interactive=True) # textbox oculto
                    ])
            
            return [
                gr.update(value=2),  # current_page
                gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),  # páginas
                gr.update(value="""
                <div class="main-title"> Analizador de Sentimientos y Emociones en Audio</div>
                <div class="step-indicator">
                    <div class="step completed">1. Carga de Audio</div>
                    <div class="step active">2. Transcripción e Identificación</div>
                    <div class="step pending">3. Resultados</div>
                </div>
                """)
            ] + speaker_updates
        
        def perform_complete_analysis(audio_data, name1, name2, name3, name4, progress=gr.Progress()):
            """Realiza el análisis completo con barra de progreso visible"""
            global current_audioDF
            
            if audio_data is None:
                gr.Warning("No hay datos de audio para analizar")
                return [
                    gr.update(visible=False), # progress2_container
                    gr.update(value=0, label="❌ Error: No hay datos", interactive=False), 
                    None
                ]
            
            try:
                # Mostrar progress container
                yield [
                    gr.update(visible=True),   # progress2_container visible
                    gr.update(value=5, label="🚀 Iniciando análisis completo...", interactive=False),
                    None
                ]
                
                # Obtener datos reales
                groupedDF = audio_data["grouped_df"]
                speakers = audio_data["speakers"]
                
                # Crear diccionario de nombres
                names = [name1, name2, name3, name4]
                orator_names = {}
                for i, speaker in enumerate(speakers):
                    if i < len(names) and names[i] and names[i].strip():
                        orator_names[speaker] = names[i].strip()
                
                progress(0.05, desc="👥 Aplicando nombres de participantes...")
                yield [
                    gr.update(visible=True),
                    gr.update(value=10, label="👥 Aplicando nombres de participantes...", interactive=False),
                    None
                ]
                
                # Usar funciones reales con progress
                updated_audioDF = update_speaker_names(current_audioDF, orator_names)
                groupedDF = group_consecutive_speakers(updated_audioDF)
                
                progress(0.1, desc="🔍 Aplicando filtros de texto...")
                yield [
                    gr.update(visible=True),
                    gr.update(value=15, label="🔍 Aplicando filtros de texto...", interactive=False),
                    None
                ]
                filtered_audioDF = apply_text_filters(groupedDF.copy(), remove_stopwords=True, min_tokens=4)
                
                progress(0.15, desc="🧠 Iniciando análisis de sentimientos...")
                yield [
                    gr.update(visible=True),
                    gr.update(value=20, label="🧠 Analizando sentimientos y emociones...", interactive=False),
                    None
                ]
                filtered_audioDF = analyze_sentiments(filtered_audioDF, "es", progress)
                
                yield [
                    gr.update(visible=True),
                    gr.update(value=50, label="🌐 Análisis complementario en inglés...", interactive=False),
                    None
                ]
                filtered_audioDF = analyze_complementary_english(filtered_audioDF, progress)
                
                yield [
                    gr.update(visible=True),
                    gr.update(value=70, label="🤖 Análisis avanzado con Roberta...", interactive=False),
                    None
                ]
                filtered_audioDF = apply_roberta_analysis_and_replace(filtered_audioDF, progress)
                
                progress(0.8, desc="📊 Generando visualizaciones...")
                yield [
                    gr.update(visible=True),
                    gr.update(value=85, label="📊 Generando gráficos y visualizaciones...", interactive=False),
                    None
                ]
                plot_files = generate_analysis_plots(filtered_audioDF, progress)
                
                progress(1.0, desc="✅ ¡Análisis completo finalizado!")
                
                # Preparar datos finales
                analysis_data = {
                    "filtered_audioDF": filtered_audioDF,
                    "plot_files": plot_files,
                    "transcript": "\n".join([f"{row['speaker']}: {row['text']}" for _, row in filtered_audioDF.iterrows()]),
                    "sentiment_data": []
                }
                
                # Preparar tabla de sentimientos
                for _, row in filtered_audioDF.iterrows():
                    if row['text_clean'].strip():
                        # Texto original (con stopwords)
                        texto_original = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
                        
                        # Texto limpio (sin stopwords) 
                        texto_limpio = row['text_clean'][:80] + "..." if len(row['text_clean']) > 80 else row['text_clean']
                        
                        # Duración del segmento
                        duracion = f"{row['duration']:.1f}s"
                        
                        # Confianza (placeholder basado en scores si existen)
                        confianza = "85%"  # placeholder por defecto
                        if 'sentimiento_score' in row and row['sentimiento_score']:
                            try:
                                max_score = max(row['sentimiento_score'].values())
                                confianza = f"{max_score*100:.0f}%"
                            except:
                                confianza = "85%"
                        
                        analysis_data["sentiment_data"].append([
                            row['speaker'],
                            texto_original,
                            texto_limpio, 
                            row['sentimiento'],
                            row['emocion'],
                            duracion,
                            confianza
                        ])
                
                # Resultado final
                yield [
                    gr.update(visible=False),  # hide progress
                    gr.update(value=100, label="✅ ¡Análisis completado exitosamente!", interactive=False),
                    analysis_data
                ]
                
            except Exception as e:
                gr.Error(f"Error en análisis: {str(e)}")
                yield [
                    gr.update(visible=False),
                    gr.update(value=0, label=f"❌ Error: {str(e)}", interactive=False),
                    None
                ]
        
        def go_to_page3(analysis_data):
            """Navega a página 3 y muestra resultados"""
            if analysis_data is None:
                gr.Warning("Primero debe completar el análisis")
                return [
                    gr.update(value=2), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                    ]
            
            # Mapear archivos PNG específicos a componentes específicos
            plot_files = analysis_data.get("plot_files", [])
            
            # Crear diccionario de archivos por nombre
            plot_dict = {}
            for plot_file in plot_files:
                if os.path.exists(plot_file):
                    filename = os.path.basename(plot_file)
                    plot_dict[filename] = plot_file
            
            # Mapear cada componente a su archivo correspondiente
            plot_updates = [
                # plot_duration
                gr.update(value=plot_dict.get("duracion_orador.png"), visible="duracion_orador.png" in plot_dict),
                # plot_sentiment_orador  
                gr.update(value=plot_dict.get("sentimiento_orador.png"), visible="sentimiento_orador.png" in plot_dict),
                # plot_emotion_orador
                gr.update(value=plot_dict.get("emocion_orador.png"), visible="emocion_orador.png" in plot_dict),
                # plot_sentiment_dist
                gr.update(value=plot_dict.get("sentimiento_total.png"), visible="sentimiento_total.png" in plot_dict),
                # plot_emotion_dist
                gr.update(value=plot_dict.get("emocion_total.png"), visible="emocion_total.png" in plot_dict),
                # plot_timeline
                gr.update(value=plot_dict.get("sentimiento_tiempo.png"), visible="sentimiento_tiempo.png" in plot_dict),
                # plot_timeline_speakers
                gr.update(value=plot_dict.get("sentimiento_tiempo_speakers.png"), visible="sentimiento_tiempo_speakers.png" in plot_dict),
                # plot_wordcloud
                gr.update(value=plot_dict.get("wordcloud.png"), visible="wordcloud.png" in plot_dict),
                # plot_frequency
                gr.update(value=plot_dict.get("palabras_frecuentes.png"), visible="palabras_frecuentes.png" in plot_dict)
            ]
            
            return [
                gr.update(value=3),  # current_page
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),  # páginas
                gr.update(value="""
                    <div class="main-title"> Analizador de Sentimientos y Emociones en Audio</div>
                    <div class="step-indicator">
                    <div class="step completed">1. Carga de Audio</div>
                    <div class="step completed">2. Transcripción e Identificación</div>
                    <div class="step active">3. Resultados</div>
                </div>"""),
                gr.update(value=analysis_data.get("transcript", "")),  # final_transcript
                gr.update(value=analysis_data.get("sentiment_data", [])),  # sentiment_table
                gr.update(visible=False)  # csv_download_file (inicialmente oculto)                
            ] + plot_updates
        
        def restart_analysis():
            """Reinicia toda la aplicación"""
            return [
                gr.update(value=1),  # current_page
                gr.update(visible=True),   # page1
                gr.update(visible=False),  # page2
                gr.update(visible=False),  # page3
                gr.update(value="""
                <div class="main-title"> Analizador de Sentimientos y Emociones en Audio</div>
                <div class="step-indicator">
                    <div class="step active">1. Carga de Audio</div>
                    <div class="step pending">2. Transcripción e Identificación</div>
                    <div class="step pending">3. Resultados</div>
                </div>
                """),
                gr.update(value=None),  # audio_input
                None, None, None  # clear all states
            ]
        
        # EVENTOS DE LA APLICACIÓN
        
        # Procesar audio con progress bar visible
        process_btn.click(
            fn=process_audio_with_progress,
            inputs=[audio_input, speakers_input, language_input, remove_stopwords_input, min_tokens_input],
            outputs=[
                progress_container,
                progress_bar,
                transcript_preview,
                speakers_detected,
                audio_data_state
            ]
        ).then(
            fn=go_to_page2,
            inputs=[audio_data_state],
            outputs=[
                current_page, page1, page2, page3, step_indicator,
                speaker_audio_1, speaker_name_1,
                speaker_audio_2, speaker_name_2, 
                speaker_audio_3, speaker_name_3,
                speaker_audio_4, speaker_name_4
            ]
        )
        
        # Realizar análisis completo con progress bar
        analyze_btn.click(
            fn=perform_complete_analysis,
            inputs=[audio_data_state, speaker_name_1, speaker_name_2, speaker_name_3, speaker_name_4],
            outputs=[progress2_container, progress2_bar, final_analysis_state]
        ).then(
            fn=go_to_page3,
            inputs=[final_analysis_state],
            outputs=[
                current_page, page1, page2, page3, step_indicator,
                final_transcript, sentiment_table, csv_download_file,  
                plot_duration, plot_sentiment_orador, plot_emotion_orador,
                plot_sentiment_dist, plot_emotion_dist,
                plot_timeline, plot_timeline_speakers,
                plot_wordcloud, plot_frequency
            ]
        )
        
        # Botones de navegación
        back_to_page1_btn.click(
            fn=lambda: [
                gr.update(value=1),
                gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                gr.update(value="""
                <div class="main-title"> Analizador de Sentimientos y Emociones en Audio</div>
                <div class="step-indicator">
                    <div class="step active">1. Carga de Audio</div>
                    <div class="step pending">2. Transcripción e Identificación</div>
                    <div class="step pending">3. Resultados</div>
                </div>
                """)
            ],
            outputs=[current_page, page1, page2, page3, step_indicator]
        )
        
        back_to_page2_btn.click(
            fn=lambda: [
                gr.update(value=2),
                gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                gr.update(value="""
                <div class="main-title"> Analizador de Sentimientos y Emociones en Audio</div>
                <div class="step-indicator">
                    <div class="step completed">1. Carga de Audio</div>
                    <div class="step active">2. Transcripción e Identificación</div>
                    <div class="step pending">3. Resultados</div>
                </div>
                """)
            ],
            outputs=[current_page, page1, page2, page3, step_indicator]
        )
        
        restart_btn.click(
            fn=restart_analysis,
            outputs=[
                current_page, page1, page2, page3, step_indicator,
                audio_input,
                audio_data_state, speakers_data_state, final_analysis_state
            ]
        )
        
        # Generar reporte PDF
        generate_report_btn.click(
            fn=generar_reporte,
            inputs=[final_analysis_state],
            outputs=[report_download]
        )

        download_csv_btn.click(
            fn=generate_csv_from_analysis,
            inputs=[final_analysis_state],
            outputs=[csv_download_file]
        ).then(
            fn=lambda file_path: gr.update(visible=True) if file_path else gr.update(visible=False),
            inputs=[csv_download_file],
            outputs=[csv_download_file]
        )
    
    return demo

# Lanzar la aplicación
if __name__ == "__main__":
    demo = create_audio_analyzer_app()
    demo.queue()
    demo.launch()