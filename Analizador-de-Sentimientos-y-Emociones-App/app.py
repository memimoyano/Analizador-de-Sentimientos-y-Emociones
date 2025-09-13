# Procesamiento y transcripción de audio
import torch
import whisperx
import pydub
from pydub import AudioSegment
import librosa

# NLP y análisis
import pandas as pd
import re
import string
from collections import Counter
from pysentimiento import create_analyzer
from transformers import pipeline
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

# Web app
import gradio as gr
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
align_model, align_metadata = None, None
labels = ["Positive", "Negative", "Neutral"]

current_model_config = {
    "size": "tiny",
    "device": device,
    "compute_type": compute_type
}

load_dotenv("secrets.env")
HF_TOKEN = os.getenv("HF_TOKEN")

try:
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
except Exception as e:
    classifier = None

try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("Error descargando recursos NLTK, pero continuando...")
    
stop_words = set(stopwords.words('spanish'))
punctuation_to_strip = set(string.punctuation) | {"«", "»", "…"}
punctuation_to_strip -= {"¡", "¿", "!", "?"}

current_audioDF = None
current_audio_path = None

lenguaje = "es"

def reload_whisper_model(model_size, progress=gr.Progress()):
    """
    Recarga el modelo Whisper con el tamaño especificado
    """
    global whisper_model, current_model_config
    
    try:
        progress(0, desc=f"Cargando modelo {model_size}...")
        
        if whisper_model is not None:
            del whisper_model
            whisper_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        progress(0.3, desc=f"Descargando modelo {model_size}...")
        
        if device == "cuda":
            comp_type = "float16"
        else:
            comp_type = "int8"

        whisper_model = whisperx.load_model(
            model_size, 
            device=device, 
            compute_type=comp_type
        )
        
        current_model_config.update({
            "size": model_size,
            "device": device,
            "compute_type": comp_type
        })
        
        progress(1.0, desc=f"✅ Modelo {model_size} cargado correctamente")
        
        return [
            gr.update(value=f"✅ Modelo {model_size} cargado correctamente"),
            gr.update(visible=False)
        ]
        
    except Exception as e:
        progress(0, desc="❌ Error cargando modelo")
        
        try:
            whisper_model = whisperx.load_model("tiny", device=device, compute_type=comp_type)
            current_model_config.update({
                "size": "tiny",
                "device": device,
                "compute_type": comp_type
            })
            return [
                gr.update(value=f"❌ Error cargando {model_size}. Restaurado a 'tiny': {str(e)}"),
                gr.update(visible=False)
            ]
        except Exception as fallback_error:
            whisper_model = None
            return [
                gr.update(value=f"❌ Error crítico: {str(e)}"),
                gr.update(visible=False)
            ]

def get_model_info():
    """Retorna información del modelo actual"""
    global current_model_config, whisper_model
    
    if whisper_model is None:
        return "❌ No hay modelo cargado"
    
    return (f"Modelo: {current_model_config['size']} | "
            f"Dispositivo: {current_model_config['device']} | "
            f"Tipo: {current_model_config['compute_type']}")

def change_model_with_progress(model_size, progress=gr.Progress()):
    """Cambia el modelo con barra de progreso visible"""
    
    try:
        yield [
            gr.update(visible=True), 
            gr.update(value=10, label=f"Iniciando cambio a {model_size}...", interactive=False),
            gr.update(value="Cambiando modelo..."), 
            gr.update(value="Cambiando modelo...") 
        ]
        
        status_updates = reload_whisper_model(model_size, progress)

        updated_info = get_model_info()

        yield [
            gr.update(visible=False),
            gr.update(value=100, label=f"✅ Cambio completado", interactive=False),
            status_updates[0],
            gr.update(value=updated_info)
        ]
        
    except Exception as e:
        yield [
            gr.update(visible=False),
            gr.update(value=0, label="❌ Error", interactive=False),
            gr.update(value=f"❌ Error: {str(e)}"),
            gr.update(value=get_model_info())
        ]

def restart_analysis():
    """Reinicia toda la aplicación"""
    return [
        gr.update(value=1),
        gr.update(visible=True), 
        gr.update(visible=False),
        gr.update(visible=False), 
        gr.update(value="""
        <div class="main-title"> Analizador de Sentimientos y Emociones en Audio en Español</div>
        <div class="step-indicator">
            <div class="step active">1. Carga de Audio</div>
            <div class="step pending">2. Transcripción e Identificación</div>
            <div class="step pending">3. Resultados</div>
        </div>
        """),
        gr.update(value=None),
        gr.update(value=get_model_info()),
        gr.update(value=get_model_info()),
        None, None, None
    ]

def transcribe_with_whisperx_stream(audio_path, num_speakers=None, progress=gr.Progress()):
    """
    Transcripción con progress bar usando Gradio Progress
    """
    global align_model, align_metadata, current_audioDF, current_audio_path
    current_audio_path = audio_path
    progress(0, desc="Iniciando transcripción...")
    
    try:
        progress(0.05, desc="Cargando archivo de audio...")
        audio = whisperx.load_audio(audio_path)
        duration = librosa.get_duration(path=audio_path)
        chunk_size = 30
        num_chunks = int(duration // chunk_size) + 1
        partial_segments = []
        transcript_so_far = ""
        
        progress(0.1, desc="Transcribiendo audio...")
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, duration)
            
            audio_chunk = audio[int(start * 16000):int(end * 16000)]
            result = whisper_model.transcribe(audio_chunk, batch_size=8)
            
            for seg in result["segments"]:
                seg["start"] += start
                seg["end"] += start
                partial_segments.append(seg)
                transcript_so_far += f"{seg['text'].strip()} "
                
            chunk_progress = 0.1 + (0.5 * (i + 1) / num_chunks)
            progress(chunk_progress, desc=f"🎤 Transcribiendo segmento {i+1}/{num_chunks}...")
            
        progress(0.65, desc="Alineando transcripción...")
        
        if align_model is None or result["language"] != getattr(align_metadata, "language_code", None):
            align_model, align_metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result_aligned = whisperx.align(partial_segments, align_model, align_metadata, audio, device)
        
        progress(0.8, desc="Identificando hablantes...")
        
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
        if num_speakers is not None and str(num_speakers).strip() != "":
            diarize_segments = diarize_model(audio, num_speakers=int(num_speakers))
        else:
            diarize_segments = diarize_model(audio)
        result_final = whisperx.assign_word_speakers(diarize_segments, result_aligned)
        
        progress(0.9, desc="Guardando transcripción...")
        
        with open("transcription.txt", "w", encoding="utf-8") as file:
            for seg in result_final["segments"]:
                if "speaker" in seg:
                    speaker = seg["speaker"]
                    text = seg["text"].strip()
                    if text:
                        file.write(f"{speaker}: {text}\n")
        
        progress(0.95, desc="Procesando resultados...")
        
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
    Agrupa segmentos consecutivos de oradores (speakers) en oraciones únicas, ajustando el tiempo de "fin" y las palabras.
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
    audio = AudioSegment.from_file(audio_path)
    
    first_appearances = {}
    for idx, row in audioDF.iterrows():
        speaker = row['speaker']
        if speaker not in first_appearances:
            first_appearances[speaker] = row['start']
    
    speakers_ordered = sorted(first_appearances.keys(), key=lambda x: first_appearances[x])
    
    temp_list = [
        (idx, row['speaker'], row['start'], row['end'], row['end'] - row['start'])
        for idx, row in audioDF.iterrows()
    ]
    
    speaker_audio_files = {}
    orators = []
    
    for speaker in speakers_ordered:          
        found_good_segment = False
        for idx, spk, start, end, duration in temp_list:
            if spk == speaker and duration > min_duration:
                start_ms = int(start * 1000)
                end_ms = int(end * 1000)
                segment = audio[start_ms:end_ms]
                
                temp_file = f"temp_segment_{speaker}.wav"
                segment.export(temp_file, format="wav")
                speaker_audio_files[speaker] = temp_file
                orators.append(speaker)
                found_good_segment = True
                break
        
        if not found_good_segment:
            speaker_segments = [(idx, spk, start, end, duration) 
                              for idx, spk, start, end, duration in temp_list 
                              if spk == speaker]
            
            if speaker_segments:
                longest_segment = max(speaker_segments, key=lambda x: x[4])
                idx, spk, start, end, duration = longest_segment
                
                start_ms = int(start * 1000)
                end_ms = int(end * 1000)
                segment = audio[start_ms:end_ms]
                
                temp_file = f"temp_segment_{speaker}.wav"
                segment.export(temp_file, format="wav")
                speaker_audio_files[speaker] = temp_file
                orators.append(speaker)

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

def remove_stopwords_language(text, language="spanish"):
    """
    Remueve stopwords y puntuación innecesaria de un texto dado un idioma.
    Soporta: 'spanish' y 'english'
    """
    stop_words = set(stopwords.words(language))
    
    tokens = word_tokenize(text.lower(), language=language)
    cleaned_tokens = []
    
    for token in tokens:
        word_clean = token.strip("".join(punctuation_to_strip))
        if word_clean and word_clean not in stop_words:
            cleaned_tokens.append(word_clean)
    
    texto = ' '.join(cleaned_tokens)
    texto = re.sub(r'\s+([?!¡¿])', r'\1', texto)
    
    return texto

def apply_text_filters(audioDF, remove_stopwords=False, min_tokens=4):
    """
    Aplica filtros de texto al DataFrame
    """
    audioDF['duration'] = audioDF['end'] - audioDF['start']
    
    if remove_stopwords:
        audioDF["text_clean"] = audioDF["text"].apply(remove_stopwords_language)
        audioDF = audioDF[audioDF['text_clean'].str.strip().astype(bool)]
        audioDF = audioDF[audioDF['text_clean'].apply(lambda x: len(str(x).split()) >= min_tokens)]
    else:
        audioDF["text_clean"] = audioDF["text"]
        audioDF = audioDF[audioDF['text'].apply(lambda x: len(str(x).split()) >= min_tokens)]
    
    return audioDF

def analyze_sentiments(audioDF, language="es", progress=None):
    """
    Analiza sentimientos y emociones del DataFrame usando pysentimiento
    """
    global lenguaje
    lenguaje = language
    
    if progress:
        progress(0, desc="Iniciando análisis de sentimientos...")
    
    try:
        sentiment_analyzer = create_analyzer(task="sentiment", lang=lenguaje)
        emotion_analyzer = create_analyzer(task="emotion", lang=lenguaje)
        
        total_rows = len(audioDF)
        
        if progress:
            progress(0.1, desc="Analizando sentimientos...")
        
        sentiment_results = []
        for i, text in enumerate(audioDF['text_clean']):
            sentiment_results.append(sentiment_analyzer.predict(text).output)
            if i % 10 == 0 and progress: 
                progress(0.1 + 0.4 * (i / total_rows), desc=f"Analizando sentimientos ({i+1}/{total_rows})...")
        
        audioDF['sentimiento'] = sentiment_results
        
        if progress:
            progress(0.5, desc="Analizando emociones...")
        
        emotion_results = []
        emotion_scores = []
        for i, text in enumerate(audioDF['text_clean']):
            result = emotion_analyzer.predict(text)
            emotion_results.append(result.output)
            emotion_scores.append(result.probas)
            if i % 10 == 0 and progress:
                progress(0.5 + 0.4 * (i / total_rows), desc=f"Analizando emociones ({i+1}/{total_rows})...")
        
        audioDF['emocion'] = emotion_results
        audioDF['emocion_score'] = emotion_scores
        
        if progress:
            progress(0.9, desc="Finalizando análisis...")
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
        return audioDF

def traducir(texto):
    """
    Función para traducir texto 
    """
    try:
        return GoogleTranslator(source='auto', target='en').translate(texto)
    except Exception as e:
        return None

def analyze_complementary_english(audioDF, progress=None):
    """
    Análisis complementario en inglés para frases con emoción 'others' o sentimiento 'NEU'
    """
    if progress:
        progress(0, desc="Iniciando análisis complementario en inglés...")
    
    try:
        df_work = audioDF.copy()
        
        mask = (df_work['emocion'] == 'others') | (df_work['sentimiento'] == 'NEU')
        
        if not mask.any():
            if progress:
                progress(1.0, desc="✅ No se requiere análisis complementario")
            return audioDF
        
        indices_to_analyze = df_work[mask].index.tolist()
        total_items = len(indices_to_analyze)
        
        if progress:
            progress(0.1, desc=f"Procesando {total_items} textos...")
        
        for col in ['text_translated', 'text_clean_en', 'sentimiento_en', 'emocion_en', 'sentimiento_score_en', 'emocion_score_en']:
            if col not in df_work.columns:
                df_work[col] = None
        
        sentiment_analyzer_en = create_analyzer(task="sentiment", lang="en")
        emotion_analyzer_en = create_analyzer(task="emotion", lang="en")
        
        for count, idx in enumerate(indices_to_analyze):
            try:
                if idx not in df_work.index:
                    continue
                
                original_text = df_work.at[idx, 'text']
                
                translated_text = traducir(original_text)
                if translated_text is None or translated_text.strip() == '':
                    continue
                
                df_work.at[idx, 'text_translated'] = translated_text
                
                clean_text_en = remove_stopwords_language(translated_text, "english")
                
                if len(clean_text_en.split()) < 4:
                    continue
                
                df_work.at[idx, 'text_clean_en'] = clean_text_en
                
                sent_result = sentiment_analyzer_en.predict(clean_text_en)
                emo_result = emotion_analyzer_en.predict(clean_text_en)
                
                df_work.at[idx, 'sentimiento_en'] = sent_result.output
                df_work.at[idx, 'emocion_en'] = emo_result.output
                df_work.at[idx, 'sentimiento_score_en'] = sent_result.probas
                df_work.at[idx, 'emocion_score_en'] = emo_result.probas
                
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
        traceback.print_exc()
        return audioDF

def analisis_baseGoEmotions_samlowe(text):
    """
    Análisis complementario con el modelo de GoEmotions de SamLowe
    """
    try:
        if classifier is None:
            return 'neutral'
        
        model_outputs = classifier(text)

        if isinstance(model_outputs, list) and len(model_outputs) > 0:
            if isinstance(model_outputs[0], dict) and 'label' in model_outputs[0]:
                return model_outputs[0]['label']
            elif isinstance(model_outputs[0], list) and len(model_outputs[0]) > 0:
                if isinstance(model_outputs[0][0], dict) and 'label' in model_outputs[0][0]:
                    return model_outputs[0][0]['label']
        
        return 'neutral'
        
    except Exception as e:
        return 'neutral'

emotion_mapping = {
    'anger': 'anger',
    'annoyance': 'anger',
    'disapproval': 'anger',

    'disgust': 'disgust',

    'fear': 'fear',
    'nervousness': 'fear',

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

    'sadness': 'sadness',
    'disappointment': 'sadness',
    'embarrassment': 'sadness',
    'grief': 'sadness',
    'remorse': 'sadness',

    'confusion': 'surprise',
    'curiosity': 'surprise',
    'realization': 'surprise',
    'surprise': 'surprise',

    'neutral': 'neutral'
}

def apply_roberta_analysis_and_replace(audioDF, progress=None):
    """
    Aplica análisis con roberta-base-go_emotions y reemplaza valores NEU y others
    """
    if progress:
        progress(0, desc="🤖 Iniciando análisis Roberta...")
    
    try:
        df_work = audioDF.copy()
        
        if 'text_translated' not in df_work.columns:
            if progress:
                progress(0.5, desc="Sin textos traducidos, aplicando cambios básicos...")
            df_work.loc[df_work["emocion"] == "others", "emocion"] = "neutral"
            if progress:
                progress(1.0, desc="✅ Cambios básicos aplicados")
            return df_work
        
        mask_translated = df_work['text_translated'].notna()
        
        if not mask_translated.any():
            if progress:
                progress(0.5, desc="⚠️ Sin textos para análisis Roberta...")
            df_work.loc[df_work["emocion"] == "others", "emocion"] = "neutral"
            if progress:
                progress(1.0, desc="✅ Cambios básicos aplicados")
            return df_work
        
        indices_translated = df_work[mask_translated].index.tolist()
        total_items = len(indices_translated)
        
        if progress:
            progress(0.1, desc=f"Procesando {total_items} textos con Roberta...")
                
        if 'emotion_roberta' not in df_work.columns:
            df_work['emotion_roberta'] = None
        if 'sentiment_en_pysentimiento' not in df_work.columns:
            df_work['sentiment_en_pysentimiento'] = None
        
        sentiment_analyzer_en = create_analyzer(task="sentiment", lang="en")
        
        for count, idx in enumerate(indices_translated):
            try:
                translated_text = df_work.at[idx, 'text_translated']
                clean_text_en = df_work.at[idx, 'text_clean_en'] if 'text_clean_en' in df_work.columns else None
                
                if not translated_text or pd.isna(translated_text):
                    continue
                
                emotion_roberta = analisis_baseGoEmotions_samlowe(translated_text)
                emotion_mapped = emotion_mapping.get(emotion_roberta, 'neutral')
                df_work.at[idx, 'emotion_roberta'] = emotion_mapped
                
                if clean_text_en and len(str(clean_text_en).strip()) > 0:
                    sentiment_en = sentiment_analyzer_en.predict(clean_text_en).output
                    df_work.at[idx, 'sentiment_en_pysentimiento'] = sentiment_en
                else:
                    sentiment_en = sentiment_analyzer_en.predict(translated_text).output
                    df_work.at[idx, 'sentiment_en_pysentimiento'] = sentiment_en
                
                if progress:
                    progress_val = 0.1 + 0.7 * (count + 1) / total_items
                    progress(progress_val, desc=f"Procesando con Roberta {count+1}/{total_items}...")
                
            except Exception as e:
                continue
        
        if progress:
            progress(0.9, desc="Aplicando reemplazos...")
                
        if 'sentiment_en_pysentimiento' in df_work.columns:
            mask_replace_sentiment = (
                (df_work["sentimiento"] == "NEU") & 
                df_work["sentiment_en_pysentimiento"].notna()
            )
            df_work.loc[mask_replace_sentiment, "sentimiento"] = df_work.loc[mask_replace_sentiment, "sentiment_en_pysentimiento"]
        
        if 'emotion_roberta' in df_work.columns:
            mask_replace_emotion = (
                (df_work["emocion"] == "others") & 
                df_work["emotion_roberta"].notna()
            )
            df_work.loc[mask_replace_emotion, "emocion"] = df_work.loc[mask_replace_emotion, "emotion_roberta"]
        
        df_work.loc[df_work["emocion"] == "others", "emocion"] = "neutral"
        
        if progress:
            progress(1.0, desc="✅ Análisis Roberta completado")
        
        return df_work
        
    except Exception as e:
        if progress:
            progress(0, desc=f"❌ Error en análisis Roberta: {str(e)}")
        traceback.print_exc()
        audioDF.loc[audioDF["emocion"] == "others", "emocion"] = "neutral"
        return audioDF

def generate_analysis_plots(audioDF, progress=None):
    """
    Genera todos los gráficos de análisis
    """
    if progress:
        progress(0, desc="Iniciando generación de gráficos...")
    
    try:
        
        df_plot = audioDF.copy()
        df_plot['emocion'] = df_plot['emocion'].replace({
            'joy': 'alegría',
            'surprise': 'sorpresa',
            'disgust': 'disgusto',
            'sadness': 'tristeza',
            'fear': 'miedo',
            'anger': 'enojo'
        })
        
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
        
        if progress:
            progress(0.05, desc="Creando gráfico de duración por orador...")
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
        
        if progress:
            progress(0.15, desc="Creando gráfico de sentimientos por orador...")
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
        
        if progress:
            progress(0.25, desc="Creando gráfico de emociones por orador...")
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
        
        if progress:
            progress(0.35, desc="Creando gráfico circular de sentimientos...")
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
        
        if progress:
            progress(0.45, desc="Creando gráfico circular de emociones...")
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
        
        if progress:
            progress(0.55, desc="Creando línea temporal de sentimientos por speaker...")
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
        
        if progress:
            progress(0.65, desc="Creando línea temporal general de sentimientos...")
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
        
        if progress:
            progress(0.75, desc="Creando nube de palabras...")
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
        
        if progress:
            progress(0.85, desc="Creando gráfico de palabras frecuentes...")
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
        return []

def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def generar_reporte_pdf_reportlab(audioDF, plot_files, output_path="Informe_AS-EC.pdf"):
    """
    Genera el reporte final con reportlab
    """
    try:
        
        doc = SimpleDocTemplate(output_path, pagesize=A4, 
                              leftMargin=0.75*inch, rightMargin=0.75*inch,
                              topMargin=1*inch, bottomMargin=1*inch)
        styles = getSampleStyleSheet()
        
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
        
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph("INFORME DE ANÁLISIS", title_style))
        story.append(Paragraph("Sentimientos y Emociones en Audio en Español", title_style))
        story.append(Spacer(1, 1*inch))
        
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
        
        story.append(Paragraph("RESUMEN EJECUTIVO", subtitle_style))
        
        total_duracion = audioDF['duration'].sum()
        participantes = audioDF['speaker'].unique()
        sentiment_dist = audioDF.groupby('sentimiento')['duration'].sum()
        emotion_dist = audioDF.groupby('emocion')['duration'].sum()
        
        speaker_time = audioDF.groupby('speaker')['duration'].sum()
        most_active = speaker_time.idxmax()
        most_active_pct = (speaker_time.max() / total_duracion) * 100
        
        main_sentiment = sentiment_dist.idxmax()
        main_sentiment_pct = (sentiment_dist.max() / total_duracion) * 100
        
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
        
        story.append(Paragraph("ANÁLISIS POR PARTICIPANTE TOTAL", subtitle_style))
        
        participant_summary = audioDF.groupby('speaker').agg({
            'duration': ['sum', 'count'],
            'sentimiento': lambda x: x.value_counts().index[0],  
            'emocion': lambda x: x.value_counts().index[0]       
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
        
        participant_table = Table(participant_data, colWidths=[1.5*inch, 1*inch, 1*inch, 2*inch, 2*inch])
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
        
        story.append(Paragraph("ANÁLISIS POR INTERVENCIÓN", subtitle_style))

        intervencion_data = [["# Intervención", "Participante", "Duración (s)", "Sentimiento", "Emoción"]]

        for num_intervencion, (_, row) in enumerate(audioDF.iterrows(), start=1):
            intervencion_data.append([
                str(num_intervencion),
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

        story.append(Paragraph("ANÁLISIS VISUAL", subtitle_style))
        
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
        
        available_plots = {}
        for plot_file in plot_files:
            if os.path.exists(plot_file):
                filename = os.path.basename(plot_file)
                available_plots[filename] = plot_file
        
        for categoria, archivos in graficos_organizados:
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
                        story.append(Paragraph(f"[Error cargando gráfico: {title}]", styles['Normal']))
                        story.append(Spacer(1, 10))
        
        doc.build(story)
        return output_path
        
    except Exception as e:
        return None 

def generar_reporte(analysis_data):
    """
    Genera el reporte PDF con la información que se le pasa como parámetro
    """
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
        
        file_path = os.path.join(tempfile.gettempdir(), "Tabla_analisis_AS-EC.csv")
        
        with open(file_path, mode='w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
            
            headers = ["Participante", "Texto_Original", "Texto_Limpio", "Sentimiento", "Emocion", "Duracion", "Confianza"]
            csv_writer.writerow(headers)
            
            for row in sentiment_data:
                csv_writer.writerow(row)
        
        return file_path
        
    except Exception as e:
        return None, gr.update(value=f"❌ Error generando CSV: {str(e)}")

def create_audio_analyzer_app():
    with gr.Blocks(theme=gr.themes.Soft(), title="Analizador de Audio Multi-Página") as demo:
        gr.HTML("""
        <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
                
                /* Fuentes más legibles para textboxes y inputs */
                .gr-textbox input, 
                .gr-textbox textarea,
                input[type="text"],
                input[type="number"],
                textarea,
                .gr-form input,
                .gr-form textarea {
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
                    font-size: 14px !important;
                    font-weight: 400 !important;
                    line-height: 1.5 !important;
                    color: #ffffff !important;
                }
                
                /* Labels más legibles */
                label, .gr-form label {
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
                    font-size: 14px !important;
                    font-weight: 500 !important;
                    color: #2d3748 !important;
                }
                
                /* Dropdowns */
                .gr-dropdown .wrap,
                .gr-dropdown .wrap-inner,
                select {
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
                    font-size: 14px !important;
                }
                
                /* DataFrames y tablas */
                .gr-dataframe table,
                .gr-dataframe th,
                .gr-dataframe td {
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
                    font-size: 13px !important;
                }
                
                /* Código y texto monoespaciado */
                .gr-code, 
                pre, 
                code {
                    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
                    font-size: 13px !important;
                }
                
                /* Botones */
                button, .gr-button {
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
                    font-weight: 500 !important;
                }
                
                /* Mejorar contraste en textboxes */
                .gr-textbox input:focus,
                .gr-textbox textarea:focus {
                    border-color: #4299e1 !important;
                    box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.1) !important;
                }
                
                /* Placeholder text más legible */
                .gr-textbox input::placeholder,
                .gr-textbox textarea::placeholder {
                    color: #a0aec0 !important;
                    font-style: italic;
                }
                
                /* Textos de información/ayuda */
                .gr-form .gr-form-gap .help-text,
                .gr-textbox .help-text,
                small {
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
                    font-size: 12px !important;
                    color: #718096 !important;
                }
                
                /* Títulos existentes mantienen su estilo */
                .main-title {
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
                    font-size: 24px;
                    font-weight: 700;
                    text-transform: uppercase;
                    color: white;
                    text-align: center;
                    margin-bottom: 10px;
                }
                .subtitle {
                    font-family: 'Inter', sans-serif;
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
                    font-family: 'Inter', sans-serif !important;
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
                    font-weight: 600;
                    min-width: 150px;
                    text-align: center;
                    font-family: 'Inter', sans-serif !important;
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
                    font-weight: 600 !important;
                }
                
                .tab-nav button, .tabs .tab-nav button, div[role="tablist"] button {
                    color: #fff !important;
                    background-color: rgba(255,255,255,0.2) !important;
                    border: 1px solid rgba(255,255,255,0.3) !important;
                    backdrop-filter: blur(10px);
                    font-family: 'Inter', sans-serif !important;
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
                    font-weight: 600;
                }
        </style>
        """)
        
        current_page = gr.State(value=1)
        
        audio_data_state = gr.State()
        speakers_data_state = gr.State()
        final_analysis_state = gr.State()
        
        step_indicator = gr.HTML("""
        <div class="main-title"> Analizador de Sentimientos y Emociones en Audio en Español</div>
        <div class="step-indicator">
            <div class="step active">1. Carga de Audio</div>
            <div class="step pending">2. Transcripción e Identificación</div>
            <div class="step pending">3. Resultados</div>
        </div>
        """)
        
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
                    with gr.Accordion("⚙️ Configuración del Modelo Whisper", open=False):
                        gr.Markdown("""
                        ### Selección del modelo de transcripción. Tenga en cuenta que a mayor precisión, mayor necesidad de recursos
                        - **tiny**: Más rápido, menor precisión
                        - **base**: Balance velocidad/precisión 
                        - **small**: Buena precisión
                        - **medium**: Alta precisión
                        - **large**: Máxima precisión
                        """)
                        
                        with gr.Row():
                            model_selector = gr.Dropdown(
                                choices=[
                                    ("tiny (rápido)", "tiny"),
                                    ("base (equilibrado)", "base"),
                                    ("small (bueno)", "small"), 
                                    ("medium (alto)", "medium"),
                                    ("large (máximo)", "large")
                                ],
                                value="tiny",
                                label="Tamaño del modelo",
                                info="Modelos más grandes = mayor precisión pero más lento y mayor consumo de recursos"
                            )
                            
                            change_model_btn = gr.Button(
                                "Cambiar Modelo",
                                variant="secondary",
                                size="sm"
                            )
                        
                        model_status = gr.Textbox(
                            value=get_model_info(),
                            label="Estado del modelo",
                            interactive=False,
                            lines=1
                        )
                        
                        model_progress = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            step=1,
                            label="Cargando modelo...",
                            interactive=False,
                            show_label=True,
                            visible=False
                        )
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

                        current_config_display = gr.Textbox(
                            value=get_model_info(),
                            label="Configuración actual",
                            interactive=False,
                            lines=1
                        )
            
            process_btn = gr.Button(
                "Procesar Audio", 
                variant="primary", 
                size="lg"
            )
            
            with gr.Column(visible=False, elem_classes=["progress-container"]) as progress_container:
                gr.Markdown("### Progreso del Procesamiento")
                progress_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    label="Iniciando...",
                    interactive=False,
                    show_label=True
                )
        
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
            
            speaker_audio_1 = gr.Audio(label="Muestra Speaker 1", visible=False, interactive=False)
            speaker_name_1 = gr.Textbox(label="Nombre para Speaker 1", visible=False, interactive=True)
            speaker_audio_2 = gr.Audio(label="Muestra Speaker 2", visible=False, interactive=False)  
            speaker_name_2 = gr.Textbox(label="Nombre para Speaker 2", visible=False, interactive=True)
            speaker_audio_3 = gr.Audio(label="Muestra Speaker 3", visible=False, interactive=False)
            speaker_name_3 = gr.Textbox(label="Nombre para Speaker 3", visible=False, interactive=True)
            speaker_audio_4 = gr.Audio(label="Muestra Speaker 4", visible=False, interactive=False)
            speaker_name_4 = gr.Textbox(label="Nombre para Speaker 4", visible=False, interactive=True)
            speaker_audio_5 = gr.Audio(label="Muestra Speaker 4", visible=False, interactive=False)
            speaker_name_5 = gr.Textbox(label="Nombre para Speaker 4", visible=False, interactive=True)
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
            
            with gr.Column(visible=False, elem_classes=["progress-container"]) as progress2_container:
                gr.Markdown("### Progreso del Análisis")
                progress2_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    label="Esperando...",
                    interactive=False,
                    show_label=True
                )
        
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
                    
                    download_csv_btn = gr.Button(
                        "Descargar Tabla como CSV",
                        variant="primary",
                        size="lg"
                    )
                    
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
                    with gr.Row(equal_height=True):
                        plot_duration = gr.Image(label="Duración por Orador", visible=False, height=400)
                        plot_sentiment_orador = gr.Image(label="Sentimientos por Orador", visible=False, height=400)
                        plot_emotion_orador = gr.Image(label="Emociones por Orador", visible=False, height=400)
                    
                    with gr.Row(equal_height=True):
                        plot_sentiment_dist = gr.Image(label="Distribución Total de Sentimientos", visible=False, height=400)
                        plot_emotion_dist = gr.Image(label="Distribución Total de Emociones", visible=False, height=400)
                    
                    with gr.Row(equal_height=True):
                        plot_timeline = gr.Image(label="Evolución del Sentimiento", visible=False, height=400)
                        plot_timeline_speakers = gr.Image(label="Sentimientos por Participante en el Tiempo", visible=False, height=400)
                    
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
        
        def process_audio_with_progress(audio, speakers, language, remove_stopwords, min_tokens, progress=gr.Progress()):
            """Procesa el audio con barra de progreso visible usando el streaming function"""
            if audio is None:
                gr.Warning("Por favor, suba un archivo de audio")
                return [
                    gr.update(visible=False),  
                    gr.update(value=0, label="❌ Error: No se subió audio", interactive=False),        
                    None, None, None
                ]
            
            try:
                yield [
                    gr.update(visible=True),
                    gr.update(value=5, label="Iniciando procesamiento...", interactive=False),
                    None, None, None
                ]
                
                groupedDF = transcribe_with_whisperx_stream(audio, speakers, progress)
                
                yield [
                    gr.update(visible=True),
                    gr.update(value=80, label="Extrayendo muestras de voz...", interactive=False),
                    None, None, None
                ]
                
                progress(0.8, desc="Extrayendo muestras de voz...")
                orators, speaker_files = extract_speaker_samples(current_audioDF, audio)
                
                progress(0.9, desc="Preparando resultados...")
                transcript_text = "\n".join([
                    f"{row['speaker']}: {row['text'][:100]}..."
                    for _, row in groupedDF.iterrows()
                    if row['text'].strip()
                ])
                
                speakers_text = f"Detectados {len(orators)} participantes: {', '.join(orators)}"
                
                progress(1.0, desc="✅ ¡Procesamiento completado!")
                
                yield [
                    gr.update(visible=False),
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
                    gr.update(value=1),  
                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                    gr.update(value="<div>...step indicator...</div>"),
                    gr.update(visible=False), gr.update(visible=False),  
                    gr.update(visible=False), gr.update(visible=False),  
                    gr.update(visible=False), gr.update(visible=False), 
                    gr.update(visible=False), gr.update(visible=False),  
                    gr.update(visible=False), gr.update(visible=False)
                ]
            
            speakers = audio_data.get("speakers", [])
            speaker_files = audio_data.get("speaker_files", {})
            
            speaker_updates = []
            for i in range(5):
                if i < len(speakers):
                    speaker = speakers[i]
                    audio_file = speaker_files.get(speaker, None)
                    speaker_updates.extend([
                        gr.update(visible=True, value=audio_file, label=f"Muestra {speaker}"),
                        gr.update(visible=True, label=f"Nombre para {speaker}", interactive=True)
                    ])
                else:
                    speaker_updates.extend([
                        gr.update(visible=False),
                        gr.update(visible=False, interactive=True)
                    ])
            
            return [
                gr.update(value=2),
                gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                gr.update(value="""
                <div class="main-title"> Analizador de Sentimientos y Emociones en Audio en Español</div>
                <div class="step-indicator">
                    <div class="step completed">1. Carga de Audio</div>
                    <div class="step active">2. Transcripción e Identificación</div>
                    <div class="step pending">3. Resultados</div>
                </div>
                """)
            ] + speaker_updates
        
        def perform_complete_analysis(audio_data, name1, name2, name3, name4, name5, progress=gr.Progress()):
            """Realiza el análisis completo con barra de progreso visible"""
            global current_audioDF
            
            if audio_data is None:
                gr.Warning("No hay datos de audio para analizar")
                return [
                    gr.update(visible=False),
                    gr.update(value=0, label="❌ Error: No hay datos", interactive=False), 
                    None
                ]
            
            try:
                yield [
                    gr.update(visible=True),
                    gr.update(value=5, label="Iniciando análisis completo...", interactive=False),
                    None
                ]
                
                groupedDF = audio_data["grouped_df"]
                speakers = audio_data["speakers"]
                
                names = [name1, name2, name3, name4, name5]
                orator_names = {}
                for i, speaker in enumerate(speakers):
                    if i < len(names) and names[i] and names[i].strip():
                        orator_names[speaker] = names[i].strip()
                
                progress(0.05, desc="Aplicando nombres de participantes...")
                yield [
                    gr.update(visible=True),
                    gr.update(value=10, label="Aplicando nombres de participantes...", interactive=False),
                    None
                ]
                
                updated_audioDF = update_speaker_names(current_audioDF, orator_names)
                groupedDF = group_consecutive_speakers(updated_audioDF)
                
                progress(0.1, desc="Aplicando filtros de texto...")
                yield [
                    gr.update(visible=True),
                    gr.update(value=15, label="Aplicando filtros de texto...", interactive=False),
                    None
                ]
                filtered_audioDF = apply_text_filters(groupedDF.copy(), remove_stopwords=True, min_tokens=4)
                
                progress(0.15, desc="Iniciando análisis de sentimientos...")
                yield [
                    gr.update(visible=True),
                    gr.update(value=20, label="Analizando sentimientos y emociones...", interactive=False),
                    None
                ]
                filtered_audioDF = analyze_sentiments(filtered_audioDF, "es", progress)
                
                yield [
                    gr.update(visible=True),
                    gr.update(value=50, label="Análisis complementario en inglés...", interactive=False),
                    None
                ]
                filtered_audioDF = analyze_complementary_english(filtered_audioDF, progress)
                
                yield [
                    gr.update(visible=True),
                    gr.update(value=70, label="Análisis avanzado con Roberta...", interactive=False),
                    None
                ]
                filtered_audioDF = apply_roberta_analysis_and_replace(filtered_audioDF, progress)
                
                progress(0.8, desc="Generando visualizaciones...")
                yield [
                    gr.update(visible=True),
                    gr.update(value=85, label="Generando gráficos y visualizaciones...", interactive=False),
                    None
                ]
                plot_files = generate_analysis_plots(filtered_audioDF, progress)
                
                progress(1.0, desc="✅ ¡Análisis completo finalizado!")
                
                analysis_data = {
                    "filtered_audioDF": filtered_audioDF,
                    "plot_files": plot_files,
                    "transcript": "\n".join([f"{row['speaker']}: {row['text']}" for _, row in filtered_audioDF.iterrows()]),
                    "sentiment_data": []
                }
                
                for _, row in filtered_audioDF.iterrows():
                    if row['text_clean'].strip():
                        texto_original = row['text']
                        
                        texto_limpio = row['text_clean']
                        
                        duracion = f"{row['duration']:.1f}s"
                        
                        confianza = "85%" 
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
                
                yield [
                    gr.update(visible=False),
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
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update() 
                    ]
            
            plot_files = analysis_data.get("plot_files", [])
            
            plot_dict = {}
            for plot_file in plot_files:
                if os.path.exists(plot_file):
                    filename = os.path.basename(plot_file)
                    plot_dict[filename] = plot_file
            
            plot_updates = [
                gr.update(value=plot_dict.get("duracion_orador.png"), visible="duracion_orador.png" in plot_dict),
                gr.update(value=plot_dict.get("sentimiento_orador.png"), visible="sentimiento_orador.png" in plot_dict),
                gr.update(value=plot_dict.get("emocion_orador.png"), visible="emocion_orador.png" in plot_dict),
                gr.update(value=plot_dict.get("sentimiento_total.png"), visible="sentimiento_total.png" in plot_dict),
                gr.update(value=plot_dict.get("emocion_total.png"), visible="emocion_total.png" in plot_dict),
                gr.update(value=plot_dict.get("sentimiento_tiempo.png"), visible="sentimiento_tiempo.png" in plot_dict),
                gr.update(value=plot_dict.get("sentimiento_tiempo_speakers.png"), visible="sentimiento_tiempo_speakers.png" in plot_dict),
                gr.update(value=plot_dict.get("wordcloud.png"), visible="wordcloud.png" in plot_dict),
                gr.update(value=plot_dict.get("palabras_frecuentes.png"), visible="palabras_frecuentes.png" in plot_dict)
            ]
            
            return [
                gr.update(value=3),
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                gr.update(value="""
                    <div class="main-title"> Analizador de Sentimientos y Emociones en Audio en Español</div>
                    <div class="step-indicator">
                    <div class="step completed">1. Carga de Audio</div>
                    <div class="step completed">2. Transcripción e Identificación</div>
                    <div class="step active">3. Resultados</div>
                </div>"""),
                gr.update(value=analysis_data.get("transcript", "")),
                gr.update(value=analysis_data.get("sentiment_data", [])),
                gr.update(visible=False)
            ] + plot_updates
        
        
        change_model_btn.click(
            fn=change_model_with_progress,
            inputs=[model_selector],
            outputs=[model_progress, model_progress, model_status, current_config_display]
        )

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
                speaker_audio_4, speaker_name_4,
                speaker_audio_5, speaker_name_5
            ]
        )
        
        analyze_btn.click(
            fn=perform_complete_analysis,
            inputs=[audio_data_state, speaker_name_1, speaker_name_2, speaker_name_3, speaker_name_4, speaker_name_5],
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
        
        back_to_page1_btn.click(
            fn=lambda: [
                gr.update(value=1),
                gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                gr.update(value="""
                <div class="main-title"> Analizador de Sentimientos y Emociones en Audio en Español</div>
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
                <div class="main-title"> Analizador de Sentimientos y Emociones en Audio en Español</div>
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
                model_status, current_config_display,
                audio_data_state, speakers_data_state, final_analysis_state
            ]
        )
        
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

if __name__ == "__main__":
    demo = create_audio_analyzer_app()
    demo.queue()
    demo.launch()