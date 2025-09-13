# Analizador de Sentimientos y Emociones

Este proyecto es una aplicación que permite analizar sentimientos y emociones en texto.

---------------------------------------------------------------------------------------------

Requerimientos

Antes de ejecutar la aplicación, asegurate de contar con lo siguiente instalado en tu dispositivo:

Python: https://www.python.org/downloads/ (versión 3.9 o superior recomendada).
FFmpeg: https://www.ffmpeg.org/download.html 

Importante: Una vez instalado FFmpeg, agregá la carpeta `bin` al PATH de tu sistema operativo para que pueda ser utilizado desde la línea de comandos.

---------------------------------------------------------------------------------------------

Configuración del token de Hugging Face

Es necesario contar con un token de autenticación de Hugging Face.

1. Creá un archivo llamado `secrets.env` en la carpeta Analizador-de-Sentimientos-y-Emociones-App.
2. Dentro de este archivo agregá la siguiente línea:

HF_TOKEN="TU_TOKEN"

3. Reemplazá 'TU_TOKEN' por el token que podés obtener en: https://huggingface.co/docs/hub/security-tokens

---------------------------------------------------------------------------------------------

Puesta en marcha

Seguí estos pasos para ejecutar la aplicación:

1. Abrí una terminal (CMD en Windows o terminal en Linux/Mac).

2. Navegá hasta el directorio del proyecto:
cd Analizador-de-Sentimientos-y-Emociones-App

3. Creá y activá un entorno virtual de Python:
python -m venv venv
venv\Scripts\activate   # En Windows
source venv/bin/activate  # En Linux/Mac

4. Instalá las dependencias necesarias:
pip install -r requirements.txt

5. Ejecutá la aplicación:
python app.py

6. Una vez corriendo, abrí en tu navegador el siguiente enlace:
http://localhost:7860/

Tu aplicación de análisis de sentimientos y emociones ya debería estar funcionando.
