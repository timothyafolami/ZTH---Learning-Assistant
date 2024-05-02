from openai import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import librosa
import os
import soundfile as sf

load_dotenv()


OpenAI_key = os.getenv('openai_api_key')

client = OpenAI(api_key=OpenAI_key)

# Define a function to convert speech to text using the OpenAI audio transcriptions API
def speech_to_text(audio_data):
    # Open the audio file in binary mode
    with open(audio_data, "rb") as audio_file:
        # Create a transcription using the Whisper model
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file
        )
    # Return the transcript
    return transcript

def split_audio(input_file, output_folder, chunk_length=20):
    # Load the audio file
    y, sr = librosa.load(input_file, sr=None)

    # Calculate the number of samples in each chunk
    chunk_samples = chunk_length * sr * 60

    # Calculate the number of chunks
    num_chunks = len(y) // chunk_samples
    remainder = len(y) % chunk_samples
    if remainder > 0:
        num_chunks += 1

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Split the audio into chunks
    for i in range(num_chunks):
        start_sample = i * chunk_samples
        end_sample = min((i + 1) * chunk_samples, len(y))
        chunk = y[start_sample:end_sample]
        sf.write(os.path.join(output_folder, f"chunk_{i}.wav"), chunk, sr)

