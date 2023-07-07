import time
import torch
from bark.generation import (generate_text_semantic, preload_models,)
from bark import generate_audio, SAMPLE_RATE
import soundfile as sf
# from IPython.display import Audio


def synthesize(text_prompt, i):
    start_time = time.time()
    audio_array = generate_audio(text_prompt, history_prompt="output", directory='static')
    end_time = time.time()
    duration = end_time - start_time
    print(f"Time for syntesize {i}: {duration}")
    # sf.write(f"audio_{i}.wav", audio_array, SAMPLE_RATE)

if __name__=="__main__":
    preload_models()
    print("Synthesize Ready")
    text_prompt = """
Yeah, this is such an interesting story to me.
"""
    clip = "Yeah, this is a such an interesting story to me. The way this played out..."
    i = 0
    while True:
        synthesize(clip, i)
        clip = input("Type your text here: \n")
        i += 1
    
