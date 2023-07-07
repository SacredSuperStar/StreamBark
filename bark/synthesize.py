import time
import nltk
from bark.generation import preload_models
from bark import generate_audio, SAMPLE_RATE



def synthesize(text_prompt, directory="static"):
    start_time = time.time()
    index = 0
    text = text_prompt.replace("\n", " ").strip()
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        index = generate_audio(sentence, history_prompt="en_fiery", directory=directory, initial_index=index)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Time for syntesize: {duration}")


if __name__ == "__main__":
    preload_models()
    print("Synthesize Ready")
    text_prompt = """
It looks like you opted into one of our ads lookin' for information on how to scale your business using AI. Do you remember that?
Hello, I'm really excited about optimizing bark with Air AI.
"""
    test_clip = "Hello, Thanks for visiting our bebe company. My name is Mark Fiery and I'm the sales assistant. How can I help you?"
    clip = "Hi, this is warm up synthesize."
    while True:
        synthesize(clip)
        clip = input("Type your text here: \n")
