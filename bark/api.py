from typing import Dict, Optional, Union

import numpy as np
import time
import soundfile as sf
from .generation import codec_decode, generate_coarse, generate_fine, generate_text_semantic


# import matplotlib.pyplot as plt

def text_to_semantic(
        text: str,
        history_prompt: Optional[Union[Dict, str]] = None,
        temp: float = 0.7,
        silent: bool = False,
):
    """Generate semantic array from text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar

    Returns:
        numpy semantic array to be fed into `semantic_to_waveform`
    """
    x_semantic = generate_text_semantic(
        text,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
        use_kv_caching=True
    )
    return x_semantic


def semantic_to_waveform(
        semantic_tokens: np.ndarray,
        history_prompt: Optional[Union[Dict, str]] = None,
        temp: float = 0.7,
        silent: bool = False,
        output_full: bool = False,
        directory=None,
        initial_index=0
):
    """Generate audio array from semantic input.

    Args:
        semantic_tokens: semantic token output from `text_to_semantic`
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """
    last_audio = None
    index = initial_index
    s = time.time()
    print(1, time.time())
    for coarse_tokens in generate_coarse(semantic_tokens, history_prompt=history_prompt, temp=temp, silent=silent,
                                         use_kv_caching=True):
        print(2, time.time())
        fine_tokens = generate_fine(
            coarse_tokens,
            history_prompt=history_prompt,
            temp=0.5,
        )
        print(3, time.time())
        audio_arr = codec_decode(fine_tokens)
        print(4, time.time())
        print(audio_arr.shape)
        # plt.plot(audio_arr)
        if last_audio is None:
            last_audio = audio_arr
            print("First Byte Generated: ", time.time() - s)
            start = 0
        else:
            start = len(last_audio)
            audio_arr[:len(last_audio)] = last_audio
            last_audio = audio_arr
        sf.write(f"{directory}/audio_{index}.wav", audio_arr[start:], 24000)
        index += 1
    # plt.show()
    if output_full:
        full_generation = {
            "semantic_prompt": semantic_tokens,
            "coarse_prompt": coarse_tokens,
            "fine_prompt": fine_tokens,
        }
        return full_generation, audio_arr
    print("Total Audio Length: ", len(audio_arr) / 24000)
    return index


def save_as_prompt(filepath, full_generation):
    assert (filepath.endswith(".npz"))
    assert (isinstance(full_generation, dict))
    assert ("semantic_prompt" in full_generation)
    assert ("coarse_prompt" in full_generation)
    assert ("fine_prompt" in full_generation)
    np.savez(filepath, **full_generation)


def generate_audio(
        text: str,
        history_prompt: Optional[Union[Dict, str]] = None,
        text_temp: float = 0.7,
        waveform_temp: float = 0.7,
        silent: bool = False,
        output_full: bool = False,
        directory=None,
        initial_index=0
):
    """Generate audio array from input text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """
    print(0, time.time())
    semantic_tokens = text_to_semantic(
        text,
        history_prompt=history_prompt,
        temp=text_temp,
        silent=silent,
    )
    final_index = semantic_to_waveform(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=waveform_temp,
        silent=silent,
        output_full=output_full,
        directory=directory,
        initial_index=initial_index
    )
    return final_index
