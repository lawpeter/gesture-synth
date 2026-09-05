import numpy as np
import sounddevice as sd

def main():

    # Generate and play sound

    # mysound = white_noise()
    # sd.play(mysound)
    # sd.wait()

    # mysound2 = sine_tone()
    # sd.play(mysound2)
    # sd.wait()

    # mysound3 = sine_tone(261.63, 2, 0.7)
    # sd.play(mysound3)
    # sd.wait()

    c = sine_tone(261.63, 0.5, 0.7)
    g = sine_tone(392, 0.5, 0.7)
    g2 = sine_tone(392, 1, 0.7)
    a = sine_tone(440, 0.5, 0.7)

    sd.play(c)
    sd.wait()
    sd.play(c)
    sd.wait()
    sd.play(g)
    sd.wait()
    sd.play(g)
    sd.wait()
    sd.play(a)
    sd.wait()
    sd.play(a)
    sd.wait()
    sd.play(g2)
    sd.wait()

def play_note(note: str):
    if note == "c":
        sd.play(sine_tone(261.63, 3, 0.7), loop=True)
    elif note == "g":
        sd.play(sine_tone(392, 3, 0.7), loop=True)
    elif note == "a":
        sd.play(sine_tone(440, 3, 0.7), loop=True)

def sine_tone(
        frequency: int=440,
        duration: float=1.0,
        amplitude: float=0.5,
        sample_rate: int=44100
        ) -> np.ndarray:

    # Calculate number of samples needed
    n_samples = int(duration * sample_rate)

    # Create an array of time points
    time_points = np.linspace(0, duration, n_samples, False)

    # Create sine wave
    sine = np.sin(2 * np.pi * frequency * time_points)

    # Apply amplitude and return the tone
    sine *= amplitude
    return sine

def white_noise(
        duration: float=1.0, 
        amplitude: float=0.5, 
        sample_rate: int=44100 
        ) -> np.ndarray:

    # Calculate number of samples needed
    n_samples = int(duration * sample_rate)

    noise = np.random.uniform(-1, 1, n_samples)
    noise *= amplitude

    return noise

if __name__ == "__main__":
    main()