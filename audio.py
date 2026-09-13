import numpy as np
import sounddevice as sd

def main():

    c = sine_tone(261.63, 1, 0.7)
    g = sine_tone(392, 1, 0.7)
    a = sine_tone(440, 1, 0.7)

    c_maj = major_chord(261.63)
    c_oct = octave(261.63)
    c_min = minor_chord(261.63)
    c_maj7 = major_seventh_chord(261.63)
    c_dom = dominant_seventh_chord(261.63)

    sd.play(c)
    sd.wait()

    sd.play(distort(c, 0.5))
    sd.wait()
    sd.play(distort(c, 0.4))
    sd.wait()
    sd.play(distort(c, 0.3))
    sd.wait()
    sd.play(distort(c, 0.2))
    sd.wait()
    sd.play(distort(c, 0.1))
    sd.wait()

    sd.play(c_oct)
    sd.wait()

    sd.play(c_maj)
    sd.wait()

    sd.play(c_maj7)
    sd.wait()

    sd.play(c_min)
    sd.wait()

    sd.play(c_dom)
    sd.wait()

def distort(input_wave: np.ndarray, amount: float):
    return np.clip(input_wave, -1.0, amount)

def play_note(note: str):
    if note == "c":
        sd.play(sine_tone(261.63, 3, 0.7), loop=True)
    elif note == "g":
        sd.play(sine_tone(392, 3, 0.7), loop=True)
    elif note == "a":
        sd.play(sine_tone(440, 3, 0.7), loop=True)

def octave(root_freq: float) -> np.ndarray:
    # 1:2 ratio
    root = sine_tone(root_freq)
    oct = sine_tone(root_freq * 2)

    chord = root + oct
    return chord

def major_chord(root_freq: float) -> np.ndarray:
    # 4:5:6 ratio
    root = sine_tone(root_freq)
    third = sine_tone(root_freq * (5/4))
    fifth = sine_tone(root_freq * (3/2))

    chord = root + third + fifth
    return chord

def minor_chord(root_freq: float) -> np.ndarray:
    # 10:12:15 ratio
    root = sine_tone(root_freq)
    minor_third = sine_tone(root_freq * (6/5))
    fifth = sine_tone(root_freq * (3/2))

    chord = root + minor_third + fifth
    return chord

def major_seventh_chord(root_freq: float) -> np.ndarray:
    # 8:10:12:15 ratio
    root = sine_tone(root_freq)
    third = sine_tone(root_freq * (5/4))
    fifth = sine_tone(root_freq * (3/2))
    maj_seventh = sine_tone(root_freq * (15/8))

    chord = root + third + fifth + maj_seventh
    return chord

def dominant_seventh_chord(root_freq: float) -> np.ndarray:
    # 4:5:6:7 ratio
    root = sine_tone(root_freq)
    third = sine_tone(root_freq * (5/4))
    fifth = sine_tone(root_freq * (3/2))
    nat_seventh = sine_tone(root_freq * (7/4))

    chord = root + third + fifth + nat_seventh
    return chord

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