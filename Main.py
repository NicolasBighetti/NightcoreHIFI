import librosa
#import IPython.display
import numpy as np

def vocals(voiced_track, sr):
        S_full, phase = librosa.magphase(librosa.stft(voiced_track))
        S_filter = librosa.decompose.nn_filter(S_full,
                                               aggregate=np.median,
                                               metric='cosine',
                                               width=int(librosa.time_to_frames(2, sr=sr)))
        S_filter = np.minimum(S_full, S_filter)
        margin_i, margin_v = 2, 10
        power = 2
        mask_i = librosa.util.softmask(S_filter,
                                       margin_i * (S_full - S_filter),
                                       power=power)
        mask_v = librosa.util.softmask(S_full - S_filter,
                                       margin_v * S_filter,
                                       power=power)
        s_foreground = mask_v * S_full
        s_background = mask_i * S_full

        d_foreground = s_foreground * phase
        d_background = s_background * phase

        y_foreground = librosa.istft(d_foreground)
        y_background = librosa.istft(d_background)

        return y_foreground, y_background


def main():
        print("Nightcore is very nice lol")


main()

audio_path = 'ninocore.wav'

track, rate = librosa.load(audio_path)
#track_shift = librosa.effects.pitch_shift(track, rate, 15)
#IPython.display.Audio(data=track_shift, rate=rate)

vocals, vclesstrack = vocals(track, rate)

librosa.output.write_wav("voice.wav", vocals, rate)
librosa.output.write_wav("track.wav", vclesstrack, rate)

