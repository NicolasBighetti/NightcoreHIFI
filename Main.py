import librosa
from collections import OrderedDict
import numpy as np
from pysndfx import AudioEffectsChain
#import ffmpeg
import ffmpy
#https://stackoverflow.com/questions/43963982/python-change-pitch-of-wav-file

def vocals(voiced_track, sr):
        S_full, phase = librosa.magphase(voiced_track)
        S_filter = librosa.decompose.nn_filter(S_full,
                                               aggregate=np.median,
                                               metric='cosine',
                                               width=int(librosa.time_to_frames(2, sr=sr)))
        S_filter = np.minimum(S_full, S_filter)
        margin_i, margin_v = 15, 0.9
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

        return d_foreground, d_background


def vocalFromHB(vtrack, vrate, nb_steps=5, margin=5):
    if nb_steps == 0:
        print("end")
        return 0
    else:
        print("rec")
        t_vh, t_vb = librosa.decompose.hpss(vtrack, margin)
        vh, nt1 = vocals(t_vh, vrate)
        vb, nt2 = vocals(t_vb, vrate)
        return vocalFromHB((nt1 + nt2), vrate, nb_steps=(nb_steps-1)) + (vh + vb)



def main():
        print("Nightcore is very nice lol")


main()

core_fx = (
    AudioEffectsChain()
    .custom('vol -6db')
    .custom('bass +20')
)

voice_fx = (
    AudioEffectsChain()
    .custom('vol -3db')
    .custom('speed 1.25')
    .custom('pitch +38')
    .custom('compand 0.01,0.2 -85,-85 0.1')

)

correction_fx = (
    AudioEffectsChain()
    .custom('sinc 20k-10.3k')
    .normalize()
)

nightcoreify_fx = (
    AudioEffectsChain()
    .normalize()
    .custom('vol -3db')
    .custom('chorus 0.7 0.9 30 0.3 0.25 0.5 -t')
    .normalize()
    .reverb()
    .equalizer(427, 1000.0, -5.0)
)

audio_path = 'cauetcore.wav'

#mono=false
track, rate = librosa.load(audio_path, sr=44100)
#track, rate = librosa.load(audio_path, offset=0, duration=20, sr=44100)
print("Track loaded")
'''
print('useless analysis')
d_track = librosa.stft(track)
harmonic, beat = librosa.decompose.hpss(d_track, margin=5)
print("harmonic and beat separation done")
duration = librosa.core.get_duration(track)
etuning = librosa.estimate_tuning(track, rate, resolution=1e-3)
htuning = librosa.estimate_tuning(librosa.istft(harmonic), rate, resolution=1e-3)
print("Tunning : {}\nDuration : {}".format(etuning, duration))
print("Harmonic Tunning : {}".format(htuning))

print('beat analysis')
tempo, beats = librosa.beat.beat_track(track, rate)
#beat_samples = librosa.frames_to_samples(beats)
beats = librosa.frames_to_time(beats, rate)
print('Beats : {}'.format(beats))
'''
track = voice_fx(track, allow_clipping=False)
track = correction_fx(track)
track = core_fx(track, allow_clipping=True)
track = nightcoreify_fx(track)
print('Music modified')
librosa.output.write_wav('output/' + audio_path + '/fx_test.wav', track, rate)

print('Making the video')
'''
in_video = ffmpeg.input('pictures/1.jpg', framerate=25, loop=1, shortest=1)
in_video = ffmpeg.filter(in_video, 'scale', size='hd1080')
in_sound = ffmpeg.input('output/' + audio_path + '/fx_test.wav')
#in_joined = ffmpeg.output(in_video, in_sound, 'output/' + audio_path + '/fx_video_test.mp4', crf=20, preset='slower', movflags='faststart', pix_fmt='yuv420p')
in_joined = ffmpeg.output(in_video, in_sound, 'output/' + audio_path + '/fx_video_test.mp4', pix_fmt='yuv420p')
in_joined.run()
'''

fsp = ffmpy.FFmpeg(
    #global_options="-vcodec libx264",
    inputs={'output/' + audio_path + '/fx_test.wav': None},
    outputs={'output/' + audio_path + '/fsp.mp4': "-filter_complex \"[0:a]showfreqs=s=hd1080:mode=bar:ascale=log:win_func=flattop:overlap=0:colors=pink,scale=hd1080[v]\" -map \"[v]\" "}
)
print(fsp.cmd)
fsp.run()

inputs = OrderedDict([('pictures/8.jpg', None), ('output/' + audio_path + '/fsp.mp4', None)])
wov = ffmpy.FFmpeg(
    global_options="",
    inputs=inputs,
    outputs={'output/' + audio_path + '/fx_merge_test.mp4': "-shortest -filter_complex \"[0:v]scale=hd1080[bck];[1:v]format=argb,geq=r='r(X,Y)':a='0.5*alpha(X,Y)'[fb];[bck][fb]overlay[v]\" -map \"[v]\""}
)

print(wov.cmd)
wov.run()
#-filter_complex "[1:a]showfreqs=s=hd1080:mode=bar:ascale=log:win_func=flattop:overlap=0:colors=green,format=yuv420p[v]" -map "[v]" test_2.mkv
inputs = OrderedDict([('output/' + audio_path + '/fx_merge_test.mp4', None), ('output/' + audio_path + '/fx_test.wav', None)])

ff = ffmpy.FFmpeg(
    global_options="",
    inputs=inputs,
    outputs={'output/' + audio_path + '/fx_video_test.mp4': ['-vf', 'scale=1920:1080', '-shortest']}
)
print(ff.cmd)

ff.run()
#librosa.output.write_wav('output/' + audio_path + '/beat_samples_test.wav', librosa.istft(beat_samples), rate)
