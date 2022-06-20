import librosa
import numpy as np
import pickle
import os

AUDIO_FOLDER = "data/audios/utterances_final"
SAVE_FEATURES_PATH = "data/audio_new_2.p"


def librosa_feat(file_path):
    y, sr = librosa.load(file_path)
    hop_length = 512

    D = librosa.stft(y, hop_length=hop_length)
    S_full, phase = librosa.magphase(D)
    S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric="cosine",
                                           width=int(librosa.time_to_frames(0.2, sr=sr)))
    S_filter = np.minimum(S_full, S_filter)

    margin_v = 4
    power = 2
    mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)
    S_foreground = mask_v * S_full

    new_D = S_foreground * phase
    y = librosa.istft(new_D)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)

    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    melspectrogram_delta = librosa.feature.delta(melspectrogram)

    spectral_centroid = librosa.feature.spectral_centroid(S=S_full)
    spectral_contrast = librosa.feature.spectral_contrast(S=S_full)
    spectral_flatness = librosa.feature.spectral_flatness(S=S_full)
    spectral_rolloff = librosa.feature.spectral_rolloff(S=S_full)

    audio_feature = np.vstack((mfcc, mfcc_delta, melspectrogram, melspectrogram_delta, 
        spectral_centroid, spectral_contrast, spectral_flatness, spectral_rolloff))

    jump = int(audio_feature.shape[1] / 10)
    return librosa.util.sync(audio_feature, range(1, audio_feature.shape[1], jump))


def create_audio_features():
    os.chdir('..')
    audio_feature = {}
    for filename in os.listdir(AUDIO_FOLDER):
        id_ = filename.rsplit(".", maxsplit=1)[0]
        audio_feature[id_] = librosa_feat(os.path.join(AUDIO_FOLDER, filename))
        print(audio_feature[id_].shape)

    with open(SAVE_FEATURES_PATH, "wb") as file:
        pickle.dump(audio_feature, file, protocol=2)