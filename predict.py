from tensorflow.python.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm


def make_prediction(args):

    model = load_model(args.model_fn,
        custom_objects={'STFT':STFT,
                        'Magnitude':Magnitude,
                        'ApplyFilterbank':ApplyFilterbank,
                        'MagnitudeToDecibel':MagnitudeToDecibel})
    wav_paths = glob('{}/**'.format(args.src_dir), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
    classes = sorted(os.listdir(args.src_dir))
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    le = LabelEncoder()
    y_true = le.fit_transform(labels)
    results = []

    for z, wav_fn in tqdm(enumerate(wav_paths), total=len(wav_paths)):
        rate, wav = downsample_mono(wav_fn, args.sr)
        mask, env = envelope(wav, rate, threshold=args.threshold)
        clean_wav = wav[mask]
        step = int(args.sr*args.dt)
        batch = []

        for i in range(0, clean_wav.shape[0], step):
            sample = clean_wav[i:i+step]
            sample = sample.reshape(-1, 1)
            if sample.shape[0] < step:
                tmp = np.zeros(shape=(step, 1), dtype=np.float32)
                tmp[:sample.shape[0],:] = sample.flatten().reshape(-1, 1)
                sample = tmp
            batch.append(sample)
        X_batch = np.array(batch, dtype=np.float32)
        y_pred = model.predict(X_batch)
        y_mean = np.mean(y_pred, axis=0)
        y_pred = np.argmax(y_mean)
        real_class = os.path.dirname(wav_fn).split('/')[-1]
        print('Actual class: {}, Predicted class: {}'.format(real_class, classes[y_pred]))
        results.append(y_mean)

    np.save(os.path.join('logs', args.pred_fn), np.array(results))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_fn', type=str, default='models/lstm.h5',
                        help='model file to make predictions')
    parser.add_argument('--pred_fn', type=str, default='y_pred',
                        help='fn to write predictions in logs dir')
    parser.add_argument('--src_dir', type=str, default='wavfiles',
                        help='directory containing wavfiles to predict')
    parser.add_argument('--dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='sample rate of clean audio')
    parser.add_argument('--threshold', type=str, default=20,
                        help='threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()

    make_prediction(args)

# audio_classes = [
#     'burping',
#     'hungry',
#     'noise',
#     'sick',
#     'tired',
# ]

# def make_prediction(args):
#     test_audio = 'wavfiles/burp-1.wav'

#     audio_result = []
    

#     model = load_model(args.model_fn,
#         custom_objects={'STFT':STFT,
#                         'Magnitude':Magnitude,
#                         'ApplyFilterbank':ApplyFilterbank,
#                         'MagnitudeToDecibel':MagnitudeToDecibel})
#     wav_paths = glob('{}/**'.format(args.src_dir), recursive=True)
#     wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
#     classes = audio_classes
#     labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
#     le = LabelEncoder()
#     y_true = le.fit_transform(labels)
#     results = []


#     rate, wav = downsample_mono(test_audio, args.sr)
#     mask, env = envelope(wav, rate, threshold=args.threshold)
#     clean_wav = wav[mask]
#     step = int(args.sr*args.dt)
#     batch = []

#     for i in range(0, clean_wav.shape[0], step):
#         sample = clean_wav[i:i+step]
#         sample = sample.reshape(-1, 1)
#         if sample.shape[0] < step:
#             tmp = np.zeros(shape=(step, 1), dtype=np.float32)
#             tmp[:sample.shape[0],:] = sample.flatten().reshape(-1, 1)
#             sample = tmp
#         batch.append(sample)


#         X_batch = np.array(batch, dtype=np.float32)
#         y_pred = model.predict(X_batch)
#         y_mean = np.mean(y_pred, axis=0)
#         y_pred = np.argmax(y_mean)
#         real_class = os.path.dirname(test_audio).split('/')[-1]
#         print('Actual class: {}, Predicted class: {}'.format(real_class, classes[y_pred]))
#         results.append(y_mean)
#         audio_result.append(classes[y_pred])

        

   

#     counter = 0
#     num = audio_result[0]

#     for i in audio_result:
#         curr_frequency = audio_result.count(i)
#         if curr_frequency > 1:
#             counter = curr_frequency
#             num = i


#     return num
        

#         # print(audio_result)

#     # np.save(os.path.join('logs', args.pred_fn), np.array(results))


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description='Audio Classification Training')
#     parser.add_argument('--model_fn', type=str, default='models/lstm.h5',
#                         help='model file to make predictions')
#     parser.add_argument('--pred_fn', type=str, default='y_pred',
#                         help='fn to write predictions in logs dir')
#     parser.add_argument('--src_dir', type=str, default='wavfiles',
#                         help='directory containing wavfiles to predict')
#     parser.add_argument('--dt', type=float, default=1.0,
#                         help='time in seconds to sample audio')
#     parser.add_argument('--sr', type=int, default=16000,
#                         help='sample rate of clean audio')
#     parser.add_argument('--threshold', type=str, default=20,
#                         help='threshold magnitude for np.int16 dtype')
#     args, _ = parser.parse_known_args()

#     make_prediction(args)

