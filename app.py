from flask import Flask, flash,render_template, request, redirect, url_for, session
import re
import collections
import contextlib
import sys
import wave
import webrtcvad
import librosa
from sklearn import preprocessing
import numpy as np
from sklearn.mixture import GaussianMixture
from copy import deepcopy
from sklearn.cluster import SpectralClustering
import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence
from textblob import TextBlob

app = Flask(__name__)
app.secret_key = 'TIGER'
relevance_factor = 16

@app.route('/', methods=['GET', 'POST'])
def select_audio():
    msg = 'Please select audio file'
    if request.method == 'POST' and 'audiopath' in request.form :
        audiopath = request.form['audiopath']
        session['audiopath']=audiopath
        return redirect(url_for('home'))
        
    return render_template('index.html', msg=msg)

@app.route("/home")
def home():

    return render_template("index.html",audiopath=session['audiopath'])

@app.route('/home/diarization', methods=['POST'])
def diratization():    
    print('hooo')
    if request.method == 'POST':
        print(2)       
        audiopath=session['audiopath']                
                # validate the received values    
        session['chunkPath']=audiopath+'output/'        
                    
                
        ########################### IMPLEMENTATION ###########################
        

        audio, sample_rate = read_wave(audiopath+'input/test.wav')
        vad = webrtcvad.Vad(2)
        frames = frame_generator(30, audio, sample_rate)
        frames = list(frames)
        segments = vad_collector(sample_rate, 30, 300, vad, frames)
        c = 0
        for i, segment in enumerate(segments):
            path = audiopath+'output/chunk-%002d.wav' % (i,)
            print(' Writing %s' % (path,))
            write_wave(path, segment, sample_rate)
            c +=1
        #count of chunks
        # c = 14

        sampling_rate = 8000
        n_mfcc = 13
        n_fft = 0.032
        hop_length = 0.010

        components = 16

        cov_type = 'full'

        ########################### Global GMM i.e UBM ###########################
        test_file_path = audiopath+"output/chunk-00.wav"
        y,sr = librosa.load(test_file_path)
        print(np.shape(y))

        mfcc = librosa.feature.mfcc(np.array(y),sr,hop_length=int(hop_length * sr),n_fft=int(n_fft*sr),n_mfcc=n_mfcc,dct_type=2)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_second_order = librosa.feature.delta(mfcc,order=2)
        temp = librosa.feature.delta(mfcc_delta)
        inter = np.vstack((mfcc,mfcc_delta,mfcc_delta_second_order))
        ubm_feature = inter.T
        #ubm_feature = preprocessing.scale(ubm_feature)

        # ubm_feature -= np.mean(ubm_feature)
        # ubm_feature /= np.std(ubm_feature)

        ubm_model = GaussianMixture(n_components = components, covariance_type = cov_type)
        ubm_model.fit(ubm_feature)

        print(ubm_model.score(ubm_feature))
        print(ubm_model.means_)       
                    
                
        Total = []
        for i in range(c):
            fname=audiopath+'output/chunk-%002d.wav' % (i,)
            print('MAP adaptation for {0}'.format(fname))
            temp_y,sr_temp = librosa.load(fname,sr=None)
            
            temp_mfcc = librosa.feature.mfcc(np.array(temp_y),sr_temp,hop_length=int(hop_length * sr_temp),n_fft=int(n_fft*sr_temp),n_mfcc=n_mfcc,dct_type=2)
            temp_mfcc_delta = librosa.feature.delta(temp_mfcc)
            temp_mfcc_delta_second_order = librosa.feature.delta(temp_mfcc,order=2)
            temp_inter = np.vstack((temp_mfcc,temp_mfcc_delta,temp_mfcc_delta_second_order))
            temp_gmm_feature = temp_inter.T
            #data = preprocessing.scale(temp_gmm_feature)

            gmm  = deepcopy(ubm_model)

            gmm = MAP_Estimation(gmm,temp_gmm_feature,m_iterations =1)
            
            sv = gmm.means_.flatten()
            #sv = preprocessing.scale(sv)
            Total.append(sv)

        N_CLUSTERS = 2        
        sc = SpectralClustering(n_clusters=N_CLUSTERS, affinity='cosine')

        #Labels help us identify between chunks of customer and call center agent
        labels = sc.fit_predict(Total)
        labels = rearrange(labels, N_CLUSTERS)
        print(labels)
        session['labels']=labels
        speaker0=[]
        speaker1=[]
        #Since there is no way to identify the voice of a customer just from the audio
        #we have assumed that customer is the one who speaks 2nd
        #Normally the call center agent is the first one to speak and then the customer
        #If that is not the case for a specific audio, change the condition from 'x==1' to 'x==0'
        for i, x in enumerate(labels) :
            if x == 0 :
                speaker0.append(i)
            elif x == 1 :    
                speaker1.append(i)

        print('speaker0:',speaker0)
        print('speaker1:',speaker1)
        session['speaker0']=speaker0
        session['speaker1']=speaker1

  

        return redirect(url_for('.home'))
    else:           
        return 'Error while diratization'

def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])




def MAP_Estimation(model,data,m_iterations):

    N = data.shape[0]
    D = data.shape[1]
    K = model.n_components


    mu_new = np.zeros((K,D))
    n_k = np.zeros((K,1))

    mu_k = model.means_
    
    pi_k = model.weights_

    old_likelihood = model.score(data)
    new_likelihood = 0
    iterations = 0
    while(iterations < m_iterations):
        iterations += 1
        old_likelihood = new_likelihood
        z_n_k = model.predict_proba(data)
        n_k = np.sum(z_n_k,axis = 0)
        n_k = n_k.reshape(np.shape(n_k)[0],1)

        mu_new = np.dot(z_n_k.T,data)
        n_k[n_k == 0] = 1e-20
        mu_new = mu_new / n_k

        adaptation_coefficient = n_k/(n_k + relevance_factor)
        I = np.ones(shape=np.shape(n_k))
        # for k in range(K):
        #     mu_k[k] = (adaptation_coefficient[k] * mu_new[k]) + ((1 - adaptation_coefficient[k]) * mu_k[k])
        mu_k = (adaptation_coefficient*mu_new) + (( I - adaptation_coefficient) * mu_k)
        model.means_ = mu_k

        log_likelihood = model.score(data)

        new_likelihood = log_likelihood

        if abs(old_likelihood - new_likelihood) < 1e-20:
            break
        print(log_likelihood)
    return model

def rearrange(labels, n):
    seen = set()
    distinct = [x for x in labels if x not in seen and not seen.add(x)]
    correct = [i for i in range(n)]
    dict_ = dict(zip(distinct, correct))
    return [x if x not in dict_ else dict_[x] for x in labels]

def get_sentiment(sentx):
    analysis=TextBlob(sentx)
    if analysis.sentiment.polarity > 0 :
        return ('positive')
    elif analysis.sentiment.polarity == 0 :
        return ('neutral')  
    else :
        return ('negative')  

def get_sentimentpol(sentx):
    analysis=TextBlob(sentx)
    return analysis.sentiment.polarity

@app.route('/home/sentimentAnalysis', methods=['POST'])
def sentimentAnalysis():    
    if request.method == 'POST':
        audiopath=session['audiopath']
        r = sr.Recognizer()
        speaker0Lines=[]
        speaker1Lines=[]
        speaker0=session['speaker0']
        speaker1=session['speaker1']
        total_duration=0
        for i in speaker0:
            # filename = "output/chunk-0{i}.wav"
            filename=audiopath+'output/chunk-%002d.wav' % (i,)
            with contextlib.closing(wave.open(filename,'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                total_duration+=duration
                print(duration)
            print(filename)
            r = sr.Recognizer()
            with sr.AudioFile(filename) as source:
                # listen for the data (load audio to memory)
                audio_data = r.record(source)
                # recognize (convert from speech to text)
                try:
                    text = r.recognize_google(audio_data)
                except sr.UnknownValueError as e:
                    print("Error:", str(e))
                else:
                    text = f"{text.capitalize()}. "
                    print(text)
                    speaker0Lines.append(text)
        print(speaker0Lines)
        session['duration0']=total_duration

        total_duration=0
        for i in speaker1:
            # filename = "output/chunk-0{i}.wav"
            filename=audiopath+'output/chunk-%002d.wav' % (i,)
            with contextlib.closing(wave.open(filename,'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                total_duration+=duration
                print(duration)
            print(filename)
            r = sr.Recognizer()
            with sr.AudioFile(filename) as source:
                # listen for the data (load audio to memory)
                audio_data = r.record(source)
                # recognize (convert from speech to text)
                try:
                    text = r.recognize_google(audio_data)
                except sr.UnknownValueError as e:
                    print("Error:", str(e))
                else:
                    text = f"{text.capitalize()}. "
                    print(text)
                    speaker1Lines.append(text)
        print(speaker1Lines)
        session['duration1']=total_duration

        sentiments_total0 = {'neutral':0,'positive':0,'negative':0}
        mood0=[]
        mood00=[]
        for recd_sent in speaker0Lines:
            sentiment = get_sentiment(recd_sent)
            print(sentiment,'===>',recd_sent)
            sentiments_total0[sentiment]= sentiments_total0[sentiment]+1
            mood0.append(get_sentimentpol(recd_sent))
            print('##################')
        print(sentiments_total0)    
        for i in range(len(mood0)):
            mood00.append(i+1)
        session['sentiments0']=sentiments_total0
        session['mood0']=mood0
        session['mood00']=mood00

        sentiments_total1 = {'neutral':0,'positive':0,'negative':0}
        mood1=[]
        mood10=[]
        for recd_sent in speaker1Lines:
            sentiment = get_sentiment(recd_sent)
            print(sentiment,'===>',recd_sent)
            sentiments_total1[sentiment]= sentiments_total1[sentiment]+1
            mood1.append(get_sentimentpol(recd_sent))
            print('##################')
        print(sentiments_total1)    
        for i in range(len(mood1)):
            mood10.append(i+1)
        session['sentiments1']=sentiments_total1
        session['mood1']=mood1
        session['mood10']=mood10

        return redirect(url_for('.home'))
    else:           
        return 'Error while diratization'

if __name__=="__main__":
    app.run(debug=True);