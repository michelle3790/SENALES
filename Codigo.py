import streamlit as st
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter, freqz
from scipy.io import wavfile
from scipy.io.wavfile import write
from scipy.io.wavfile import read
from scipy.signal import upfirdn

audio_file = open("C:/Users/miche/Downloads/Audioparcial.wav", 'rb')
audio_bytes = audio_file.read()

st.subheader("Señal original: ")
st.audio(audio_bytes, format='Audioparcial/wav')

framerate_sf, signal = wavfile.read("C:/Users/miche/Downloads/Audioparcial.wav")
t = len(signal)/framerate_sf
time_sf = np.arange(0,t, (1/framerate_sf))
st.write(signal.dtype)

fft = np.fft.fft(signal)
f1 = np.fft.fftfreq(fft.size)*framerate_sf

fig, (ax1, ax2) = plt.subplots(2)
st_a = st.pyplot(plt)
fig.suptitle('Vertically stacked subplots')
ax1.plot(time_sf, signal)
ax2.plot(f1, abs(fft))
st_a.pyplot(plt)



u2 = lambda t: np.piecewise(t,t>=2,[1,0])
u4 = lambda t: np.piecewise(t,t>=4,[1,0])
rectangular = lambda t:u2(t) - u4(t)
rect_i = rectangular((f1/500)+2)

fig6, graph10 =plt.subplots()
st_h = st.pyplot(plt)
plt.plot(f1,rect_i)
plt.xlabel('f')
plt.grid()


conv = fft*rect_i
st.write(conv.dtype)
fig3, ax3 = plt.subplots()
st_c= st.pyplot(plt)
ax3.plot(f1, abs(conv))

st_c.pyplot(plt)


n_audio = np.fft.ifft(conv)

fig4, ax5 = plt.subplots(1,1,figsize=(15, 6))
st_e= st.pyplot(plt)
ax5.plot(time_sf, n_audio)

st_e.pyplot(plt)

write('nuevapista.wav', framerate_sf, n_audio.astype(np.int16))

audio_file3 = open('nuevapista.wav', 'rb')
audio_bytes3 = audio_file3.read()


st.audio(audio_bytes3, format='audio/wav')


