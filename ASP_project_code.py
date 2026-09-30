# -*- coding: utf-8 -*-
"""

@author: Rahul
"""

##****************Adaptive channel equalisation********************##
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import adi

#functions 
def bits_to_qpsk(bits):
    symbols = []
    for i in range(0, len(bits), 2):
        b1, b2 = bits[i], bits[i+1]

        if (b1, b2) == (0,0):
            symbols.append(1+1j)
        elif (b1, b2) == (0,1):
            symbols.append(-1+1j)
        elif (b1, b2) == (1,1):
            symbols.append(-1-1j)
        else:
            symbols.append(1-1j)

    return np.array(symbols)/np.sqrt(2)

#bit stream
def qpsk_to_bits(symbols):
    bits = []
    for s in symbols:
        I = 1 if s.real > 0 else -1
        Q = 1 if s.imag > 0 else -1

        if I > 0 and Q > 0:
            bits.extend([0,0])
        elif I < 0 and Q > 0:
            bits.extend([0,1])
        elif I < 0 and Q < 0:
            bits.extend([1,1])
        else:
            bits.extend([1,0])
    return np.array(bits)

def rrc_filter(beta,sps, num_taps):
    t = np.arange(num_taps) - (num_taps-1)//2
    t = t/sps
    h = np.zeros_like(t)
    for i in range(len(t)):
      if t[i] == 0.0:
        h[i] = 1 - beta + 4*beta/np.pi
      elif np.isclose(abs(t[i]), 1/(4*beta)):
        h[i] = (beta/np.sqrt(2)) * (
               (1 + 2/np.pi)*np.sin(np.pi/(4*beta)) +
               (1 - 2/np.pi)*np.cos(np.pi/(4*beta))
               )
      else:
        h[i] = (
               np.sin(np.pi*t[i]*(1-beta)) +
               4*beta*t[i]*np.cos(np.pi*t[i]*(1+beta))
               ) / (
               np.pi*t[i]*(1 - (4*beta*t[i])**2)
               )
    return h / np.sqrt(np.sum(h**2))


# For QPSK
def phase_detector_4(sample):
    I = np.real(sample)
    Q = np.imag(sample)
    a= 1 if I>0 else -1
    b= 1 if Q>0 else -1
    
    return (a * Q) - (b * I)


def binary_to_string(bits):    
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) == 8:
            chars.append(chr(int("".join(str(b) for b in byte), 2)))
    decoded_message = ''.join(chars)       
    return decoded_message

def string_to_binary(msg):
    binary_str = ''.join(format(ord(c), '08b') for c in msg)
    bits = np.array([int(b) for b in binary_str])
    return bits

def psd_fun(samples):
    psd1 = np.abs(np.fft.fftshift(np.fft.fft(samples)))**2
    psd_dB1 = 10*np.log10(psd1)
    f1 = np.linspace(sample_rate/-2, sample_rate/2, len(psd1))
    return f1, psd1, psd_dB1


def coarse_freq_synch(samples):
    samples_p = samples**4
    f1,psd1,psd_dB1 = psd_fun(samples_p)
    psd1 = np.convolve(psd1, np.ones(5)/5, mode='same')
    
    #adjusting frequency offset
    max_freq = f1[np.argmax(psd1)]
    #print(max_freq)
    Ts = 1/sample_rate     # calc sample period
    t = np.arange(0, Ts*len(samples), Ts) # create time vector
    samples_cf = samples * np.exp(-1j*2*np.pi*max_freq*t/4.0)
    return samples_cf

def timing_synh(samples,sps):
    mu = 0 # initial estimate of phase of sample
    out = np.zeros(len(samples) + 10, dtype=np.complex64)
    out_rail = np.zeros(len(samples) + 10, dtype=np.complex64) # stores values, each iteration we need the previous 2 values plus current value
    i_in = 0 # input samples index
    i_out = 2 # output index (let first two outputs be 0)
    while i_in + sps < len(samples):
        out[i_out] = samples[i_in] # grab what we think is the "best" sample
        out_rail[i_out] = ((1 if out[i_out].real > 0 else -1) +    1j*(1 if out[i_out].imag > 0 else -1))
        x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
        y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
        mm_val = np.real(y - x)
        gain = 0.01
        mu += sps + gain * mm_val
        i_in += int(np.floor(mu)) # round down to nearest int since we are using it as an index
        mu = mu - np.floor(mu) # remove the integer part of mu
        i_out += 1 # increment output index
    out = out[2:i_out] # remove the first two, and anything after i_out (that was never filled out)
  
    return out

def fine_freq_synch(samples):
    N = len(samples)
    phase = 0
    freq = 0
    alpha = 0.01
    beta = 0.0001
    out = np.zeros(N, dtype=complex)
    for i in range(N):
        out[i] = samples[i] * np.exp(-1j*phase) 
        error = phase_detector_4(out[i])
        
        # Advance the loop (recalc phase and freq offset)
        freq += (beta * error)
        phase += freq + alpha * error
    
        # Optional: Adjust phase so its always between 0 and 2pi, recall that phase wraps around every 2pi
        while phase >= 2*np.pi:
            phase -= 2*np.pi
        while phase < 0:
            phase += 2*np.pi
            
    return out

def frame_synch(samples):
    Lp = len(preamble_symbols)
    Lq = len(qpsk)
    k = 5
    corr = np.correlate(samples, preamble_symbols, mode='valid')
    power = np.sqrt(np.convolve(np.abs(samples)**2, np.ones(Lp), mode='valid'))
    corr = corr / (power + 1e-12)
    candidate_idxs = np.argsort(np.abs(corr))[-k:]

    best_score = -1
    best_aligned = None

    for idx in candidate_idxs:
        aligned = samples[idx:idx + Lq]
        if len(aligned) < Lp:
            continue
        
        phase_est = np.angle(np.vdot(preamble_symbols, aligned[:Lp]))
        aligned = aligned * np.exp(-1j * phase_est)

        # --- Fine timing refinement (small local search) ---
        for shift in range(4):
            test = aligned[shift:]
            if len(test) < Lp:
                continue
            
            test = test / np.sqrt(np.mean(np.abs(test)**2))
            err = np.sum(np.abs(test[:Lp] - preamble_symbols))
            score = -err              # lower error = higher score

            if score > best_score:
                best_score = score
                best_aligned = test

    if best_aligned is None:
        return samples  # fallback

    return best_aligned

def rotation_fix(y_eq):
    rotations = np.array([1, 1j, -1, -1j])
    best_score = -np.inf
    best_rot = 1
    Lp = len(preamble_symbols)

    for r in rotations:
        test = y_eq[:Lp] * r
        score = -np.sum(np.abs(test - preamble_symbols))
        if score > best_score:
            best_score = score
            best_rot = r

    y_eq *= best_rot

    return y_eq

def one_tap_equalizer(aligned2, preamble_symbols):
    x = aligned2 

    Lp = len(preamble_symbols)

    h = np.sum(x[:Lp] * np.conj(preamble_symbols)) / (
        np.sum(np.abs(preamble_symbols)**2) + 1e-12
    )

    y_eq = x / (h + 1e-12)
    return h, y_eq

def LMS(aligned2, preamble_symbols):
    x = aligned2 

    # --- Parameters ---
    L = 15
    mu_train = 0.001
    mu_dd = 0.0005

    # --- Init ---
    w = np.zeros(L, dtype=complex)
    w[L//2] = 1.0

    delay = L // 2
    x_pad = np.concatenate([np.zeros(delay, dtype=complex), x, np.zeros(delay, dtype=complex)])
    N = len(x)

    y = np.zeros(N, dtype=complex)

    # 1) TRAINING
    Lp = len(preamble_symbols)

    for n in range(Lp):
        x_vec = x_pad[n:n+L][::-1]
        y[n] = np.dot(w.conj(), x_vec)

        d = preamble_symbols[n]
        e = d - y[n]

        w += mu_train * e * x_vec
        
    # 2) DECISION DIRECTED
    for n in range(Lp, N):
        x_vec = x_pad[n:n+L][::-1]
        y[n] = np.dot(w.conj(), x_vec)

        d = (np.sign(y[n].real) + 1j*np.sign(y[n].imag)) / np.sqrt(2)
        e = d - y[n]

        w += mu_dd * e * x_vec

    # --- Remove delay ---
    y_eq = y

    return w, y_eq




def NLMS(aligned2, preamble_symbols):
    # --- Normalize once ---
    x = aligned2 

    # --- Parameters ---
    L = 15
    mu_train = 0.01
    mu_dd = 0.001
    eps = 1e-3

    # --- Initialize ---
    w = np.zeros(L, dtype=complex)
    w[L//2] = 1.0

    delay = L // 2

    # Pad only once
    x_pad = np.concatenate([np.zeros(delay, dtype=complex), x, np.zeros(delay, dtype=complex)])
    N = len(x)

    y = np.zeros(N, dtype=complex)

    # 1) TRAINING USING PREAMBLE

    Lp = len(preamble_symbols)

    for n in range(Lp):
        x_vec = x_pad[n:n+L][::-1]
        y[n] = np.dot(w.conj(), x_vec)

        d = preamble_symbols[n]   # known desired symbol
        e = d - y[n]

        norm = np.dot(x_vec.conj(), x_vec).real + eps
        w += (mu_train / norm) * e * x_vec

    # 2) DECISION-DIRECTED MODE

    for n in range(Lp, N):
        x_vec = x_pad[n:n+L][::-1]
        y[n] = np.dot(w.conj(), x_vec)

        # slicer (QPSK)
        d = (np.sign(y[n].real) + 1j*np.sign(y[n].imag)) / np.sqrt(2)

        e = d - y[n]

        norm = np.dot(x_vec.conj(), x_vec).real + eps
        w += (mu_dd / norm) * e * x_vec

 

    return w, y

def RLS(aligned2, preamble_symbols):
    x = aligned2 

    L = 15
    lam = 0.999
    delta = 10.0

    w = np.zeros((L,1), dtype=complex)
    w[L//2] = 1.0

    P = delta * np.eye(L, dtype=complex)

    delay = L // 2
    x_pad = np.concatenate([np.zeros(delay, dtype=complex), x, np.zeros(delay, dtype=complex)])
    N = len(x)

    y = np.zeros(N, dtype=complex)
    Lp = len(preamble_symbols)

    for n in range(Lp):
        x_vec = x_pad[n:n+L][::-1].reshape(-1,1)

        y[n] = (w.conj().T @ x_vec).item()

        d = preamble_symbols[n]
        e = d - y[n]

        Px = P @ x_vec
        den = lam + (x_vec.conj().T @ Px).item().real + 1e-6
        k = Px / den

        w = w + k * e.conj()
        P = (P - k @ (x_vec.conj().T @ P)) / lam

    # APPLY FILTER
    for n in range(N):
        x_vec = x_pad[n:n+L][::-1].reshape(-1,1)
        y[n] = (w.conj().T @ x_vec).item()

    return w.T, y


def ber_eq(rx_eq, method_name):
    rx_eq = rx_eq / (np.sqrt(np.mean(np.abs(rx_eq)**2)) + 1e-12)

    rx_bits_eq = qpsk_to_bits(rx_eq[len(preamble_symbols):])[:len(msg_bits)]
    tx_bits_eq = msg_bits
    
    # BER
    min_len = min(len(tx_bits_eq), len(rx_bits_eq))
    tx_bits_eq = tx_bits_eq[:min_len]
    rx_bits_eq = rx_bits_eq[:min_len]
    
    bit_errors_eq = np.sum(tx_bits_eq != rx_bits_eq)
    ber_eq = bit_errors_eq / min_len
        
    decoded_message = binary_to_string(rx_bits_eq[:len(tx_bits_eq)])
    
    return ber_eq, decoded_message
    
    
    
# ---------------- AWGN ----------------
def add_awgn(signal, snr_dB):
    sig_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_dB/10)
    noise_power = sig_power / snr_linear
    
    noise = np.sqrt(noise_power/2) * (
        np.random.randn(len(signal)) + 1j*np.random.randn(len(signal))
    )
    return signal + noise

def plot_constellation(sig, title):
    plt.figure()
    plt.scatter(sig.real, sig.imag, marker=".")
    plt.title(title)
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True)

def plot_amplitude(sig, title):
    plt.figure()
    plt.plot(np.abs(sig))
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Magnitude")

def plot_psd(f, psd_dB, title):
    plt.figure()
    plt.plot(f,psd_dB)
    plt.title(title)
    plt.xlabel("Sfrequency(hz")
    plt.ylabel("psd(dB)")

                 ### main code steps ###

fs = sample_rate = 1e6 
center_freq = 915e6
N = num_samps = int(1e5)         
sps=16
num_taps = 161
beta = 0.35
preamble = np.array([1,1,1,0,0,0,1,0,0,1])
preamble_symbols = bits_to_qpsk(preamble)
delay = num_taps//2

##### transmission data
msg = "hello good morning"
msg_bits = string_to_binary(msg)
msg_symbols = bits_to_qpsk(msg_bits)

bits = np.concatenate([preamble, msg_bits])  

#upsampling
qpsk = np.concatenate([preamble_symbols, msg_symbols])
upsampled = np.zeros(len(qpsk)*sps, dtype=complex)
upsampled[::sps] = qpsk

rrc = rrc_filter(beta,sps, num_taps)    #rrc filtering

# Create transmit waveform (QPSK, 16 samples per symbol)
tx_samples = np.convolve(upsampled,rrc)         # 16 samples per symbol (rectangular pulses)
tx_samples = tx_samples / np.sqrt(np.mean(np.abs(tx_samples)**2)) 
tx_samples *= 2**14     # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

                   ###### pluto config
                  
sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)

# Config Tx
sdr.tx_rf_bandwidth = int(sample_rate)    # filter cutoff, just set it to the same as sample rate
sdr.tx_lo = int(center_freq)
sdr.tx_hardwaregain_chan0 = 0.0     # Increase to increase tx power, valid range is -90 to 0 dB

# Config Rx
sdr.rx_lo = int(center_freq)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_buffer_size = num_samps
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 0.0      # dB, increase to increase the receive gain, but be careful not to saturate the ADC

# Start the transmitter
sdr.tx_cyclic_buffer = True # Enable cyclic buffers
sdr.tx(tx_samples) # start transmitting


# Receive samples
rx_samples = sdr.rx()
rx_samples = add_awgn(rx_samples, snr_dB=2)
rx_samples = np.convolve(rx_samples,rrc)
rx_samples = rx_samples[delay:]
rx_samples = rx_samples / np.sqrt(np.mean(np.abs(rx_samples)**2))
rx_samples = rx_samples[:int(1e5)]
sdr.tx_destroy_buffer()


                   ###Coarse frequecy simulation
rx_coarse = coarse_freq_synch(rx_samples)

                  #### muller loop (timing synch)
rx_timing = timing_synh(rx_coarse,sps)

                  ##### costas loop(fine freuency synch. to adjust phase)
rx_costas = fine_freq_synch(rx_timing)   

                              ##### frame synch
rx_frame = frame_synch(rx_costas)
rx_frame = rotation_fix(rx_frame)
aligned = rx_frame.copy()
                        
print("trasmitted message :", msg)

ber_frame, decoded_message = ber_eq(rx_frame, "frame synch")
print("Ber and Decoded Message after rls_equalisation :", ber_frame,decoded_message)


                ###### channel equalization
h, one_tap = one_tap_equalizer(aligned, preamble_symbols)
ber_1tap, decoded_message = ber_eq(one_tap, "one_tap_equalisation")
print("Ber and Decoded Message after one_tap_equalisation :",ber_1tap, decoded_message)

w_lms, lms_eq = LMS(aligned, preamble_symbols)
ber_lms, decoded_message = ber_eq(lms_eq, "lms_equalisation")
print("Ber and Decoded Message after lms_equalisation :",ber_lms, decoded_message)

w_nlms,nlms_eq = NLMS(aligned, preamble_symbols)
ber_nlms, decoded_message = ber_eq(nlms_eq, "nlms_equalisation")
print("Ber and Decoded Message after nlms_equalisation :", ber_nlms,decoded_message)

w_rls, rls_eq = RLS(aligned, preamble_symbols)
ber_rls, decoded_message = ber_eq(rls_eq, "rls_equalisation")
print("Ber and Decoded Message after rls_equalisation :", ber_rls, decoded_message)


plot_constellation(tx_samples, "transmitted Samples")
plot_constellation(rx_samples, "Received Signal after rrc")
plot_constellation(rx_coarse, "After Coarse Frequency Sync")
plot_constellation(rx_timing, "After Timing Sync")
plot_constellation(rx_costas, "After Costas Loop")
plot_constellation(rx_frame, "After Frame Synchronisation")
plot_constellation(one_tap, "one_tap equalisation(for flat fading)")
plot_constellation(lms_eq, "lms equalisation(basic adaptive eq)")
plot_constellation(nlms_eq, "After Nlms equalisation")
plot_constellation(rls_eq, "After Rls equalisation")

   
plot_amplitude(tx_samples, "Amplitude (Raw)")
plot_amplitude(rx_samples, "Amplitude (Raw)")
plot_amplitude(rx_coarse, "Amplitude (Coarse Sync)")
plot_amplitude(rx_timing, "Amplitude (Timing Sync)")
plot_amplitude(rx_costas, "Amplitude (Costas Loop)")
plot_amplitude(rx_frame, "Amplitude (Frame Sync)")

f1, psd1, psd_dB1 = psd_fun(rx_samples)
f2,psd2,psd_dB2 = psd_fun(rx_coarse)
f3,psd3,psd_dB3 = psd_fun(rx_timing)
f4,psd4,psd_dB4 = psd_fun(rx_costas)
f5, psd5, psd_dB5 = psd_fun(rx_frame)
    
plot_psd(f1, psd_dB1, "psd_after rrc")
plot_psd(f2, psd_dB2, "psd_after coarse sync")
plot_psd(f3, psd_dB3, "psd_after timing sync")
plot_psd(f4, psd_dB4, "psd_after costas sync")
plot_psd(f5, psd_dB5, "psd_after frame_sync")
plt.show()

print("Estimated channel coefficient (1-tap):\n", h)
print("\nEstimated weights — LMS:\n",  np.abs(w_lms))    
print("\nEstimated weights — NLMS:\n", np.abs(w_nlms))
print("\nEstimated weights — RLS:\n",  np.abs(w_rls))
 



  
                        #ber by adding noise in tx samples
# simulate channel once
msg = "thank you"
msg_bits = string_to_binary(msg)
msg_symbols = bits_to_qpsk(msg_bits)

preamble = np.array([1,1,1,0,0,0,1,0,0,1])
preamble_symbols = bits_to_qpsk(preamble)

bits = np.concatenate([preamble, msg_bits])  
qpsk = np.concatenate([preamble_symbols, msg_symbols])

spss = [4,8,16,32] 
snr_range = np.arange(-20, 20, 2 )  

symbols = qpsk
ber_dict_frame = {s: [] for s in spss}
ber_dict_1tap  = {s: [] for s in spss}
ber_dict_lms   = {s: [] for s in spss}
ber_dict_nlms  = {s: [] for s in spss}
ber_dict_rls   = {s: [] for s in spss}
for sps in spss:
    print(f"calculating for SPS = {sps}")
    
    # --- RRC ---
    span = 12
    num_taps = span * sps + 1
    delay = num_taps // 2
    rrc = rrc_filter(beta, sps, num_taps)

    # --- TX (fixed once per SPS) ---
    upsampled = np.zeros(len(qpsk)*sps, dtype=complex)
    upsampled[::sps] = qpsk

    tx_samples = np.convolve(upsampled, rrc)
    tx_samples /= np.sqrt(np.mean(np.abs(tx_samples)**2))

    for snr_dB in snr_range:

        trials_frame = []
        trials_1tap  = []
        trials_lms   = []
        trials_nlms  = []
        trials_rls   = []
        
        

        for _ in range(200):   # increased averaging
            rx_samples = add_awgn(tx_samples, snr_dB)
            rx_samples = np.convolve(rx_samples, rrc)
            rx_samples = rx_samples[delay:]
            rx_samples /= np.sqrt(np.mean(np.abs(rx_samples)**2))

            # --- SYNCH PIPELINE ---
            rx_coarse = coarse_freq_synch(rx_samples)
            rx_timing = timing_synh(rx_coarse, sps)
            rx_costas = fine_freq_synch(rx_timing)

            # --- FRAME ---
            rx_frame = frame_synch(rx_costas)
            rx_frame = rotation_fix(rx_frame)

            # normalize BEFORE equalization
            aligned = rx_frame.copy()

            # -------- BER --------
            ber_frame, _ = ber_eq(rx_frame, "frame")
            trials_frame.append(ber_frame)

            _, one_tap = one_tap_equalizer(aligned, preamble_symbols)
            ber_1tap, _ = ber_eq(one_tap, "1tap")
            trials_1tap.append(ber_1tap)

            _, lms_eq = LMS(aligned, preamble_symbols)
            ber_lms, _ = ber_eq(lms_eq, "lms")
            trials_lms.append(ber_lms)

            _, nlms_eq = NLMS(aligned, preamble_symbols)
            ber_nlms, _ = ber_eq(nlms_eq, "nlms")
            trials_nlms.append(ber_nlms)

            _, rls_eq = RLS(aligned, preamble_symbols)
            ber_rls, _ = ber_eq(rls_eq, "rls")
            trials_rls.append(ber_rls)

        # --- STORE AVERAGE ---
        ber_dict_frame[sps].append(np.mean(trials_frame))
        ber_dict_1tap[sps].append(np.mean(trials_1tap))
        ber_dict_lms[sps].append(np.mean(trials_lms))
        ber_dict_nlms[sps].append(np.mean(trials_nlms))
        ber_dict_rls[sps].append(np.mean(trials_rls))

for sps in spss:
  
    print(f"\n===== SPS = {sps} =====")
    print("SNR(dB) | before eq   1-TAP     LMS      NLMS     RLS")
    print("-------------------------------------------------------")

    for idx, snr in enumerate(snr_range):
        print(
            f"{snr:>6} | "
            f"{ber_dict_frame[sps][idx]:.5f}  "
            f"{ber_dict_1tap[sps][idx]:.5f}  "
            f"{ber_dict_lms[sps][idx]:.5f}  "
            f"{ber_dict_nlms[sps][idx]:.5f}  "
            f"{ber_dict_rls[sps][idx]:.5f}"
        )
    
    plt.figure()
    #plt.plot(snr_range, ber_dict_frame[sps], '.-', label = 'Before Eq')
    plt.plot(snr_range, ber_dict_1tap[sps], 'o-', label="1-TAP")
    plt.plot(snr_range, ber_dict_lms[sps], 's-', label="LMS")
    plt.plot(snr_range, ber_dict_nlms[sps], '^-', label="NLMS")
    plt.plot(snr_range, ber_dict_rls[sps], 'x-', label="RLS")

    plt.title(f"BER vs SNR for for SPS = {sps}")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.grid(True, which='both',linestyle='--', linewidth=0.5)
    plt.legend()
    

#plots for different methods
methods = {
    #"Frame Sync": ber_dict_frame,
    "1-Tap": ber_dict_1tap,
    "LMS": ber_dict_lms,
    "NLMS": ber_dict_nlms,
    "RLS": ber_dict_rls
}

for method_name, ber_dict in methods.items():
    plt.figure()

    for sps in spss:
        plt.plot(snr_range, ber_dict[sps], marker='.', label=f"SPS = {sps}")

    #plt.yscale('log')   # important for BER
    plt.title(f"BER vs SNR ({method_name})")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER (log scale)")

    plt.grid(True, which='both')
    plt.legend()
    
plt.show()

