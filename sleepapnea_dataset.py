#%%
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import mne
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------
# Helper
# ---------------------------
def time_to_sec(t):
    h, m, s = map(int, t.split(':'))
    return h*3600 + m*60 + s

def load_respevt(file_path):
    events = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:  # duration 포함
                continue
            try:
                time_to_sec(parts[0])
            except:
                continue
            # (시작시간, 이벤트타입, 지속시간)
            events.append((parts[0], parts[1], parts[2]))
    return events

def is_bad_window(x):
    return np.std(x) < 1e-6 or np.any(np.isnan(x)) or np.any(np.isinf(x))


def load_respevt(file_path):
    events = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                time_to_sec(parts[0])
                duration = int(parts[2])  # 실제 지속시간(초)
            except:
                continue

            event_type = parts[1]  # HYP-C, HYP-O, APNEA-O 등
            events.append((parts[0], event_type, duration))
    return events


from scipy.signal import resample_poly
from math import gcd

def resample(signal, orig_fs, target_fs):
    g = gcd(int(orig_fs), int(target_fs))
    up = int(target_fs) // g
    down = int(orig_fs) // g
    return resample_poly(signal, up, down)

from scipy.signal import butter, filtfilt

def bandpass(x, fs, low=0.5, high=40):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, x)


def clean_spo2(x):
    x = np.clip(x, 50, 100)
    # 스무딩 윈도우를 줄이거나 제거
    x = np.convolve(x, np.ones(3)/3, mode='same')  # 5→3으로 축소
    return x

def remove_spikes(x, z=5):
    m = np.mean(x)
    s = np.std(x) + 1e-8
    return np.clip(x, m - z*s, m + z*s)

# ---------------------------
# Optional: spectrogram
# ---------------------------
def sound_to_spec(signal, target_len):
    spec = torch.stft(
        torch.tensor(signal, dtype=torch.float32),
        n_fft=256,
        hop_length=128,
        return_complex=True
    )
    spec = torch.abs(spec)
    spec = spec.flatten()

    spec = torch.nn.functional.interpolate(
        spec.unsqueeze(0).unsqueeze(0),
        size=target_len,
        mode='linear'
    ).squeeze()

    return spec.numpy()


# ---------------------------
# Dataset
# ---------------------------
class SleepApneaDataset(Dataset):

    def __init__(self, root_dir, file_list, cfg):

        self.root_dir = root_dir
        self.file_list = file_list
        self.cfg = cfg
        self.samples = []

        print("🔥 Building dataset...")

        WINDOW_SEC = cfg.window_sec
        TARGET_FS = cfg.base_sample_rate
        DEBUG_MODE = getattr(cfg, "debug", False)
        USE_SPEC = getattr(cfg, "use_spectrogram", False)

        MAX_SEGMENTS = 999999
        STRIDE = 30
    

        # 🔥 전체 missing 카운트
        total_missing_sound = 0
        total_missing_spo2 = 0

        for subject in self.file_list:
            count = 0 
            print(f"\n--- {subject} ---")

            ecg_path = os.path.join(root_dir, subject + "_lifecard.edf")
            psg_path = os.path.join(root_dir, subject + ".edf")
            label_path = os.path.join(root_dir, subject + "_respevt.txt")

            if not (os.path.exists(ecg_path) and os.path.exists(psg_path)):
                print("❌ Missing file")
                continue

            try:
                raw_ecg = mne.io.read_raw_edf(ecg_path, preload=False)
                raw_psg = mne.io.read_raw_edf(psg_path, preload=False)

                ecg_fs = int(raw_ecg.info['sfreq'])
                psg_fs = int(raw_psg.info['sfreq'])

                ch_names = raw_psg.ch_names

                # 🔍 채널 찾기
                spo2_idx = [i for i, ch in enumerate(ch_names) if 'spo2' in ch.lower()]
                sound_idx = [i for i, ch in enumerate(ch_names)
                             if any(k in ch.lower() for k in ['sound', 'snore', 'mic'])]

                # 🔥 subject-level missing flag
                missing_spo2_flag = len(spo2_idx) == 0
                missing_sound_flag = len(sound_idx) == 0

                ecg_len_sec = raw_ecg.n_times / ecg_fs
                psg_len_sec = raw_psg.n_times / psg_fs
                total_sec = int(min(ecg_len_sec, psg_len_sec))

                print(f"⏱ usable seconds: {total_sec}")

                # ---------------------------
                # label
                # ---------------------------
                labels_full = np.zeros(int(total_sec * TARGET_FS))

                if os.path.exists(label_path):
                    events = load_respevt(label_path)

                    if len(events) > 0:
                        start_time = time_to_sec(events[0][0])

                        for t, etype, duration in events:
                            sec = time_to_sec(t) - start_time
                            start_idx = int(sec * TARGET_FS)
                            end_idx   = int((sec + duration) * TARGET_FS)  # ✅ 실제 duration 반영
                            end_idx   = min(end_idx, len(labels_full))      # ✅ 범위 초과 방지
                            labels_full[start_idx:end_idx] = 1

                # ---------------------------
                # sliding window
                # ---------------------------
            

                for start_sec in range(0, total_sec - WINDOW_SEC, STRIDE):

                    ecg_start = int(start_sec * ecg_fs)
                    ecg_end = int((start_sec + WINDOW_SEC) * ecg_fs)

                    psg_start = int(start_sec * psg_fs)
                    psg_end = int((start_sec + WINDOW_SEC) * psg_fs)

                    # ECG
                    ecg = raw_ecg.get_data(start=ecg_start, stop=ecg_end)[0]

                    # SpO2
                    if not missing_spo2_flag:
                        spo2 = raw_psg.get_data(
                            picks=[spo2_idx[0]],
                            start=psg_start,
                            stop=psg_end
                        )[0]
                    else:
                        spo2 = np.zeros_like(ecg)
                        total_missing_spo2 += 1

                    # Sound
                    if not missing_sound_flag:
                        sound = raw_psg.get_data(
                            picks=[sound_idx[0]],
                            start=psg_start,
                            stop=psg_end
                        )[0]
                    else:
                        sound = np.zeros_like(ecg)
                        total_missing_sound += 1
                        
                    # resample
                    ecg = resample(ecg, ecg_fs, TARGET_FS)
                    spo2 = resample(spo2, psg_fs, TARGET_FS)
                    sound = resample(sound, psg_fs, TARGET_FS)

                    ecg = bandpass(ecg, TARGET_FS, 0.5, 40)
                
                    spo2 = clean_spo2(spo2)
                    sound = remove_spikes(sound)

                    # spectrogram option
                    if USE_SPEC:
                        sound = sound_to_spec(sound, len(ecg))

                    # 길이 맞추기
                    min_len = min(len(ecg), len(spo2), len(sound))
                    ecg = ecg[:min_len]
                    spo2 = spo2[:min_len]
                    sound = sound[:min_len]

                    # label
                    label_ratio = np.mean(labels_full[
                        int(start_sec * TARGET_FS):
                        int((start_sec + WINDOW_SEC) * TARGET_FS)
                    ])

                   
                    label_seg = 1 if label_ratio >= 0.3 else 0

                    if is_bad_window(ecg) or is_bad_window(spo2):
                        continue

                    if (not missing_sound_flag) and is_bad_window(sound):
                        continue
                    sound_mask = 0 if missing_sound_flag else 1           
                    self.samples.append((ecg, spo2, sound, label_seg, sound_mask))
                    count += 1

                    if count % 100 == 0:
                        print("Segments:", count)

                    if count >= MAX_SEGMENTS:
                        break

                # 🔥 subject-level 로그
                if missing_spo2_flag:
                    print(f"⚠️ {subject}: SpO2 channel missing")

                if missing_sound_flag:
                    print(f"⚠️ {subject}: Sound channel missing")

            except Exception as e:
                print("❌ error:", e)
                continue

        # 🔥 전체 summary
        print("\n📊 Missing Channel Summary")
        print("Missing SpO2 segments:", total_missing_spo2)
        print("Missing Sound segments:", total_missing_sound)

        print("\n🔥 Total samples:", len(self.samples))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):

        ecg, spo2, sound, label, sound_mask = self.samples[idx]

        def normalize(x):
            return (x - np.mean(x)) / (np.std(x) + 1e-8)

        ecg = normalize(ecg)
        spo2 = normalize(spo2)

        if sound_mask == 1:
            sound = normalize(sound)
        else:
            sound = np.zeros_like(sound)

        sample = {
            'ECG': torch.tensor(ecg, dtype=torch.float32),
            'SpO2': torch.tensor(spo2, dtype=torch.float32),
            'sound': torch.tensor(sound, dtype=torch.float32),
            'sound_mask': torch.tensor(sound_mask, dtype=torch.float32)
        }

        return sample, torch.tensor(label, dtype=torch.long)