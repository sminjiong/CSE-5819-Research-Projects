#%%
import os


class SleepApnea:

    def __init__(self, root_dir):
    
    
        # 🔥 root_dir 그대로 사용 (덮어쓰기 ❌)
        self.root_dir = root_dir

        # 🔥 subject 리스트 (ECG + REC 둘 다 있는 것만)
        all_files = os.listdir(root_dir)

        all_subjects = sorted([
            f.split("_")[0]
            for f in all_files
            if f.endswith("_lifecard.edf") and
               f.replace("_lifecard.edf", ".rec") in all_files
        ])

        import random

        random.seed(42)      # 🔥 중요 (재현성)
        random.shuffle(all_subjects)

        self.modalities = ['ECG', 'SpO2', 'sound']

        self.variates = {
            'ECG': 1,
            'SpO2': 1,
            'sound': 1
        }


        self.num_classes = 2   # apnea vs normal
        self.duration = 30              # 30초 window
        self.base_sample_rate = 100     # 모든 modality를 100Hz로 맞출 것
        self.input_length = self.duration * self.base_sample_rate
        # self.train_set = all_subjects[:18]
        # self.val_set   = all_subjects[18:22]
        # self.eval_set  = all_subjects[22:]

        # self.train_set = all_subjects[:17]
        # self.val_set   = all_subjects[17:22]
        # self.eval_set  = all_subjects[22:]
        # dataset_cfg.py 수정
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(kf.split(all_subjects))

        # fold 번호 선택 (0~4)
        fold_idx = 0
        train_idx, test_idx = folds[fold_idx]

        train_subjects = [all_subjects[i] for i in train_idx]  # 20명
        test_subjects  = [all_subjects[i] for i in test_idx]   # 5명

        # train에서 val 분리 (train 20명 중 4명)
        self.val_set   = train_subjects[:4]
        self.train_set = train_subjects[4:]
        self.eval_set  = test_subjects
        self.window_sec = 30
    
        print("Total subjects:", len(all_subjects))
        print("Train:", self.train_set)
        print("Val:", self.val_set)
        print("Test:", self.eval_set)
        
        print("length (should be 25) :", len(all_subjects))
