import pickle
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader


__all__ = ['MMDataLoader']


class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.train_mode = args['base']['train_mode']
        self.datasetName = args['dataset']['datasetName']
        self.dataPath = args['dataset']['dataPath']
        self.missing_rate_eval_test = args['base']['missing_rate_eval_test']
        self.missing_seed = args['base']['seed']
        # Which modalities should receive missing_rate_eval_test during evaluation.
        # Default keeps the original behavior: only Audio and Vision are degraded.
        # Use 'TAV' for a full-modality stress test. T and L both mean text/language.
        self.eval_missing_modalities = args.get('base', {}).get('eval_missing_modalities', 'AV')
        # self.missing_rate_eval_test = [0.0,0.0,1.0]# t a v


        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
        }
        DATA_MAP[self.datasetName]()

    def _normalize_eval_missing_modalities(self):
        mods = self.eval_missing_modalities
        if mods is None:
            return set(['A', 'V'])
        if isinstance(mods, str):
            mods = mods.replace(',', '').replace(' ', '').upper()
            # Accept L for language/text.
            mods = mods.replace('L', 'T')
            return set(mods)
        if isinstance(mods, (list, tuple, set)):
            normalized = set()
            for m in mods:
                m = str(m).strip().upper()
                if m in ('TEXT', 'LANGUAGE', 'L'):
                    m = 'T'
                elif m in ('AUDIO',):
                    m = 'A'
                elif m in ('VISION', 'VIDEO', 'V'):
                    m = 'V'
                if m:
                    normalized.add(m[0])
            return normalized
        raise TypeError(f"Unsupported eval_missing_modalities: {mods!r}")

    def _resolve_eval_missing_rates(self, n_samples):
        """Return [text_rate, audio_rate, vision_rate] arrays for evaluation.

        Supported config styles:
        - base.missing_rate_eval_test: 0.5 and base.eval_missing_modalities: 'AV' or 'TAV'
        - base.missing_rate_eval_test: [t_rate, a_rate, v_rate]
        - base.missing_rate_eval_test: {text: ..., audio: ..., vision: ...}
        """
        rate = self.missing_rate_eval_test

        def arr(x):
            return float(x) * np.ones((n_samples, 1), dtype=np.float32)

        if isinstance(rate, dict):
            t = rate.get('text', rate.get('language', rate.get('l', rate.get('T', 0.0))))
            a = rate.get('audio', rate.get('a', rate.get('A', 0.0)))
            v = rate.get('vision', rate.get('video', rate.get('v', rate.get('V', 0.0))))
            return [arr(t), arr(a), arr(v)]

        if isinstance(rate, (list, tuple, np.ndarray)):
            if len(rate) != 3:
                raise ValueError(
                    'missing_rate_eval_test as a list/tuple must be [text_rate, audio_rate, vision_rate].'
                )
            return [arr(rate[0]), arr(rate[1]), arr(rate[2])]

        base_rate = float(rate)
        mods = self._normalize_eval_missing_modalities()
        return [
            arr(base_rate if 'T' in mods else 0.0),
            arr(base_rate if 'A' in mods else 0.0),
            arr(base_rate if 'V' in mods else 0.0),
        ]

    def __init_mosi(self):
        with open(self.dataPath, 'rb') as f:
            data = pickle.load(f)
        
        self.data = data

        self.text = data[self.mode]['text_bert'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)

        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']
        self.labels = {
            'M': data[self.mode][self.train_mode+'_labels'].astype(np.float32),
            'missing_rate_l': np.zeros_like(data[self.mode][self.train_mode+'_labels']).astype(np.float32),
            'missing_rate_a': np.zeros_like(data[self.mode][self.train_mode+'_labels']).astype(np.float32),
            'missing_rate_v': np.zeros_like(data[self.mode][self.train_mode+'_labels']).astype(np.float32),
        }

        if self.datasetName == 'sims':
            for m in "TAV":
                self.labels[m] = data[self.mode][self.train_mode+'_labels_'+m]

        self.audio_lengths = data[self.mode]['audio_lengths']
        self.vision_lengths = data[self.mode]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0

        if self.mode == 'train':
            missing_rate = [np.random.uniform(size=(len(data[self.mode][self.train_mode+'_labels']), 1)) for i in range(3)]
            for i in range(3):
                sample_idx = random.sample([i for i in range(len(missing_rate[i]))], int(len(missing_rate[i])* 0.5))
                missing_rate[i][sample_idx] = 0
            self.labels['missing_rate_l'] = missing_rate[0]
            self.labels['missing_rate_a'] = missing_rate[1]
            self.labels['missing_rate_v'] = missing_rate[2]
        
        else:
            # Evaluation missing policy.
            # Original behavior was [0.0, r, r], i.e. text was always complete.
            # For full robustness testing, set base.eval_missing_modalities: 'TAV'
            # or set base.missing_rate_eval_test: [text_r, audio_r, vision_r].
            n_samples = len(data[self.mode][self.train_mode+'_labels'])
            missing_rate = self._resolve_eval_missing_rates(n_samples)
            self.labels['missing_rate_l'] = missing_rate[0]
            self.labels['missing_rate_a'] = missing_rate[1]
            self.labels['missing_rate_v'] = missing_rate[2]   
        self.text_m, self.text_length, self.text_mask, self.text_missing_mask = self.generate_m(self.text[:,0,:], self.text[:,1,:], None,
                                                                                missing_rate[0], self.missing_seed, mode='text')
        Input_ids_m = np.expand_dims(self.text_m, 1)
        Input_mask = np.expand_dims(self.text_mask, 1)
        Segment_ids = np.expand_dims(self.text[:,2,:], 1)
        self.text_m = np.concatenate((Input_ids_m, Input_mask, Segment_ids), axis=1) 

        self.audio_m, self.audio_length, self.audio_mask, self.audio_missing_mask = self.generate_m(self.audio, None, self.audio_lengths,
                                                                                    missing_rate[1], self.missing_seed, mode='audio')
        self.vision_m, self.vision_length, self.vision_mask, self.vision_missing_mask = self.generate_m(self.vision, None, self.vision_lengths,
                                                                                    missing_rate[2], self.missing_seed, mode='vision')

    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def generate_m(self, modality, input_mask, input_len, missing_rate, missing_seed, mode='text'):
        
        if mode == 'text':
            input_len = np.argmin(input_mask, axis=1)
        elif mode == 'audio' or mode == 'vision':
            input_mask = np.array([np.array([1] * length + [0] * (modality.shape[1] - length)) for length in input_len])
        np.random.seed(missing_seed)
        missing_mask = (np.random.uniform(size=input_mask.shape) > missing_rate.repeat(input_mask.shape[1], 1)) * input_mask
        
        assert missing_mask.shape == input_mask.shape
        
        if mode == 'text':
            # CLS SEG Token unchanged.
            for i, instance in enumerate(missing_mask):
                instance[0] = instance[input_len[i] - 1] = 1 #cls sep
            modality_m = missing_mask * modality + (100 * np.ones_like(modality)) * (input_mask - missing_mask) # UNK token: 100.
        elif mode == 'audio' or mode == 'vision':
            modality_m = missing_mask.reshape(modality.shape[0], modality.shape[1], 1) * modality
        return modality_m, input_len, input_mask, missing_mask


    def __len__(self):
        return len(self.labels['M'])

    def __getitem__(self, index):
        if (self.mode == 'train') and (index == 0):
            # missing_rate = [np.random.uniform(0, 0.5, size=(len(self.data[self.mode][self.train_mode+'_labels']), 1)) for i in range(3)]
            missing_rate = [np.random.uniform(size=(len(self.data[self.mode][self.train_mode+'_labels']), 1)) for i in range(3)]
            
            for i in range(3):
                sample_idx = random.sample([i for i in range(len(missing_rate[i]))], int(len(missing_rate[i])* 0.5))
                missing_rate[i][sample_idx] = 0

            self.labels['missing_rate_l'] = missing_rate[0]
            self.labels['missing_rate_a'] = missing_rate[1]
            self.labels['missing_rate_v'] = missing_rate[2]

            self.text_m, self.text_length, self.text_mask, self.text_missing_mask = self.generate_m(self.text[:,0,:], self.text[:,1,:], None,
                                                                                    missing_rate[0], self.missing_seed, mode='text')
            Input_ids_m = np.expand_dims(self.text_m, 1)
            Input_mask = np.expand_dims(self.text_mask, 1)
            Segment_ids = np.expand_dims(self.text[:,2,:], 1)
            self.text_m = np.concatenate((Input_ids_m, Input_mask, Segment_ids), axis=1) 

            self.audio_m, self.audio_length, self.audio_mask, self.audio_missing_mask = self.generate_m(self.audio, None, self.audio_lengths,
                                                                                        missing_rate[1], self.missing_seed, mode='audio')
            self.vision_m, self.vision_length, self.vision_mask, self.vision_missing_mask = self.generate_m(self.vision, None, self.vision_lengths,
                                                                                    missing_rate[2], self.missing_seed, mode='vision')


        sample = {
            'text': torch.Tensor(self.text[index]),
            'text_m': torch.Tensor(self.text_m[index]),
            'text_missing_mask': torch.Tensor(self.text_missing_mask[index]),
            'text_mask': torch.Tensor(self.text_mask[index]),
            'audio': torch.Tensor(self.audio[index]),
            'audio_m': torch.Tensor(self.audio_m[index]),
            # [新增] 导出音频的掩码和缺失掩码
            'audio_mask': torch.Tensor(self.audio_mask[index]),
            'audio_missing_mask': torch.Tensor(self.audio_missing_mask[index]),
            'vision': torch.Tensor(self.vision[index]),
            'vision_m': torch.Tensor(self.vision_m[index]),
            # [新增] 导出视觉的掩码和缺失掩码
            'vision_mask': torch.Tensor(self.vision_mask[index]),
            'vision_missing_mask': torch.Tensor(self.vision_missing_mask[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        }

        return sample


def MMDataLoader(args):
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args['base']['batch_size'],
                       num_workers=args['base']['num_workers'],
                       shuffle=True if ds == 'train' else False)
        for ds in datasets.keys()
    }
    
    return dataLoader


def MMDataEvaluationLoader(args):
    datasets = MMDataset(args, mode='test')

    dataLoader = DataLoader(datasets,
                       batch_size=args['base']['batch_size'],
                       num_workers=args['base']['num_workers'],
                       shuffle=False)
    
    return dataLoader