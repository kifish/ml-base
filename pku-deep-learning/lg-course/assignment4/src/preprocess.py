from keras_wrapper.dataset import Dataset, saveDataset
from data_engine.prepare_data import keep_n_captions
ds = Dataset('nmt_dataset', 'nmt', silence=False)

ds.setOutput('../dataset/train.chn.txt',
             'train',
             type='text',
             id='target_text',
             tokenization='tokenize_none',
             build_vocabulary=True,
             pad_on_batch=True,
             sample_weights=True,
             max_text_len=30,
             max_words=30000,
             min_occ=0)

ds.setOutput('../dataset/val.chn.txt',
             'val',
             type='text',
             id='target_text',
             pad_on_batch=True,
             tokenization='tokenize_none',
             sample_weights=True,
             max_text_len=30,
             max_words=0)


ds.setInput('../dataset/train.eng.txt',
            'train',
            type='text',
            id='source_text',
            pad_on_batch=True,
            tokenization='tokenize_none',
            build_vocabulary=True,
            fill='end',
            max_text_len=30,
            max_words=30000,
            min_occ=0)
ds.setInput('../dataset/val.eng.txt',
            'val',
            type='text',
            id='source_text',
            pad_on_batch=True,
            tokenization='tokenize_none',
            fill='end',
            max_text_len=30,
            min_occ=0)




ds.setInput('../dataset/train.eng.txt',
            'train',
            type='text',
            id='state_below',
            required=False,
            tokenization='tokenize_none',
            pad_on_batch=True,
            build_vocabulary='target_text',
            offset=1,
            fill='end',
            max_text_len=30,
            max_words=30000)
ds.setInput(None,
            'val',
            type='ghost',
            id='state_below',
            required=False)

# If we had multiple references per sentence
keep_n_captions(ds, repeat=1, n=1, set_names=['val'])

saveDataset(ds, '../dataset')