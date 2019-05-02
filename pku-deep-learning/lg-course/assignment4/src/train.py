from config import load_parameters
from nmt_keras.model_zoo import TranslationModel
from keras_wrapper.dataset import loadDataset
from keras_wrapper.callbacks import PrintPerformanceMetricOnEpochEndOrEachNUpdates
params = load_parameters()
dataset = loadDataset('../dataset/Dataset_nmt_dataset.pkl')



params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len['source_text']
params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len['target_text']


nmt_model = TranslationModel(params,
                             model_type='Transformer',
                             model_name='transformer_model',
                             vocabularies=dataset.vocabulary,
                             store_path='trained_models/transformer_model/',
                             verbose=True)

inputMapping = dict()
for i, id_in in enumerate(params['INPUTS_IDS_DATASET']):
    pos_source = dataset.ids_inputs.index(id_in)
    id_dest = nmt_model.ids_inputs[i]
    inputMapping[id_dest] = pos_source
nmt_model.setInputsMapping(inputMapping)

outputMapping = dict()
for i, id_out in enumerate(params['OUTPUTS_IDS_DATASET']):
    pos_target = dataset.ids_outputs.index(id_out)
    id_dest = nmt_model.ids_outputs[i]
    outputMapping[id_dest] = pos_target
nmt_model.setOutputsMapping(outputMapping)


extra_vars = {'language': 'en',
              'n_parallel_loaders': 8,
              'tokenize_f': eval('dataset.' + 'tokenize_none'),
              'beam_size': 12,
              'maxlen': 50,
              'model_inputs': ['source_text', 'state_below'],
              'model_outputs': ['target_text'],
              'dataset_inputs': ['source_text', 'state_below'],
              'dataset_outputs': ['target_text'],
              'normalize': True,
              'alpha_factor': 0.6,
              'val': {'references': dataset.extra_variables['val']['target_text']}
              }

vocab = dataset.vocabulary['target_text']['idx2words']
callbacks = []
callbacks.append(PrintPerformanceMetricOnEpochEndOrEachNUpdates(nmt_model,
                                                                dataset,
                                                                gt_id='target_text',
                                                                metric_name=['coco'],
                                                                set_name=['val'],
                                                                batch_size=50,
                                                                each_n_epochs=2,
                                                                extra_vars=extra_vars,
                                                                reload_epoch=0,
                                                                is_text=True,
                                                                index2word_y=vocab,
                                                                sampling_type='max_likelihood',
                                                                beam_search=True,
                                                                save_path=nmt_model.model_path,
                                                                start_eval_on_epoch=0,
                                                                write_samples=True,
                                                                write_type='list',
                                                                verbose=True))

training_params = {'n_epochs': 10,
                   'batch_size': 40,
                   'maxlen': 30,
                   'epochs_for_save': 1,
                   'verbose': 0,
                   'eval_on_sets': [],
                   'n_parallel_loaders': 8,
                   'extra_callbacks': callbacks,
                   'reload_epoch': 0,
                   'epoch_offset': 0}


nmt_model.trainNet(dataset, training_params)

