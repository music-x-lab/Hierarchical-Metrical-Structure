# Hierarchical Metrical Structure

Repository for the paper Learning Hierarchical Metrical Structure Beyond Measures (to be presented on ISMIR 2022).

## Demos

See ``output`` folder for demos from the validation and test split of RWC-Pop.

## Pretrained model

If you want to deploy the pre-trained model, download the pretrained weights ``simple_tcn_v2.0_filtered.sdict`` [here](https://drive.google.com/drive/folders/1vTuTQ0MaO5eru5h_bETjSGXKtVxlZrSw?usp=sharing) and put it in the ``cache_data`` folder.

## Configure datasets

* RWC Pop: you need to manually acquire RWC-Pop's midi files [here](https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/PR/AIST.RWC-MDB-P-2001.SMF_SYNC.zip) and put your own path in ``settings.py``.
For example, if you write in ``settings.py`` the following code:
```python
RWC_DATASET_PATH = '/my_dataset/rwc_pop_dataset'
```
Then all the RWC-Pop MIDI file should be located at:
```
/my_dataset/rwc_pop_dataset/AIST.RWC-MDB-P-2001.SMF_SYNC/RM-P001.SMF_SYNC.MID
/my_dataset/rwc_pop_dataset/AIST.RWC-MDB-P-2001.SMF_SYNC/RM-P002.SMF_SYNC.MID
/my_dataset/rwc_pop_dataset/AIST.RWC-MDB-P-2001.SMF_SYNC/RM-P003.SMF_SYNC.MID
...
```
* LMD dataset: to configure LMD dataset, download ``LMD-matched`` from [here](https://colinraffel.com/projects/lmd/) and put your own in ``settings.py``. For example, if you write in ``settings.py`` the following code:
```python
LMD_MATCHED_FOLDER = '/my_dataset/lmd_matched'
```
Then all the LMD MIDI file should be located at:
```
/my_dataset/lmd_matched/A/A/A/TRAAAGR128F425B14B/1d9d16a9da90c090809c153754823c2b.mid
/my_dataset/lmd_matched/A/A/A/TRAAAGR128F425B14B/5dd29e99ed7bd3cc0c5177a6e9de22ea.mid
/my_dataset/lmd_matched/A/A/A/TRAAAGR128F425B14B/b97c529ab9ef783a849b896816001748.mid
...
```
* Pop909 dataset: test midis already included in the repo.

## How to use

To run the pretrained model on custom MIDI files:

1. Make sure the MIDI file has correct metre, downbeat and beat labels derived from tempo change & time signature events (e.g., every tempo change events mark a downbeat position, you may use DAWs or python packages like ``pretty_midi`` to check this)
2. Run ``python simple_tcn_eval.py path/to/your_midi.mid``.
3. A prediction plot will be shown and a ``output/model_name/your_midi.mid_crf.mid`` file will be generated.
4. The output MIDI file contains an extra MIDI track with name ``Layers``. It is a drum track that labels L drum notes on a level-L boundary beyond measures. A downbeat without any drum notes is interpreted as a level-0 boundary.

To retrain the model:

1. Configure RWC dataset path as instructed above
2. Run ``data_preprocessing_train.py`` to generate the data npy file.
3. Run ``simple_tcn.py`` to train the model. The trained model will be saved in ``cache_data`` folder.

To reproduce experiments on RWC-Pop and Pop909:

1. Download the pretrained model weights as instructed above
2. Configure RWC dataset path as instructed above
3. Run ``python simple_tcn_eval.py``.

To reproduce experiments on baseline models:

1. Configure RWC dataset path as instructed above
2. Run ``midi_structure.py``.

To reproduce experiments on LMD-matched:

1. Download the pretrained model weights as instructed above
2. Configure LMD dataset path as instructed above
3. Run ``python statistics.py``.

