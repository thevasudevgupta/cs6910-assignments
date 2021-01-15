# About Me

```python
pip install -r requirements.txt
# below script will fetch you text-dataset and prepare vocab in `data` directory
python nlp_preprocessing.py

# For running scripts in Part-A
cd part-A
python word2vec.py [--options]
python save_embeddings.py [--options]

# For running scripts in Part-B
cd part-B
# You need to download glove manually and keep `glove.6B.300d.txt` in `data` directory
python prepare_glove.py

python sentiment_analysis.py [--options]
python get_confusion_matrix.py [--options]
```

# Repositary structure

```
    .
    |- data/                        this hosts train.csv, test.csv, val.csv, text8.txt, vocab.txt and embedding saved from part-A
    |- nlp_preprocessing.py         this will prepare data for word2vec training and save it in data dir
    |- part-A
        |- word2vec.py              run this script to start word2vec training
        |- save_embeddings.py       run this script to extract embeddings from weights saved in weights directory into data directory
        |- weights/                 keep all the weights trained from word2vec training in this directory
        |- dataloader.py            having DataLoader class
        |- utils.py                 having model architecture and other utilities used in training
        |- trainer.py               having Trainer class
    |- part-B
        |- sentiment_analysis.py    run this script to start classification model training
        |- get_confusion_matrix.py  run this script to print confusion matrix of the predictions
        |- prepare_glove.py         run this script will load downloaded embedding from official site and save glove them in torch tensor format.
        |- weights/                 keep all the weights trained from sentiment-analysis training in this directory
        |- dataloader.py            having DataLoader class
        |- utils.py                 having model architecture and other utilities used in training
        |- trainer.py               having Trainer class
    |- me18b182_report.pdf          my report
```  