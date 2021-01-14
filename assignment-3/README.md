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
