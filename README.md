# Seq2SeqTranslation

English-German language translation based on **Seq2Seq with Attention** architecture

## Related papers:
  - Sequence to Sequence Learning with Neural Networks: **https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf**
  - Neural Machine Translation by jointly learning to align and translate: **https://arxiv.org/pdf/1409.0473.pdf**
  
## Project structure:
 - ***main.py*** file performs data preprocessing steps using **spacy** tokenizers, **torchtext** library and training loop
 - ***model.py*** includes 3 main components of Seq2Seq architectures written on **PyTorch**: **Encoder**, **Decoder**, **Seq2Seq**
 - **Seq2Seq.ipynb** colab notebook with architectures gathered together and appropriate library configuration

## Dataset
**Multi30k** dataset for english-german translation, cleaned version can be found in data/ folder

## Results
Increased BLEU score to ~0.26 comparing to similar Seq2Seq architecture without attention mechanism, that performed with ~0.21 BLEU score

