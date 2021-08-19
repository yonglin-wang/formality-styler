# Training fasttext and transformer models

## Required Packages

### fasttext

I used [fasttext](https://fasttext.cc) to train a classifier for distinguishing formality. In this project, I only used the CLI version. To install fasttext with CLI, if you have [homebrew](https://docs.brew.sh/Installation), run:

```bash
brew install fasttext
```

**Optional, not necessary in this project**: Alternatively, if you'd like to install and use [their Python API](https://pypi.org/project/fasttext/) (which unfortunately didn't install on my local machine), run the following:

```bash
pip install fasttext
```

### fairseq

I used [fairseq](https://pypi.org/project/fairseq/) to train sequence-to-sequence models for restyling the input sentence to the other level of formality (i.e. formal vs informal). To install fairseq, run:

```bash
pip install fairseq
```

### sacrebleu

I used [sacrebleu 1.5.1](https://pypi.org/project/sacrebleu/1.5.1/) to calculate BLEU scores for the rewriter models. To install sacrebleu, run:

```bash
pip install sacrebleu
```

## Required Datasets

Before you train the models, you'll need the preprocessed datasets. 

### Procedure to obtain datasets

To obtain the pre-processed datasets for this project, do the following:

1. First, if you haven't already, you'll need to obtain permission to [the original GYAFC dataset](https://github.com/raosudha89/GYAFC-corpus) first
2. Then contact me for access to the preprocessed data files
3. Once the access is grated, download the preproseed datasets (```data.zip```) [from this Google Drive link](https://drive.google.com/file/d/1Q9b6dR6bivdB-S-Ot_8Fglk6j_cgEY9v/view?usp=sharing).
4. Extract the file under [training/](./), so that you'll have a new directory: ```training/data```.

### Dataset Structure

After you are done extracting the files, the data directory under [training/](./) should now look like the following:

```
training
├── README.md
├── data
│   ├── fairseq_data
│   │   ├── test.formal
│   │   ├── test.informal
│   │   ├── train.formal
│   │   ├── train.informal
│   │   ├── tune.formal
│   │   └── tune.informal
│   ├── fairseq_eval
│   │   ├── formal_to_informal
│   │   │   ├── example.informal.detok.output (sample output from trained model)
│   │   │   ├── informal.ref0
│   │   │   ├── informal.ref1
│   │   │   ├── informal.ref2
│   │   │   └── informal.ref3
│   │   └── informal_to_formal
│   │       ├── example.formal.detok.output (sample output from trained model)
│   │       ├── formal.ref0
│   │       ├── formal.ref1
│   │       ├── formal.ref2
│   │       └── formal.ref3
│   └── ft_data
│       ├── test
│       ├── train
│       └── tune
└── ...

```

#### How did you derive the data from the original GYAFC corpus? 

For files under ```data/fairseq_data/``` and ```data/fairseq_eval/```: these files are for training the transformer seq2seq models. To obtain these, simply join the files with the same name together across domain. E.g. ```/corpus/dir/Entertainment_Music/test/formal``` and ```/corpus/dir/Family_Relationships/test/formal``` combine into ```data/fairseq_data/test.formal```.

> Crucially, alignment is important: in ```data/fairseq_data/```, formal and informal files for each split should be aligned line-by-line. In ```data/fairseq_eval/```, the .ref* files under each direction should all align, E.g. the n-th line of all .ref* files under ```data/fairseq_eval/informal_to_formal``` should be variants of the same sentence. 

For files under ```data/ft_data/```: also combined across domains. ```train``` is straightforward, but for ```test``` and ```tune```, this project used only the .ref0 files. 

> Alignment is not required for ```data/ft_data/```. In fact, I shuffled the three splits under ```data/ft_data/``` for the potentially better training result. 

Partial code for generating test sets in ```data/fairseq_eval/``` and all code for generating ```data/ft_data/``` can be found at [fasttext_prep.py](./fasttext_prep.py). Code might contain bugs and you'll need to go into the code and change the first two variables to run the code properly. 


## Training Fasttext Classifier

We trained a [fasttext (FT) classifier](https://fasttext.cc/docs/en/supervised-tutorial.html) for distinguishing formality. 

### FT Classifier under the Hood

According to [their paper](https://arxiv.org/pdf/1607.01759.pdf), the FT model trains a bag-of-words embedding model, averages over the input word representation to obtain a text representation for the input sentence, and uses the text representation to train a linear classifier. 

This simple, shallow model architecture, combined with some engineering tricks, allows the training to be extremely fast, taking only ~12 seconds to train the classifier on my non-fancy local machine on CPUs. 

### Commands Used

#### Training

To train a default model and name it as ```defaut_model``` when saving, run:

```bash
fasttext supervised -input data/ft_data/train -output default_model
```

which will use the preprocessed ```data/ft_data/train``` as the training data, and save the model at ```./default_model.bin```. 

You can also modify the structure by adding more options ([link](https://fasttext.cc/docs/en/options.html) to full list of options). For example, the best performing set of options for this project is: 

```
fasttext supervised -input data/ft_data/train -output best_model \
                    -wordNgrams 2 -minn 2 -maxn 3 -epoch 10
```

#### Reducing Model Size

To quantize the saved model (i.e. prune the model), use:

```bash
fasttext quantize -output default_model
```

which will reduce the size of ```default_model.bin``` and save the reduced sized model as ```default_model.ftz```. 

#### Inference

To evaluate the model on dev or test set, run:

```
fasttext test default_model.ftz data/ft_data/tune
```

To get a list of model predictions and save to, e.g., test_out.txt, run:

```
fasttext predict-prob default_model.ftz data/ft_data/test 1 > test_out.txt
```

### Model Selection and Result Analysis

The following table shows the models trained on different setting. For settings that are not listed, we used their default values as described [here](https://fasttext.cc/docs/en/options.html).

|                    | test set accuracy | word n-gram | context window | char. n-gram (min-max) | total epochs | Model size (before Quantization) |
| ------------------ | ----------------- | ----------- | -------------- | ---------------------- | ------------ | -------------------------------- |
| Default Model      | 0.88              | 1           | 5              | 0                      | 5            | 40 MB                            |
| Highest Acc. Model | 0.92              | 2           | 5              | 2-3                    | 10           | 2 GB                             |

However, after preliminary error analysis, we found that the original classifier is "good enough" in that it correctly labels most examples, and the misclassified examples are generally indeed harder even for human readers to judge the formality. 

Therefore, in this project, we chose to favor saving (a significant amount of) space over a slight increase in accuracy. The default model was pruned to a size of 8.5 MB after quantization. 

### Further Reading

- [Comparing supervised and unsupervised models](https://stackoverflow.com/a/57508776/6716783): talks about difference between word n-gram and context window in the word representation models
- [Quick fasttext Tutorial](https://fasttext.cc/docs/en/supervised-tutorial.html) for using FT Text Classication Tools

## Training Transformer-based Rewriter

To train a rewriter, we trained a transformer model, one for each transfer direction (formal to informal and informal to formal). In this way, we are especially treating the task as a NMT task.

Specifically, we will be training ```transformer_iwslt_de_en```, whose sole difference from the original transformer model was [claimed](https://github.com/pytorch/fairseq/issues/1301) to having half of the original parameters in FFN. 

### Procedure

#### 1. Preprocess Translation Data

The preprocessed data obtained in [Required Dataset](#procedure-to-obtain-datasets) is only a joined version of the two topics in the original corpus with no extra filtering, correcting, or other modification, so we'll still need to perform some more preprocessing. 

The preprocessing code is based on [prepare-iwslt14.sh](https://github.com/pytorch/fairseq/blob/8df9e3a4a55bad55078967e97e8a8f31d90ec987/examples/translation/prepare-iwslt14.sh), but crucially, since casing and spelling are both important dimensions of formality, we will not be performing these lowercasing and cleaning on the corpus. 

To perform the preprocessing in this corpus, run: 

```bash
source translation_preprocessing.sh
```

which does the following:

1. download essential packages from GitHub: ```mosesdecoder``` and ```subowrd-nmt```

2. tokenize data in each split and save them to ```./data/fairseq_data/tmp/```

3. learn and apply Byte-Pair Encoding (BPE) to the training data by

   1. concatenating the informal and formal training data in one file (```./data/fairseq_data/tmp/train.informal-formal```)

   2. Learn BPE on the concat file, and generate the code at ```./data/fairseq_data/bpe_output/bpe_code```. See [this awesome tutorial](http://ethen8181.github.io/machine-learning/deep_learning/subword/bpe.html#Implementation-From-Scratch) for how BPE is learned.

   3. Apply BPE code to each of the test, tune, and training file for each formality level. See [the same awesome tutorial](http://ethen8181.github.io/machine-learning/deep_learning/subword/bpe.html#Applying-Encodings) for how the BPE code is applied.

   4. Now you will have the following files under ```./data/fairseq_data/bpe_output/bpe_code```, which has the files you'll need for the following training.

      ```
      bpe_output
      ├── bpe_code
      ├── test.formal
      ├── test.informal
      ├── train.formal
      ├── train.informal
      ├── tune.formal
      └── tune.informal
      ```

#### 2. Train Models using Scripts and Generate Output

For the interest of time, **GPU is required** for training this model. 

(Many things could go wrong here, so it's better to run the commands in each script *step-by-step*, instead of running the entire script all at once.)

- For training a formal to informal model, see [train_formal_informal.sh](./train_formal_informal.sh).

- For training an informal to formal model, see [train_informal_formal.sh](./train_informal_formal.sh).

Both scripts does the following:

1. Binarize the input files using ```fairseq-preprocess```
2. Train a ```transformer_iwslt_de_en``` model using ```fairseq-train```
3. Generate output using ```fairseq-generate```, note that the input order is shuffled (see discussion [here](https://github.com/pytorch/fairseq/issues/2036)), so you'll re-order the output based on their line IDs (the integer after each S, D, H, etc.). 
   1. Alternatively, you can ```cat``` the input file and pipe it to ```fairseq-interactive``` (see discussion/tutorial [here](https://stackoverflow.com/a/65220482/6716783)). It will take MUCH longer, but the order will be preserved. 
4. Lastly, enter interactive mode, where you can type in your input and get output real-time.

#### 3. Evaluate the Model

We evaluate the model with [BLEU](https://en.wikipedia.org/wiki/BLEU) scores using [sacrebleu 1.5.1](https://pypi.org/project/sacrebleu/1.5.1/).

The rough steps to follow are:

1. For each of formal and informal eval files: 

   1. generate output from the model (use the (D)etokenized output, since sacrebleu has its own detokenization rules), 1 sentence per line, same as the input order
   2. name the output files as, e.g., ```formal.detok.output``` and ```informal.detok.output```

2. Put the files under the correct folder (based on direction) under ```./data/fairseq_eval```; then you will have a directory structure like the following (example files such as ```example.informal.detok.output``` are not shown):

   ```
   fairseq_eval
   ├── formal_to_informal
   │   ├── informal.detok.output
   │   ├── informal.ref0
   │   ├── informal.ref1
   │   ├── informal.ref2
   │   └── informal.ref3
   └── informal_to_formal
       ├── formal.detok.output
       ├── formal.ref0
       ├── formal.ref1
       ├── formal.ref2
       └── formal.ref3
   ```

3. Evaluate the output using sacrebleu with, for example, for formal to informal direction: 

   ```
   cd data/fairseq_eval/formal_to_informal/
   sacrebleu --input formal.detok.output formal.ref*
   ```

   If you'd like to ignore casing during evaluation, swap the second command with

   ```
   sacrebleu -lc --input formal.detok.output formal.ref* 
   ```

   where ```-lc``` means to (l)ower (c)ase all texts during evaluation.

### Result Analysis

We compared our model against the original baseline model and a Variational Autoencoder (VAE) model (see [the next section](#training-vae-based-rewriter) for more detail). The table below shows the results. We can see that:

1. The transformer approach performed roughly similarly to the baseline, if not slightly better
2. Ratio: transformer generates longer texts than VAE
3. The VAE approach lowercases all the input and outputs, so we only have lower cased BLEU to compare with transformer
4. Yes, VAE only has BLEU to the ones digit, no typo here! You can view the test set output [on this Google Sheet](https://docs.google.com/spreadsheets/d/1IAvSQ_EjUZaAXhmhohLXbTaJTUXgigtiuWhpsCKGo7E/edit?usp=sharing) (Brandeis account required).

|                    | Transformer (transformer_iwslt_de_en) |           |                | Disentangled-VAE |                | GYAFC Best Baseline |
| ------------------ | :-----------------------------------: | :-------: | :------------: | :--------------: | :------------: | :-----------------: |
|                    |                 BLEU                  | BLEU (lc) | Ratio, ref/gen |    BLEU (lc)     | Ratio, ref/gen |   BLEU (case-NA)    |
| Informal -> Formal |                 68.9                  |   71.1    |     0.998      |       6.7        |     0.834      |       67.67*        |
| Formal -> Informal |                 37.5                  |   43.0    |     1.039      |       4.8        |     0.952      |         NA          |

*Evaluated based on a subset of test set. Casing & tokenization not specified. 

## Training VAE-based Rewriter

In addition to the transformer-based sequence-to-sequence rewriter model, we also trained a model based on a previously proposed, VAE-based, style transfer system orginially used for sentiment transfer ([John et al., 2018](https://arxiv.org/abs/1808.04339)).

The detailed description and discussion of this model can be found in [this repository](https://github.com/yonglin-wang/disentagled-style-rep) for my final project for Information Extraction. 

In summary, this model, if used as-is without any modification to data preprocessing and training, does not prove to be as effective as the transformer model (6.7 BLEU vs. 68.9 BLEU), due to the linguistic differences between sentiment (mostly only choice of word token; you'll usually only need to change a word to change the sentiment) and formality (encompassing word choices, spelling, casing, grammatical structure, etc.).

