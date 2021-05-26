# Based on https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt14.sh

# ### download necessary scripts and set paths
echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=10000

# ### get data ready
FIFOLDER=formal-informal
IFFOLDER=informal-formal

prep=data/fairseq_data
tmp=$prep/tmp

# define source and target language names
src=informal
tgt=formal

##### Tokenization #####
mkdir -p $tmp
# basically, we're just tokenizing the dataset; for formality, we don't want to clean input or erase casing and misspelling
for split in "train" "test" "tune"; do
  echo "pre-processing $split data..."
  for l in $tgt $src; do
    cat $prep/$split.$l | \
      perl $TOKENIZER -threads 8 -l en > $tmp/$split.$l
    echo "$tmp/$split.$l file created!"
  done
done

##### BPE #####
# prepare bpe
TRAIN=$tmp/train.informal-formal
BPE_CODE=bpe_code
bpe_out=$prep/bpe_output
mkdir -p $bpe_out
rm -f $TRAIN

# append src and tgt to one file for learning corpus-wide bpe
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

# learn bpe from the combined training file
echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $bpe_out/$BPE_CODE

# apply bpe rules to the splits
for L in $src $tgt; do
    for f in train.$L tune.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $bpe_out/$BPE_CODE < $tmp/$f > $bpe_out/$f
    done
done
