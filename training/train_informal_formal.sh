# Based on https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt14.sh

# ### Record input folders
prep=data/fairseq_data
# directory to BPE output
bpe_out=$prep/bpe_output
BPE_CODE=bpe_code

# ### Prepare formal-informal output folders
IFFOLDER=informal-formal
# folder to save preprocessed binarized data in
BINDIR=data-bin
ifbin=$BINDIR/$IFFOLDER
# folder to save training results in
RESULTS=data/fairseq_results
ifoutput=$RESULTS/$IFFOLDER

# ### Prepare informal-formal
mkdir -p $ifbin
fairseq-preprocess --source-lang informal --target-lang formal \
    --trainpref $bpe_out/train --validpref $bpe_out/tune --testpref $bpe_out/test \
    --destdir $ifbin \
    --workers 20

# ### Training informal-formal
mkdir -p $ifoutput
echo "saving training output to ${ifoutput}..."
CUDA_VISIBLE_DEVICES=0 fairseq-train \
   $ifbin \
   --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
   --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
   --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
   --dropout 0.3 --weight-decay 0.0001 \
   --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
   --max-tokens 4096 \
   --eval-bleu \
   --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
   --eval-bleu-detok moses \
   --eval-bleu-remove-bpe \
   --eval-bleu-print-samples \
   --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
   --patience 10 \
   --save-dir $ifoutput/checkpoints \
   --log-format json --log-interval 10 2>&1 | tee $ifoutput/train.log

# inference
fairseq-generate $ifbin \
    --path $ifoutput/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    --bpe subword_nmt --bpe-codes $bpe_out/$BPE_CODE --tokenizer moses --moses-target-lang en \
    --results-path $ifoutput/generate_output

# open interactive mode with trained model, where user types in sentence to translate. 
# Alternatively, you can pipe input to this command, e.g. cat test.txt | fairseq-interactive ...
fairseq-interactive $ifbin \
    --path $ifoutput/checkpoints/checkpoint_best.pt \
    --beam 5 --source-lang informal --target-lang formal \
    --bpe subword_nmt --bpe-codes $bpe_out/$BPE_CODE --tokenizer moses --moses-target-lang en \
    --remove-bpe
    