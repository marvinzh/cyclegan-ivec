#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e
trials=data/sre10_test/trials
num_components=2048 # Larger than this doesn't make much of a difference.


sid/new_spkivec.sh --cmd "$train_cmd --mem 8G" --nj 40 data/sre10_train exp/C5/adapted_train

local/scoring_common_1.sh data/swbd_train data/sre10_train data/sre10_test exp/ivectors_swbd_train exp/C5/adapted_train exp/C5/adapted_test

local/plda_scoring.sh data/swbd_train data/sre10_train data/sre10_test exp/ivectors_swbd_train exp/C5/adapted_train exp/C5/adapted_test $trials exp/scores_gmm_2048_ind_pooled


echo "GMM-$num_components EER and mini-DCF for C5 ext condition"
eer=`compute-eer <(python3 local/prepare_for_eer.py $trials exp/scores_gmm_${num_components}_ind_pooled/plda_scores) 2> /dev/null`
mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_gmm_2048_ind_pooled/plda_scores $trials 2> /dev/null`
mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_gmm_2048_ind_pooled/plda_scores $trials 2> /dev/null`

echo "score: $eer $mindcf1 $mindcf2"
echo "EER: $eer"
echo "minDCF(p-target=0.01): $mindcf1"
echo "minDCF(p-target=0.001): $mindcf2"
