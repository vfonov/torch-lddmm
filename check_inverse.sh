#! /bin/bash

fw=$1
bw=$2
o=$3

if [[ -z $o ]];then
echo "Usage $0 fw bw o"
exit 1
fi


tmpdir=$(mktemp -d --tmpdir)
trap "rm -rf $tmpdir" 0 1 2 15

xfmconcat $fw $bw $tmpdir/concat.xfm
xfm_normalize.pl $tmpdir/concat.xfm $tmpdir/concat_norm.xfm --like mni_icbm152_t1_tal_nlin_sym_09c.mnc --exact
grid_proc --mag --float $tmpdir/concat_norm_grid_0.mnc $o

