#! /bin/bash

s=$1
t=$2
o=$3

if [[ -z $o ]];then
    echo "Usage $0 <source> <target> <output>"
    exit 1
fi

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${PRL:-8}

antsRegistration --float 1 --minc 1 \
    --collapse-output-transforms 1 \
    --dimensionality 3 \
    --initialize-transforms-per-stage 0 --write-composite-transform 1 \
    --interpolation Linear \
    --output "[${o}]" \
    --transform SyN[ 0.1, 3.0, 0.5 ] \
    --metric CC[  $s , $t, 1, 4 ] \
    --convergence [ 200x150x100x50, 1e-07, 20 ] \
    --smoothing-sigmas 3.0x2.0x1.0x0.0vox --shrink-factors 8x4x2x1 --use-histogram-matching 1 \
    --winsorize-image-intensities [ 0.05, 0.95 ]  \
    --write-composite-transform 0 

