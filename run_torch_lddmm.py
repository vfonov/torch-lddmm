import argparse
import os
import sys


from minc.geo import decompose
import minc.io

import numpy as np


### local
import torch_lddmm

def parse_args():
    parser = argparse.ArgumentParser(description="Launch pytorch lddmm registration",
        epilog="""
""" )
    
    parser.add_argument("s", help="source image")
    parser.add_argument("t", help="target image")
    parser.add_argument("o", help="output prefix")

    return  parser.parse_args()

def main():
    _history=minc.io.format_history(sys.argv)

    args=parse_args()
    src_vol,src_aff = minc.io.load_minc_volume_np(args.s, False, 'float32')
    trg_vol,trg_aff = minc.io.load_minc_volume_np(args.t, False, 'float32')
    _,dx,_ = decompose(src_aff)

    lddmm = torch_lddmm.LDDMM(template=src_vol,target=trg_vol,do_affine=0,do_lddmm=1,
                              a=10,niter=200,epsilon=4e0,
                              sigma=20.0,sigmaR=40.0,optimizer='gdr',dx=dx,
                              verbose=0)
    
    # set 100 iterations instead
    lddmm.setParams('niter',400)
    # run computation
    lddmm.run()

    # now shrink the lddmm kernel size to 7
    lddmm.setParams('a',7)
    lddmm.setParams('niter',400)
    lddmm.run()


    (phi0,phi1,phi2) = lddmm.computeThisDisplacement() # output resultant displacement field

    grid = np.stack([phi0,phi1,phi2],axis=3)

    minc.io.save_minc_volume( args.o + '_grid_0.mnc',
        grid, src_aff, ref_fname=args.s, history=_history)

    #deformed_s = lddmm.outputDeformedTemplate()[0]
    (deformed_s,_,_,_) = lddmm.applyThisTransform(src_vol)

    minc.io.save_minc_volume( args.o + '_resampled.mnc',
        deformed_s[-1].cpu().numpy(), src_aff, 
        ref_fname=args.s, history=_history)
    
# execute script
if __name__ == '__main__':
    main()
