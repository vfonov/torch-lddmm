import argparse
import os
import sys


from minc.geo import decompose
import minc.io

import numpy as np

import torch

### local
import torch_lddmm
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Launch pytorch lddmm registration",
        epilog="""
""" )
    
    parser.add_argument("s", help="source image")
    parser.add_argument("t", help="target image")
    parser.add_argument("o", help="output prefix")

    parser.add_argument("--gpu", help="GPU number", type=int, default=0)
    parser.add_argument("--cpu", help="Run on CPU", action="store_true",default=False)
    parser.add_argument("--threads", help="Number of threads", type=int, default=0)

    return  parser.parse_args()


def calculate_jacobian(lddmm,phi0,phi1,phi2):
    # add identity
    phi0_=lddmm.X0+torch.tensor(phi0).to(lddmm.X0.device)
    phi1_=lddmm.X1+torch.tensor(phi1).to(lddmm.X0.device)
    phi2_=lddmm.X2+torch.tensor(phi2).to(lddmm.X0.device)

    # calculate gradients
    phi0_0,phi0_1,phi0_2 = lddmm.torch_gradient(phi0_, lddmm.dx[0], lddmm.dx[1], lddmm.dx[2],  lddmm.grad_divisor_x, lddmm.grad_divisor_y, lddmm.grad_divisor_z)
    phi1_0,phi1_1,phi1_2 = lddmm.torch_gradient(phi1_, lddmm.dx[0], lddmm.dx[1], lddmm.dx[2],  lddmm.grad_divisor_x ,lddmm.grad_divisor_y, lddmm.grad_divisor_z)
    phi2_0,phi2_1,phi2_2 = lddmm.torch_gradient(phi2_, lddmm.dx[0], lddmm.dx[1], lddmm.dx[2],  lddmm.grad_divisor_x, lddmm.grad_divisor_y, lddmm.grad_divisor_z)

    detjac = phi0_0*(phi1_1*phi2_2 - phi1_2*phi2_1)\
           - phi0_1*(phi1_0*phi2_2 - phi1_2*phi2_0)\
           + phi0_2*(phi1_0*phi2_1 - phi1_1*phi2_0)

    return detjac


def main():
    _history=minc.io.format_history(sys.argv)

    args=parse_args()
    src_vol,src_aff = minc.io.load_minc_volume_np(args.s, False, 'float32')
    trg_vol,trg_aff = minc.io.load_minc_volume_np(args.t, False, 'float32')
    _,dx,_ = decompose(src_aff)

    if args.cpu:
        gpu = None
    else:
        gpu = args.gpu

    if args.threads>0:
        torch.set_num_threads(args.threads)

    lddmm = torch_lddmm.LDDMM(template=src_vol,target=trg_vol,do_affine=0,do_lddmm=1,
                              a=10,niter=200,epsilon=1e-3,
                              sigma=1.0,sigmaR=1.0,optimizer='gdr',dx=dx,
                              gpu_number=gpu,
                              verbose=0)

    start_time = time.time()
    lddmm.setParams('a',10)
    lddmm.setParams('niter',400)
    lddmm.setParams('sigma',10)
    lddmm.setParams('sigmaR',10)
    lddmm.setParams('epsilon',1e-2)
    lddmm.run()
    print(f"---  {(time.time() - start_time)} sec ---" )

    start_time = time.time()
    lddmm.setParams('a',5)
    lddmm.setParams('epsilon',1e-3)
    lddmm.setParams('niter', 200)
    lddmm.setParams('sigma', 2)
    lddmm.setParams('sigmaR',2)
    lddmm.run()
    print(f"---  {(time.time() - start_time)} sec ---" )

    start_time = time.time()
    lddmm.setParams('a',2.5)
    lddmm.setParams('niter',  200)
    lddmm.setParams('sigma',  1.2)
    lddmm.setParams('sigmaR', 1.2)
    lddmm.setParams('epsilon',1e-3)
    lddmm.run()
    print(f"---  {(time.time() - start_time)} sec ---" )

    if False: # becomes unstable (?)
        start_time = time.time()
        lddmm.setParams('a',1.5)
        lddmm.setParams('niter',  200)
        lddmm.setParams('sigma',  1.1)
        lddmm.setParams('sigmaR', 1.1)
        lddmm.setParams('epsilon',1e-3)
        lddmm.run()
        print(f"---  {(time.time() - start_time)} sec ---" )


    (phi0,phi1,phi2) = lddmm.computeThisDisplacement() # output resultant displacement field
    (phi0i,phi1i,phi2i) = lddmm.computeInversedDisplacement() # output resultant displacement field

    grid = np.stack([phi0,phi1,phi2],axis=3)
    grid_i = np.stack([phi0i,phi1i,phi2i],axis=3)

    minc.io.save_minc_volume( args.o + '_grid_0.mnc',
        grid, src_aff, ref_fname=args.s, history=_history)
    
    minc.io.save_minc_volume( args.o + '_Inv_grid_0.mnc',
        grid_i, src_aff, ref_fname=args.s, history=_history)

    (deformed_s,_,_,_) = lddmm.applyThisTransform(src_vol)
    minc.io.save_minc_volume( args.o + '_resampled.mnc',
        deformed_s[-1].cpu().numpy(), src_aff, 
        ref_fname=args.s, history=_history)
    
    # calculate the jacobian 
    jac = calculate_jacobian(lddmm, phi0, phi1, phi2)
    minc.io.save_minc_volume( args.o + '_J.mnc',
        jac.cpu().numpy(), src_aff,
        ref_fname=args.s, history=_history)

    # calculate the jacobian 
    inv_jac = calculate_jacobian(lddmm, phi0i, phi1i, phi2i)
    minc.io.save_minc_volume( args.o + '_Inv_J.mnc',
        jac.cpu().numpy(), src_aff,
        ref_fname=args.s, history=_history)


    
# execute script
if __name__ == '__main__':
    main()
