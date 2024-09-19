import argparse
import os
import sys


from minc.geo import decompose
import minc.io

import numpy as np

import torch

### local
import torch_lddmm

import pandas as pd
import tqdm 

def parse_args():
    parser = argparse.ArgumentParser(description="Launch pytorch lddmm registration",
        epilog="""
""" )
    
    # parser.add_argument("l", help="source list")
    # parser.add_argument("t", help="target image")
    # parser.add_argument("o", help="output lst")

    parser.add_argument("ss", help="subset: 15T 3T")
    parser.add_argument("slice", help="slice number", type=int)
    parser.add_argument("--inv", help="inverse", action="store_true",default=False)


    return  parser.parse_args()


def calculate_jacobian(lddmm,phi0,phi1,phi2):
    # add identity
    phi0_,phi1_,phi2_=lddmm.X0+torch.tensor(phi0).to(lddmm.X0.device),lddmm.X1+torch.tensor(phi1).to(lddmm.X0.device),lddmm.X2+torch.tensor(phi2).to(lddmm.X0.device)

    # calculate gradients
    phi0_0,phi0_1,phi0_2 = lddmm.torch_gradient(phi0_, lddmm.dx[0], lddmm.dx[1], lddmm.dx[2],  lddmm.grad_divisor_x,lddmm.grad_divisor_y,lddmm.grad_divisor_z)
    phi1_0,phi1_1,phi1_2 = lddmm.torch_gradient(phi1_, lddmm.dx[0], lddmm.dx[1], lddmm.dx[2],  lddmm.grad_divisor_x,lddmm.grad_divisor_y,lddmm.grad_divisor_z)
    phi2_0,phi2_1,phi2_2 = lddmm.torch_gradient(phi2_, lddmm.dx[0], lddmm.dx[1], lddmm.dx[2],  lddmm.grad_divisor_x,lddmm.grad_divisor_y,lddmm.grad_divisor_z)

    detjac = phi0_0*(phi1_1*phi2_2 - phi1_2*phi2_1)\
        - phi0_1*(phi1_0*phi2_2 - phi1_2*phi2_0)\
        + phi0_2*(phi1_0*phi2_1 - phi1_1*phi2_0)

    return detjac


def run_lddmm(src, trg, out, out_jac=None,gpu=0):
    src_vol,src_aff = minc.io.load_minc_volume_np(src, False, 'float32')
    trg_vol,trg_aff = minc.io.load_minc_volume_np(trg, False, 'float32')
    _,dx,_ = decompose(src_aff)

    lddmm = torch_lddmm.LDDMM(template=src_vol,target=trg_vol,do_affine=0,do_lddmm=1,
                              a=10,niter=200,epsilon=1e-3,
                              sigma=1.0,sigmaR=1.0,optimizer='gdr',dx=dx,
                              gpu_number=gpu,
                              verbose=0)
    
    lddmm.setParams('a',10)
    lddmm.setParams('niter',400)
    lddmm.setParams('sigma',10)
    lddmm.setParams('sigmaR',10)
    lddmm.setParams('epsilon',1e-2)
    lddmm.run()

    if False:
        lddmm.setParams('a',5)
        lddmm.setParams('epsilon',1e-3)
        lddmm.setParams('niter', 200)
        lddmm.setParams('sigma', 2)
        lddmm.setParams('sigmaR',2)
        lddmm.run()

    if False:
        lddmm.setParams('a',2.5)
        lddmm.setParams('niter',  400)
        lddmm.setParams('sigma',  1.2)
        lddmm.setParams('sigmaR', 1.2)
        lddmm.setParams('epsilon',1e-3)
        lddmm.run()



    (phi0,phi1,phi2) = lddmm.computeThisDisplacement() # output resultant displacement field

    grid = np.stack([phi0,phi1,phi2],axis=3)

    minc.io.save_minc_volume( out,
        grid, src_aff, ref_fname=src)
    
    if out_jac is not None:
        jac = calculate_jacobian(lddmm,phi0,phi1,phi2)
        minc.io.save_minc_volume( out_jac,
            jac.cpu().numpy(), src_aff, 
            ref_fname=src)

def main():
    args=parse_args()

    out_pfx="out_20240408"
    
    model="adni_model_3d_v2/model_t1w_r.mnc"

    df=(pd.read_csv(f"ADNI_{args.ss}_v2.lst", names=['subject','visit','t1w','t2w','pdw','flair'], header=0, dtype=str)
        .assign(fld=args.ss)
        .assign(pfx=lambda x: out_pfx+os.sep+x.fld+os.sep+x.subject+os.sep+x.visit)
        .assign(stx2_t1      =lambda x: x.pfx+os.sep+'stx2/stx2_'+x.subject+'_'+x.visit+'_t1.mnc')

        .assign(lddm_nl      =lambda x: x.pfx+os.sep+'nl/nl_lddm_'+x.subject+'_'+x.visit+'_grid_0.mnc')
        .assign(lddm_nl_inv  =lambda x: x.pfx+os.sep+'nl/nl_lddm_'+x.subject+'_'+x.visit+'_inverse_grid_0.mnc')

        .assign(lddm_j      =lambda x: x.pfx+os.sep+'nl/nl_lddm_'+x.subject+'_'+x.visit+'_grid_j.mnc')
        .assign(lddm_j_inv  =lambda x: x.pfx+os.sep+'nl/nl_lddm_'+x.subject+'_'+x.visit+'_inverse_grid_j.mnc')

        .assign(t1_exists= lambda x: x.agg(lambda y: os.path.exists(str(y.stx2_t1)),axis=1))
        .query("t1_exists")
        )

    print(df.shape)
    N_slices=8

    n=len(df)//N_slices
    d=len(df)-n*N_slices

    df=df.iloc[args.slice*n:(args.slice+1)*n+(d if args.slice==N_slices-1 else 0)]
    print(f"{df.shape=}")

    for i in tqdm.tqdm(range(df.shape[0])):
        row=df.iloc[i]

        if args.inv: # WARNING, the order is actually reversed already , this will calculate mapping from the subject to model
            if not os.path.exists(row.lddm_nl_inv): 
                run_lddmm( model, row.stx2_t1, row.lddm_nl, gpu=args.slice, out_jac=row.lddm_j_inv)
        else:
            if not os.path.exists(row.lddm_nl): # this will calculate mapping from the model to the subject
                run_lddmm(row.stx2_t1, model, row.lddm_nl_inv, gpu=args.slice, out_jac=row.lddm_j)
        #break

    df.to_csv(f"{out_pfx}/{args.ss}/lddm_{args.slice}_{args.inv}.csv",index=False)
    
# execute script
if __name__ == '__main__':
    main()
