import argparse
from PIL import Image
import numpy as np
from .utils import _str_to_array

metrics = dict(mse=full_ref.mse,
               rmse=full_ref.rmse,
               psnr=full_ref.psnr,
               rmse_sw=full_ref.rmse_sw,
               uqi=full_ref.uqi,
               ssim=full_ref.ssim,
               ergas=full_ref.ergas,
               scc=full_ref.scc,
               rase=full_ref.rase,
               sam=full_ref.sam,
               msssim=full_ref.msssim,
               vifp=full_ref.vifp,
               psnrb=full_ref.psnrb,
               )
                #d_lambda=sewar.no_ref.d_lambda,
                #d_s=sewar.no_ref.d_s,
                #qnr=sewar.no_ref.qnr)

desc = """Description: command-line interface to sewar: image quality package"""
epilog = """You can add any extra argument needed for the function (check documentation for more info).
            Example: passing window size (ws) would be as [-ws 11]
        """

extra_args = dict(ws=int,
                MAX=float,
                K1=float,
                K2=float,
                r=int,
                fltr=_str_to_array,
                weights=_str_to_array,
                sigma_nsq=float)

def parse_args():
    parser = argparse.ArgumentParser(prog='sewar',description=desc,epilog=epilog)
    parser.add_argument('metric',
                    choices=metrics.keys(),
                    help='metric to calculate')
    parser.add_argument('GT', help='first (reference) image path')
    parser.add_argument('P', help='second (query) image path')

    args,unknown = parser.parse_known_args()

    kwargs = dict()
    for i in range(0,len(unknown),2):
        if unknown[i][0] != '-':
            raise Exception('wrong supplied argument "%s". all arguments should start with "-"'%(unknown[i]))
        key =unknown[i][1:]
        val =unknown[i+1]
        kwargs[key] = extra_args[key](val)
    return args, kwargs


def cli(args=None):	
    if args is None:
        args, kwargs = parse_args()

        if args.metric == 'psnrb':
            gt = np.asarray(Image.open(args.GT).convert('YCbCr'))
            p = np.asarray(Image.open(args.P).convert('YCbCr'))
        else:
            gt = np.asarray(Image.open(args.GT))
            p = np.asarray(Image.open(args.P))

        print(args.metric, ":", metrics[args.metric](gt, p, **kwargs))
    else:
        # for testing purposes
        gt = np.asarray(Image.open(args['GT']))
        p = np.asarray(Image.open(args['P']))
        met = args['metric']
        del args['GT']
        del args['P']
        del args['metric']
        return metrics[met](gt, p, **args)


def main():
    cli()
