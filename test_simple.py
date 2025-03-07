from __future__ import absolute_import, division, print_function

import os
import sys
import glob
sys.path.append('core')
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from R_MSFM import R_MSFM3,R_MSFM6
import torch
from torchvision import transforms, datasets
import time
import networks
import time
import shutil

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def parse_args():
    parser = argparse.ArgumentParser(description='Simple testing funtion for R-MSFM models.')
    parser.add_argument('--image_path', type=str,help='path to a test image or folder of images', required=True)
    parser.add_argument('--ext', type=str,help='image extension to search for in folder', default="jpeg")
    parser.add_argument('--model_path', type=str,help='path to a models.pth', default="./3M")
    parser.add_argument('--update', type=int,help='iterative update', default=3)
    parser.add_argument("--no_cuda",help='if set, disables CUDA',action='store_true')
    parser.add_argument("--x",help='if set, R-MSFMX',action='store_true')
    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    # assert args.model_name is not None, \
    #     "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # download_model_if_doesnt_exist(args.model_name)
    model_path = args.model_path
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")



    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    if args.x:
        encoder = networks.ResnetEncoder(50, False)
    else:
        encoder = networks.ResnetEncoder(18, False)
    encoder .load_state_dict(torch.load(encoder_path, map_location= device),False)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    if args.update == 3:
        depth_decoder = R_MSFM3(args.x)
    else:
        depth_decoder = R_MSFM6(args.x)
    depth_decoder.load_state_dict(torch.load(depth_decoder_path, map_location= device))
    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory=os.path.join(args.image_path,'output')
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    if os.path.exists(output_directory):

        shutil.rmtree(output_directory)

    os.makedirs(output_directory)
    
    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        min_infer_time = 10
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue
            feed_width = 640
            feed_height = 192
            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            
            #torch.cuda.synchronize()
            start = time.time()
            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)
            #torch.cuda.synchronize()
            end = time.time()
            infer_time = end-start
            

            if infer_time < min_infer_time:
                min_infer_time = infer_time


            if args.update == 3:
                disp = outputs[("disp_up", 2)]
            else:
                disp = outputs[("disp_up", 5)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
            #np.save(name_dest_npy, scaled_disp.cpu().numpy())
            
            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)

            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_im))

    print('min_infer_time:', min_infer_time)
    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
    

'''
python test_simple.py --image_path='/path/to/your/data/' --model_path='/path/to/your/model/' --update=6

'''
