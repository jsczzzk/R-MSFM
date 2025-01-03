# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import sys
sys.path.append('core')
import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import json
from utils import *
from kitti_utils import *
from layers import *
from R_MSFM import R_MSFM3,R_MSFM6
import datasets
import networks

accu = 2
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        self.scaler = GradScaler(enabled=False)
        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        self.models = {}
        self.parameters_to_train = []
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.num_input_frames = len(self.opt.frame_ids)
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, True)
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        if self.opt.iters = 6:
            self.models["depth"] = R_MSFM6()
        else:
            self.models["depth"] = R_MSFM3()
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())


        if self.use_pose_net:
            self.models["pose_encoder"] = networks.ResnetEncoder2(18,True,2)
            self.models["pose_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())

            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)
            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())
  
        self.optimizer =optim.AdamW(self.parameters_to_train, lr=self.opt.learning_rate, 
                                        weight_decay=self.opt.wdecay)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,self.opt.learning_rate, epochs=40, div_factor= 25 ,
            steps_per_epoch=len(train_dataset)//self.opt.batch_size, pct_start= 0.1, cycle_momentum=False)
        
        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)

        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        scale = 0
        h = self.opt.height // (2 ** scale)
        w = self.opt.width // (2 ** scale)

        self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
        self.backproject_depth[scale].cuda()

        self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
        self.project_3d[scale].cuda()

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()


    def run_epoch(self):
        """Run a single epoch of training and validation
        # """
        # self.scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)
            losses["loss"] = losses["loss"]/accu
            losses["loss"].backward()
            if self.step%accu==0:
                torch.nn.utils.clip_grad_norm_(self.parameters_to_train, self.opt.clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.scheduler.step()
            Lr = self.scheduler.get_lr()[0] 

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 6000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses, Lr)
                # self.val()
            if self.epoch>0 and (self.step %1500) ==0 :
                self.save_model()
            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        
        
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features, iters = self.opt.iters)
        if self.opt.gc:
            aa = self.models["encoder"](inputs["color_aug", -1, 0])
            bb = self.models["depth"](aa, iters = self.opt.iters)
            outputs_1 = {}
            outputs_1[("disp_up", -1, 0)] = bb[("disp_up", 0)]
            outputs_1[("disp_up", -1, 1)] = bb[("disp_up", 1)]
            outputs_1[("disp_up", -1, 2)] = bb[("disp_up", 2)]
            outputs_1[("disp_up", -1, 3)] = bb[("disp_up", 3)]
            outputs_1[("disp_up", -1, 4)] = bb[("disp_up", 4)]
            outputs_1[("disp_up", -1, 5)] = bb[("disp_up", 5)]

            aa = self.models["encoder"](inputs["color_aug", 1, 0])
            bb = self.models["depth"](aa, iters = self.opt.iters)
            outputs1 = {}
            outputs1[("disp_up", 1, 0)] = bb[("disp_up", 0)]
            outputs1[("disp_up", 1, 1)] = bb[("disp_up", 1)]
            outputs1[("disp_up", 1, 2)] = bb[("disp_up", 2)]
            outputs1[("disp_up", 1, 3)] = bb[("disp_up", 3)]
            outputs1[("disp_up", 1, 4)] = bb[("disp_up", 4)]
            outputs1[("disp_up", 1, 5)] = bb[("disp_up", 5)]
            outputs1.update(outputs_1)
            outputs.update(outputs1)


        if self.use_pose_net:
            outputs.update(self.predict_poses2(inputs))


        self.generate_images_pred(inputs, outputs, iters = self.opt.iters)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses


    def predict_poses2(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

        for f_i in self.opt.frame_ids[1:]:
            if f_i != "s":
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                axisangle, translation = self.models["pose"](pose_inputs)
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs, iters=3):

        for scale in range(iters):

            disp = outputs[("disp_up", scale)]

            source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords, computed_depth = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)
                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")
                if self.opt.gc:
                    outputs[("computed_depth", frame_id, scale)] = computed_depth
                    _, outputs[("projected_depth", frame_id, scale)] = disp_to_depth(F.grid_sample(
                        outputs[("disp_up", frame_id, scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border"), self.opt.min_depth, self.opt.max_depth)

                outputs[("color_identity", frame_id, scale)] = \
                    inputs[("color", frame_id, source_scale)]

    def smooth_l1_loss_ours(self, input,target,beta=0.15):
        n = torch.abs(input-target)
        cond = n<beta
        loss = torch.where(cond, 0.5*n**2, n)
        return loss
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        sq_diff = torch.square(target - pred)/2
        l2_loss = sq_diff.mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l2_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in range(self.opt.iters):
            loss = 0
            reprojection_losses = []
            i_weight = 0.9**(self.opt.iters - scale - 1)
            if self.opt.gc:
                diff_depth_1 = ((outputs[("computed_depth", -1, scale)] - outputs[("projected_depth", -1, scale)]).abs() / (outputs[("computed_depth", -1, scale)] + outputs[("projected_depth", -1, scale)]).abs())
                diff_depth1 = ((outputs[("computed_depth", 1, scale)] - outputs[("projected_depth", 1, scale)]).abs() / (outputs[("computed_depth", 1, scale)] + outputs[("projected_depth", 1, scale)]).abs())
            
            source_scale = 0

            disp = outputs[("disp_up", scale)]
            color = inputs[("color", 0, source_scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if 1:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if 0:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if 0:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            
            identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                

            identity_reprojection_loss1, idxs = torch.min(identity_reprojection_loss, dim=1)
            identity_reprojection_loss = torch.stack((identity_reprojection_loss1, identity_reprojection_loss1), dim=1)
            
            sta = reprojection_loss<identity_reprojection_loss
            
            beta = 50

            reprojection_loss[sta==0] =100

            to_optimise = torch.where((reprojection_loss[:,0]<=beta) + (reprojection_loss[:,1]<=beta),
                            torch.where(
                                        
                                        reprojection_loss[:,0] < reprojection_loss[:,1],
                                        reprojection_loss[:,0],  
                                      reprojection_loss[:,1]
                                       ),
                            torch.zeros_like(reprojection_loss[:,0])
                            )
            
            loss += i_weight*to_optimise.sum()/(to_optimise != 0).sum()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            # loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            loss += self.opt.disparity_smoothness * smooth_loss * i_weight

            if self.opt.gc:
                diff_loss = torch.where((reprojection_loss[:,0]<=beta) + (reprojection_loss[:,1]<=beta),
                            torch.where(
                                        
                                        reprojection_loss[:,0] < reprojection_loss[:,1],
                                       diff_depth_1.squeeze(),  
                                     diff_depth1.squeeze()
                                       ),
                            torch.zeros_like(reprojection_loss[:,0])
                            )


                diff_loss = diff_loss.sum()/(diff_loss != 0).sum() * i_weight *0.1
                loss += diff_loss

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))


    def log(self, mode, inputs, outputs, losses, Lr):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
        writer.add_scalar('learningRate', Lr, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            s = 0
            writer.add_image(
                "color_{}_{}/{}".format(s, s, j),
                inputs[("color", s, s)][j].data, self.step)

            writer.add_image(
                "disp_{}/{}".format(s, j),
                normalize_image(outputs[("disp_up", s)][j]), self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}_{}".format(self.epoch, self.step))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location='cpu')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict, strict = True)

