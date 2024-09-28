import torch
import torch.nn as nn
from pycocotools import mask as mask_utils
from tqdm import tqdm
from utility import compute_iou, compute_batch_iou, get_trainable_parameters, get_optimizer_and_scheduler, compute_flattened_iou
import timeit
import copy
from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss
from monai.metrics import compute_iou as monai_compute_iou
import random
import torchvision.transforms as transforms
from torch.nn.functional import threshold, normalize
from ofm import OFM

class COCO_NAS_Trainer:
    #Initialize dataloader, args
    def __init__(self,args):
        for key, value in vars(args).items():
            setattr(self, key, value)
        self.best_smallest_submodel_iou = 0
        self.scheduler = None
        self.target_miou = .75
        
        if self.loss == 'dice':
            #self.seg_loss_func = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
            #seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
            self.seg_loss_func = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
        elif self.loss == 'dicefocal':
            self.seg_loss_func = DiceFocalLoss(lambda_focal=0.95,lambda_dice=0.05,sigmoid=True, squared_pred=True, reduction='mean')
        
        if self.loss == 'mse' or self.trainable == 'e':
            self.seg_loss_func = nn.MSELoss()
        
        self.iou_pred_loss_func = nn.MSELoss()
        self.iou_pred_optimizer, self.iou_pred_scheduler = get_optimizer_and_scheduler(self.supermodel.model.mask_decoder.iou_prediction_head,0.001)
        
    
    def eval(self,model,map=None):
        mIoU = []
        model = model.eval()
        model = nn.DataParallel(model).to(self.device)
        
        for inputs in self.test_dataloader: #enumerate(tqdm(self.test_dataloader, disable=self.no_verbose)): 

            torch.cuda.empty_cache()

            # print(f'inputs["pixel_values"] : {inputs["pixel_values"].shape}')
            # print(f'inputs["input_points"] : {inputs["input_points"].shape}')
            # print(f'inputs["input_boxes"] : {inputs["input_boxes"].shape}')
            # print(f'inputs["ground_truth_masks"] : {inputs["ground_truth_masks"].shape}')

            if len(inputs["input_points"].shape) > 4:
                inputs["input_points"] = inputs["input_points"].squeeze((2))

            with torch.no_grad():
                if self.test_prompt == 'p':
                    outputs = model(pixel_values=inputs["pixel_values"].to(self.device),
                                    input_points=inputs["input_points"].to(self.device),
                                    multimask_output=True)
                elif self.test_prompt == 'b': 
                    outputs = model(pixel_values=inputs["pixel_values"].to(self.device),
                                    input_boxes=inputs["input_boxes"].to(self.device),
                                    multimask_output=True)
            
            outs = outputs.pred_masks
            scores = outputs.iou_scores

            #loop through batch (images)
            for i, one_output in enumerate(outs):
                #loop through objects
                for j, preds in enumerate(one_output):
                    pt, bx = inputs["points"][i][j], inputs["boxes"][i][j]

                    #loop through objects
                    pred_1,pred_2,pred_3 = torch.sigmoid(preds[0]),torch.sigmoid(preds[1]),torch.sigmoid(preds[2])
                    pred_1,pred_2,pred_3 = (pred_1 > 0.5),(pred_2 > 0.5),(pred_3 > 0.5)

                    gt = inputs["ground_truth_masks"][i][j]

                    _, mious = compute_iou([pred_1, pred_2, pred_3],[gt,gt,gt])

                    mIoU.append(max(mious))

                    #Record ground truth with prediction
                    if map != None:
                        img_idx = inputs["img_id"][i]
                        map[f'img-{img_idx}-obj-{j}-pred-0'] = (inputs["image"][i],pt,bx,gt,pred_1,mious[0])
                        map[f'img-{img_idx}-obj-{j}-pred-1'] = (inputs["image"][i],pt,bx,gt,pred_2,mious[1])
                        map[f'img-{img_idx}-obj-{j}-pred-2'] = (inputs["image"][i],pt,bx,gt,pred_3,mious[2])

        model = model.module
        model = model.to('cpu')
        mIoU = torch.tensor(mIoU)
        average_mIoU = torch.mean(mIoU).item()
        return average_mIoU, mIoU, map

    
    def training_step(self, model, inputs):
            
        local_grad = {k: v.cpu() for k, v in model.state_dict().items()}

        model = model.train()
        model = nn.DataParallel(model).to(self.device)

        self.optimizer.zero_grad()

        # print(f'inputs["pixel_values"] : {inputs["pixel_values"].shape}')
        # print(f'inputs["input_points"] : {inputs["input_points"].shape}')
        # print(f'inputs["input_boxes"] : {inputs["input_boxes"].shape}')
        # print(f'inputs["ground_truth_masks"] : {inputs["ground_truth_masks"].shape}')

        if self.test_prompt == 'p':
            outputs = model(pixel_values=inputs["pixel_values"].to(self.device),
                            input_points=inputs["input_points"].to(self.device),
                            multimask_output=False)
        elif self.test_prompt == 'b': 
            outputs = model(pixel_values=inputs["pixel_values"].to(self.device),
                            input_boxes=inputs["input_boxes"].to(self.device),
                            multimask_output=False)

        stk_gt = inputs["ground_truth_masks"] #torch.stack([gt for gt in inputs["ground_truth_masks"]], dim=0)
        stk_out = torch.stack([out for out in outputs.pred_masks], dim=0)
        stk_gt = stk_gt.squeeze(1).to(self.device)
        stk_out = stk_out.squeeze(2).to(self.device)

        # print(f'stk_gt.shape : {stk_gt.shape}')
        # print(f'stk_out.shape : {stk_out.shape}')

        # ious = compute_flattened_iou(stk_gt,stk_out)
        # pred_ious = outputs.iou_scores.squeeze(2)
        # print(f'Actual ious : {ious.shape}')
        # print(f'Pred ious : {pred_ious.shape}')

        loss = self.seg_loss_func(stk_out, stk_gt)
        # iou_loss = self.iou_pred_loss_func(pred_ious,ious)
        # loss = (0.9 * seg_loss) + (0.1 * iou_loss)

        # print(f'loss : {loss}')

        # backward pass (compute gradients of parameters w.r.t. loss)
        loss.backward()
        # loss.sum().backward()
        self.optimizer.step()
        self.scheduler.step()

        model = model.module
        model = model.to('cpu')

        with torch.no_grad():
            for k, v in model.state_dict().items():
                local_grad[k] = local_grad[k] - v.cpu()

        self.supermodel.apply_grad(local_grad,model.config.arch['remove_layer_idx'])

        train_metrics = {
            "train_loss": loss.sum().item(),
            "params": model.config.num_parameters,
        }

        return train_metrics

    
    def single_step(self,submodel,data, model_size, do_test):
        trainable_params = get_trainable_parameters(submodel, self.trainable)
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(trainable_params,self.lr,self.weight_decay, self.scheduler)
        start_train = timeit.default_timer()
        metrics = self.training_step(submodel, data)

        end_train = timeit.default_timer()
        if do_test:
            start_test = timeit.default_timer()
            miou, _, _ = self.eval(submodel)
            end_test = timeit.default_timer()
            #Save the best supermodel in a separate folder
            if model_size == 'Smallest' and miou > self.best_smallest_submodel_iou:
                self.supermodel.save_ckpt(f'{self.log_dir}Best/')
                self.best_smallest_submodel_iou = miou
            metrics['test_miou'] = miou
            self.logger.info(f'\t{model_size} submodel train time : {round(end_train-start_train,4)}, test time {round(end_test-start_test,4)}, metrics {metrics}')
            #Shrink search space further
            # if miou > self.target_miou:
            #     regular_config = {
            #         "atten_out_space": [768], #must be divisbly by num_heads==12
            #         "inter_hidden_space": [3072],
            #         "residual_hidden": [1020],
            #     }
            #     elastic_config = {
            #         "atten_out_space": [768], #Don't go over 768
            #         "inter_hidden_space": [768,1020,1536], #Reduce for minimizing model size [1536,2304]
            #         "residual_hidden": [1020],
            #     }

            #     config = {'0':regular_config, '1':elastic_config, '2':elastic_config,'3':regular_config, '4':regular_config,
            #             '5':elastic_config,'6':elastic_config,'7':regular_config,'8':regular_config,'9':elastic_config,
            #             '10':regular_config,'11':regular_config,
            #             "layer_elastic":{
            #             "elastic_layer_idx":[1,2,5,6,9],
            #             "remove_layer_prob":[0.5,0.5,0.5,0.5,0.5]
            #             }}
            #     self.supermodel = OFM(self.supermodel.model, elastic_config=config)
                
        else:
            self.logger.info(f'\t{model_size} submodel train time : {round(end_train-start_train,4)}, metrics {metrics}')

    
    def train(self):
        #Training loop
        for epoch in range(self.epochs):
            self.logger.info(f'EPOCH {epoch}: starts')

            #Reorder mlp layers per epoch
            if self.reorder == 'per_epoch':
                if self.reorder_method == 'magnitude':
                    self.supermodel.mlp_layer_reordering()
                elif self.reorder_method == 'wanda':
                    self.supermodel.mlp_layer_reordering(self.reorder_dataloader,'wanda')

            start_epoch = timeit.default_timer()
            for idx, inputs in enumerate(tqdm(self.train_dataloader, disable=self.no_verbose)):

                #Reorder mlp layers for every batch 
                if self.reorder == 'per_batch':
                    if self.reorder_method == 'magnitude':
                        self.supermodel.mlp_layer_reordering()
                    elif self.reorder_method == 'wanda':
                        self.supermodel.mlp_layer_reordering(self.reorder_dataloader,'wanda')                    

                #set to True to test the submodels after training step
                do_test = (idx == len(self.train_dataloader) - 1)
                
                #Intervals at which to save the model checkpoint
                save_interval = len(self.train_dataloader) // self.save_interval if len(self.train_dataloader) > self.save_interval else 1
                do_save = ((idx + 1) % save_interval == 0)

                # #Test when saving
                do_test = do_save
                                
                #Train largest submodel (= supernet)
                if 'l' in self.sandwich:
                    submodel, submodel.config.num_parameters, submodel.config.arch = (
                            copy.deepcopy(self.supermodel.model),
                            self.supermodel.total_params,
                            {'remove_layer_idx':[]},
                        )
                    
                    self.single_step(submodel,inputs,'Largest',do_test)

                #Train smallest model
                if 's' in self.sandwich:
                    submodel, submodel.config.num_parameters, submodel.config.arch = self.supermodel.smallest_model() 
                    self.single_step(submodel,inputs,'Smallest',do_test)


                #Train arbitrary medium sized model
                if 'm' in self.sandwich:
                    submodel, submodel.config.num_parameters, submodel.config.arch = self.supermodel.random_resource_aware_model()
                    self.single_step(submodel,inputs,'Medium',do_test)
                    
                # Save the supermodel state dictionary to a file
                if do_save:
                    self.supermodel.save_ckpt(f'{self.log_dir}')
                    self.logger.info(f'\tInterval {(idx + 1) // save_interval}: Model checkpoint saved.')

            end_epoch = timeit.default_timer()

            # Save the supermodel state dictionary to a file
            self.supermodel.save_ckpt(f'{self.log_dir}')
            self.logger.info(f'EPOCH {epoch}: ends {round(end_epoch-start_epoch,4)} seconds. Model checkpoint saved.')   