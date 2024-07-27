import torch
import torch.nn as nn
from pycocotools import mask as mask_utils
from tqdm import tqdm
from utility import compute_iou, get_trainable_parameters, get_optimizer_and_scheduler, save_preds
import timeit
import copy
from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss
import random


class SA1B_NAS_Trainer:
    #Initialize dataloader, args
    def __init__(self,args):
        for key, value in vars(args).items():
            setattr(self, key, value)
        self.best_smallest_submodel_iou = 0
        self.scheduler = None
        
        if self.loss == 'dice':
            #self.loss_func = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
            #seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
            self.loss_func = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
        elif self.loss == 'dicefocal':
            self.loss_func = DiceFocalLoss(lambda_focal=0.95,lambda_dice=0.05,sigmoid=True, squared_pred=True, reduction='mean')
        
        if self.loss == 'mse' or self.trainable == 'e':
            self.loss_func = nn.MSELoss()
    
    def eval(self,model,map=None):
        mIoU = []
        model.eval()
        model = nn.DataParallel(model).to(self.device)
        for idx,(inputs, images, labels, boxes, points) in enumerate(tqdm(self.test_dataloader, disable=self.no_verbose)): 

            torch.cuda.empty_cache()

            #Set num_objects to 32 or num of points in the data with fewest points
            num_objects = min(self.prompt_batch_size, min([len(pts) for pts in points]))

            # Filter num_objects prompt indices from points and boxes
            input_boxes, input_points = [], []
            for i,d in enumerate(inputs):
                input_boxes.append(d['input_boxes'][:,:num_objects, :])
                input_points.append(d['input_points'][:num_objects, :, :])
                labels[i] = labels[i][:num_objects]
                boxes[i] = boxes[i][:num_objects]
                points[i] = points[i][:num_objects]
            input_boxes = torch.stack(input_boxes)
            input_points = torch.stack(input_points)
                

            data = {'pixel_values': torch.stack([d['pixel_values'] for i,d in enumerate(inputs)]),
                 'original_sizes' : torch.stack([d['original_sizes'] for i,d in enumerate(inputs)]),
                 'reshaped_input_sizes' : torch.stack([d['reshaped_input_sizes'] for i,d in enumerate(inputs)]),
                 'input_boxes' : input_boxes, 
                 'input_points' : input_points }
            
            data["pixel_values"] = data["pixel_values"].squeeze(1)
            data['original_sizes'] = data['original_sizes'].squeeze(1)
            data['reshaped_input_sizes'] = data['reshaped_input_sizes'].squeeze(1)
            if len(data['input_points'].shape) > 4:
                data['input_points'] = data['input_points'].squeeze(1)
            data['input_boxes'] = data['input_boxes'].squeeze(1)
            

            with torch.no_grad():
                if self.test_prompt == 'p':
                    outputs = model(pixel_values=data["pixel_values"].to(self.device),
                                    input_points=data["input_points"].to(self.device),
                                    multimask_output=True)
                elif self.test_prompt == 'b': 
                    outputs = model(pixel_values=data["pixel_values"].to(self.device),
                                    input_boxes=data["input_boxes"].to(self.device),
                                    multimask_output=True)
                
                masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), 
                                                                        data["original_sizes"].cpu(), 
                                                                        data["reshaped_input_sizes"].cpu())
                scores = outputs.iou_scores

                #loop through batch (images)
                for i, mask in enumerate(masks):
                    pts, bxs = points[i], boxes[i]
                    #loop through objects
                    for j,(pred_1, pred_2, pred_3) in enumerate(mask):

                        pred_1,pred_2,pred_3 = torch.sigmoid(pred_1),torch.sigmoid(pred_2),torch.sigmoid(pred_3)
                        pred_1,pred_2,pred_3 = (pred_1 > 0.5),(pred_2 > 0.5),(pred_3 > 0.5)

                        gt = mask_utils.decode(labels[i][j]['segmentation'])
                        gt = torch.from_numpy(gt)
                        _, mious = compute_iou([pred_1, pred_2, pred_3],[gt,gt,gt])

                        mIoU.append(max(mious))

                        #Record ground truth with prediction
                        if map != None:
                            pt, bx = [pts[j]], bxs[j]
                            map[f'img-{i}-obj-{j}-pred-0'] = (images[i],torch.tensor(pt),torch.tensor(bx),gt,pred_1,mious[0])
                            map[f'img-{i}-obj-{j}-pred-1'] = (images[i],torch.tensor(pt),torch.tensor(bx),gt,pred_2,mious[1])
                            map[f'img-{i}-obj-{j}-pred-2'] = (images[i],torch.tensor(pt),torch.tensor(bx),gt,pred_3,mious[2])


        model = model.module
        model = model.to('cpu')
        mIoU = torch.tensor(mIoU)
        average_mIoU = torch.mean(mIoU).item()
        return average_mIoU, mIoU, map
    
    def training_step_encoder(self, model, inputs, gts_embeddings):

        local_grad = {k: v.cpu() for k, v in model.state_dict().items()}

        num_parameters = model.config.num_parameters

        model = model.vision_encoder.to(self.device)

        model = nn.DataParallel(model)

        model.train()
        self.optimizer.zero_grad()

        data = {'pixel_values': torch.stack([d['pixel_values'].squeeze(0) for d in inputs]),
                'original_sizes' : torch.stack([d['original_sizes'].squeeze(0) for d in inputs]),
                'reshaped_input_sizes' : torch.stack([d['reshaped_input_sizes'].squeeze(0) for d in inputs])}

        gts_embeddings = torch.stack(gts_embeddings).to(self.device)

        output = model(data["pixel_values"].to(self.device))
        pred_embeddings = output[0]

        loss = self.loss_func(gts_embeddings, pred_embeddings)

        loss.backward()
        # loss.sum().backward()
        self.optimizer.step()
        self.scheduler.step()

        model = model.module
        model = model.to('cpu')

        with torch.no_grad():
            for k, v in model.state_dict().items():
                local_grad[f'vision_encoder.{k}'] = local_grad[f'vision_encoder.{k}'] - v.cpu()

        self.supermodel.apply_grad(local_grad)

        train_metrics = {
            "train_loss": loss.sum().item(),
            "params": num_parameters,
        }

        return train_metrics

    
    def training_step(self, model, inputs, labels,images, boxes, points):
            
        local_grad = {k: v.cpu() for k, v in model.state_dict().items()}

        model.to(self.device)
        model = nn.DataParallel(model)

        model.train()
        self.optimizer.zero_grad()

        for i, (image, label, pts, bxs) in enumerate(zip(images,labels, points, boxes)):

            torch.cuda.empty_cache()

            #Set num_objects to 64 or num of points in the data with fewest points
            num_objects = min(self.prompt_batch_size, len(points[i]) )

            indices = random.sample(range(len(pts)), num_objects)
            
            label = [labels[i][j] for j in indices]
            bxs = [boxes[i][j] for j in indices]
            pts = [points[i][j] for j in indices]

            inputs = self.processor(image, input_points=pts, input_boxes=[[bxs]], return_tensors="pt")
            
            data = {'pixel_values': torch.stack([inputs['pixel_values'].squeeze(0) for i in range(num_objects)]),
                'original_sizes' : torch.stack([inputs['original_sizes'].squeeze(0) for i in range(num_objects)]),
                'reshaped_input_sizes' : torch.stack([inputs['reshaped_input_sizes'].squeeze(0) for i in range(num_objects)]),
                'input_boxes' : inputs['input_boxes'], 
                'input_points' :  inputs['input_points']}

            outputs = model(pixel_values=data["pixel_values"].to(self.device),
                        input_points=data["input_points"].to(self.device),
                        multimask_output=False)
            
            #post process outputs and retreive masks    
            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks, data["original_sizes"], data["reshaped_input_sizes"], binarize=False
            )

            loss = torch.zeros(1, dtype=torch.float32, device=self.device)

            for j, (gt, mask) in enumerate(zip(label,masks)):
                
                bin_pred = mask.squeeze().to(self.device)
                
                gt = mask_utils.decode(gt['segmentation'])
                bin_gt = torch.from_numpy(gt).to(self.device)

                # #Record ground truth with prediction
                # pt, bx = pts[j], bxs[j]
                # m_detached = m.detach().cpu()
                # m_detached = (m_detached > 0.5).float()
                # iou = compute_iou(m_detached,gt)
                # map[f'img-{i}-obj-{j}-pred-0'] = (images[i],torch.tensor(pt),torch.tensor(bx),gt,m_detached,iou)

                loss += self.loss_func(bin_pred,bin_gt)

                
            #save_preds(map,'junk')    
            
            loss /= len(masks)

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

        self.supermodel.apply_grad(local_grad)

        train_metrics = {
            "train_loss": loss.sum().item(),
            "params": model.config.num_parameters,
        }
        return train_metrics

    
    def single_step(self,submodel,data,gts, model_size, do_test,images, boxes, points):
        trainable_params = get_trainable_parameters(submodel, self.trainable)
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(trainable_params,self.lr,self.weight_decay, self.scheduler)
        start_train = timeit.default_timer()
        if 'e' in self.trainable and 'm' in self.trainable:
            metrics = self.training_step(submodel, data, gts,images, boxes, points)
        elif 'e' in self.trainable:
            metrics = self.training_step_encoder(submodel, data, gts)

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
        else:
            self.logger.info(f'\t{model_size} submodel train time : {round(end_train-start_train,4)}, metrics {metrics}')

    
    def train(self):
        #Training loop
        for epoch in range(self.epochs):
            self.logger.info(f'EPOCH {epoch}: starts')

            #Reorder mlp layers per epoch
            if self.reorder == 'per_epoch':
                self.supermodel.mlp_layer_reordering()

            start_epoch = timeit.default_timer()
            for idx, batch in enumerate(tqdm(self.train_dataloader, disable=self.no_verbose)):

                #Reorder mlp layers for every batch 
                if self.reorder == 'per_batch':
                    self.supermodel.mlp_layer_reordering() 
                
                if 'e' in self.trainable and 'm' in self.trainable:
                    inputs, images, labels, boxes, points = batch
                elif 'e' in self.trainable:
                    inputs, labels = batch
                    images, boxes, points = None, None, None

                #set to True to test the submodels after training step
                do_test = (idx == len(self.train_dataloader) - 1)
                
                #Intervals at which to save the model checkpoint
                save_interval = len(self.train_dataloader) // self.save_interval if len(self.train_dataloader) > self.save_interval else 1
                do_save = ((idx + 1) % save_interval == 0)

                # #Test when saving
                # do_test = do_save
                                
                #Train largest submodel (= supernet)
                if 'l' in self.sandwich:
                    submodel, submodel.config.num_parameters, submodel.config.arch = (
                            copy.deepcopy(self.supermodel.model),
                            self.supermodel.total_params,
                            {},
                        )
                    
                    self.single_step(submodel,inputs,labels,'Largest',do_test,images, boxes, points)

                #Train smallest model
                if 's' in self.sandwich:
                    submodel, submodel.config.num_parameters, submodel.config.arch = self.supermodel.smallest_model() 
                    self.single_step(submodel,inputs,labels,'Smallest',do_test,images, boxes, points)


                #Train arbitrary medium sized model
                if 'm' in self.sandwich:
                    submodel, submodel.config.num_parameters, submodel.config.arch = self.supermodel.random_resource_aware_model()
                    self.single_step(submodel,inputs,labels,'Medium',do_test,images, boxes, points)
                    
                # Save the supermodel state dictionary to a file
                if do_save:
                    self.supermodel.save_ckpt(f'{self.log_dir}')
                    self.logger.info(f'\tInterval {(idx + 1) // save_interval}: Model checkpoint saved.')

            end_epoch = timeit.default_timer()

            # Save the supermodel state dictionary to a file
            self.supermodel.save_ckpt(f'{self.log_dir}')
            self.logger.info(f'EPOCH {epoch}: ends {round(end_epoch-start_epoch,4)} seconds. Model checkpoint saved.')   