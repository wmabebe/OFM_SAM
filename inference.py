import os
import datetime
from transformers import SamModel, SamConfig, SamProcessor
from ofm import OFM
from torch.utils.data import DataLoader, Subset
from utility import *
from logger import init_logs, get_logger
from dataset import SA1BDataset, MitoDataset
import timeit

def compare_models(model1, model2):
    # Step 1: Check model architectures
    if str(model1) != str(model2):
        print("\tModels have different architectures.")
        return False

    # Step 2: Compare model parameters
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2:
            print("\tParameter names do not match:", name1, name2)
            return False
        if param1.data.ne(param2.data).sum() > 0:
            print("\tParameter values do not match:", name1)
            return False

    print("\tModels have the same architecture and weights.")
    return True

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #torch.cuda.empty_cache()
    args = get_args()

    # Initialize the original SAM model and processor
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    print(args)

    if args.dataset == 'mito':

        # Create an instance of the SAMDataset
        train_dataset = load_dataset("datasets/mitochondria/training.tif", "datasets/mitochondria/training_groundtruth.tif")
        test_dataset = load_dataset("datasets/mitochondria/testing.tif", "datasets/mitochondria/testing_groundtruth.tif")
        train_dataset = MitoDataset(dataset=train_dataset, processor=processor)
        test_dataset = MitoDataset(dataset=test_dataset, processor=processor)

        # Apply subset for shorter training
        if args.train_subset:
            subset_dataset = Subset(train_dataset, indices=range(args.train_subset))
            train_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        
        # Apply subset for shorter testing
        if args.test_subset:
            subset_dataset = Subset(test_dataset, indices=range(args.test_subset))
            test_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        else:
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    
    elif args.dataset == 'sa1b':
        DATA_ROOT = '../SAM-finetuner/datasets/'
        test_dataset = SA1BDataset(f'{DATA_ROOT}SA1B', processor=processor, do_crop=False,label='all_test')
        subset_dataset = Subset(test_dataset, indices=range(10000,10100,1))
        test_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn = sa1b_collate_fn) #collate_fn = sa1b_collate_fn
 

    print(f'testloader : {len(test_dataloader)}')

    layers_to_mask = [[0],[6],[6,1],[1,6,9]]

    for layers in layers_to_mask:

        original_model = SamModel.from_pretrained("facebook/sam-vit-base")
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

        elastic_config = {
            "atten_out_space": [2304], #Don't go over 768
            "inter_hidden_space": [768,1536,3072], #Reduce for minimizing model size
            "residual_hidden_space": [128,512,1020],
        }

        ofm = OFM(original_model, elastic_config=elastic_config)

        #ofm.mask_layers(layers)
        ofm.remove_layers(layers)

        average_mIoU, mIoU, map = eval(ofm.model,test_dataloader,device='cuda',disable_verbose=True,processor=processor, prompt='p')

        print(f'Masking layer {layers} : miou = {average_mIoU}')

    exit()

    # Load the model configuration

    # saved_supermodel = SamModel.from_pretrained('logs/2024-06-30--20:41:13.204558_dataset[sa1b]_trainable[em]_epochs[100]_lr[1e-05]_local_bs[1]/Best')
    # ofm = OFM(saved_supermodel)

    # arc_config = {'layer_1': {'atten_out': 768, 'inter_hidden': 1536, 'residual_hidden': 1536}, 
    #             'layer_2': {'atten_out': 768, 'inter_hidden': 1536, 'residual_hidden': 1536}, 
    #             'layer_3': {'atten_out': 768, 'inter_hidden': 1536, 'residual_hidden': 1536}, 
    #             'layer_4': {'atten_out': 768, 'inter_hidden': 1020, 'residual_hidden': 1020}, 
    #             'layer_5': {'atten_out': 768, 'inter_hidden': 1020, 'residual_hidden': 1536}, 
    #             'layer_6': {'atten_out': 768, 'inter_hidden': 1536, 'residual_hidden': 1020}, 
    #             'layer_7': {'atten_out': 768, 'inter_hidden': 1536, 'residual_hidden': 768}, 
    #             'layer_8': {'atten_out': 768, 'inter_hidden': 1536, 'residual_hidden': 1536}, 
    #             'layer_9': {'atten_out': 768, 'inter_hidden': 1536, 'residual_hidden': 1020}, 
    #             'layer_10': {'atten_out': 768, 'inter_hidden': 1536, 'residual_hidden': 1020}, 
    #             'layer_11': {'atten_out': 768, 'inter_hidden': 1536, 'residual_hidden': 768}, 
    #             'layer_12': {'atten_out': 768, 'inter_hidden': 1536, 'residual_hidden': 1020}}

    

    #miou = test(ofm.model,test_dataloader,disable_verbose=args.no_verbose,processor=processor)
    #print(f'Supermodel from file {count_parameters(saved_supermodel)}: mIoU = {miou}%')

    supernet = ofm.model
    # average_mIoU, mIoU, map = eval(supernet,test_dataloader,device='cuda',disable_verbose=True,processor=processor, prompt='p')
    # print(f'Supernet from file {count_parameters(supernet)} params: mIoU = {average_mIoU}%')

    smallest_submodel, params, config_2 = ofm.smallest_model()

    print(f'smallest size : {count_parameters(smallest_submodel)}')

    print(f'Smallest model\n: {smallest_submodel}')

    


    # average_mIoU, mIoU, map = eval(smallest_submodel,test_dataloader,device='cuda',disable_verbose=True,processor=processor, prompt='p')
    # print(f'Smallest from file {count_parameters(smallest_submodel)} params: mIoU = {average_mIoU}%')

    #for i in range(100):
    random_submodel, params, config = ofm.random_resource_aware_model()

    print(f'Random model\n: {random_submodel}')

    
    #average_mIoU, mIoU, map = eval(random_submodel,test_dataloader,device='cuda',disable_verbose=True,processor=processor, prompt='p')
    #print(f'Random medium from file {count_parameters(random_submodel)} params: mIoU = {average_mIoU}%')

    # best_submodel, total_params = ofm.resource_aware_model(arc_config)



    # average_mIoU, mIoU, map = eval(best_submodel,test_dataloader,device='cuda',disable_verbose=True,processor=processor, prompt='p')
    # print(f'Best medium from file {count_parameters(best_submodel)} params: mIoU = {average_mIoU}%')

    # print('Comparisons')
    # print('1 and 2')
    # compare_models(submodel_1, submodel_2)
    # print('1 and 3')
    # compare_models(submodel_1, submodel_3)
    # print('2 and 3')
    # compare_models(submodel_2, submodel_3)
    # print('1 and medium')
    # compare_models(submodel_1, submodel)

    #Compute embedding distances relative to supernet
    # dist1 = embedding_distance(supernet, smallest_submodel, test_dataloader, device='cuda')
    # dist2 = embedding_distance(supernet, random_submodel, test_dataloader, device='cuda')
    # dist3 = embedding_distance(supernet, best_submodel, test_dataloader, device='cuda')

    # print(f'Average euclidean distance between supernet embeddings and smallest submodel embeddings : {dist1}')
    # print(f'Average euclidean distance between supernet embeddings and random submodel embeddings : {dist2}')
    # print(f'Average euclidean distance between supernet embeddings and best submodel embeddings : {dist3}')

