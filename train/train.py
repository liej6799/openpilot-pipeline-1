import argparse
import itertools
import time
import numpy as np 
from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.optim as topt
from dataloader import CommaDataset, BatchDataLoader, BackgroundGenerator
from torch.utils.data import DataLoader
from custom_model import *
from onnx2pytorch import ConvertModel
import onnx
import wandb
from timing import *

torch.autograd.set_detect_anomaly(True)

cuda = torch.cuda.is_available()
if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("=> using '{}' for computation.".format(device))

#for reproducibility 
seed = np.random.randint(2**16)
torch.manual_seed(seed)
print("=>seed={}".format(seed))

print("=>intializing CLI args")
# CLI parser 
parser = argparse.ArgumentParser(description='Args for comma supercombo train pipeline')
parser.add_argument("--datatype", type=str, default = "", choices=["dummy", "gen_gt"], help= "directory in which the dummy data or generated gt lies" )
parser.add_argument("--phase", type =str, default="train", choices = ["train", "test"])
parser.add_argument("--batch_size", type= int, default=1, help = "batch size")
parser.add_argument("--modeltype", type = str, default = "scratch", choices= ["scratch", "onnx2torch"], help = "choose type of model for train")
args = parser.parse_args()

class Logger:
    def __init__( self, prefix ):
        self.cur_ep = 0
        self.prefix = prefix
        
    def plotTr( self, loss, lr, time, epoch=-1 ):
        if epoch == -1:
            self.cur_ep += 1
        else: self.cur_ep = epoch
        wandb.log( {"{}_Loss".format( self.prefix ): loss,
                    "{}_Time".format( self.prefix ): time,
                    "{}_lr".format( self.prefix ): lr},
                    step=self.cur_ep )
    

## intializing the object of the logger class 
print("=>intialzing wandb Logger class")

tr_logger = Logger("train")
val_logger = Logger("validation")

print("=>intializing hyperparams")
#Hyperparams
date_it  = "_20dec"
name = "onnx_gen_gt_comma_pipeline_" + date_it
comma_recordings_basedir = "/gpfs/space/projects/Bolt/comma_recordings"
path_npz_dummy = ["inputdata.npz","gtdata.npz"] # dummy data_path
onnx_path = 'supercombo.onnx'
n_workers = 10
lr = (0.001, 2e-4, 1e-3) ## (lr_conv, lr_gru, lr_outhead)
diff_lr = False
recurr_warmup = True
l2_lambda = (1e-4,1e-4,1e-4) 
lrs_factor = 0.75
lrs_patience = 50
lrs_cd = 50
lrs_thresh = 1e-4
lrs_min = 1e-6

epochs = 10
check_val_epoch =2 
batch_size = num_workers = args.batch_size # MUST BE batch_size == num_workers
assert batch_size == num_workers, 'Batch size must be equal to number of workers'
split_per = 0.8
single_frame_batches = False
prefetch_factor = 1
seq_len = 100
prefetch_warmup_time = 2  # seconds wait before starting iterating

#wandb init
# run = wandb.init(project="test-project", entity="openpilot_project", name = name, reinit= True, tags= ["supercombbo pretrain"])

### Load data and split in test and train
print("=>Loading data")
print("=>Preparing the dataloader")

if "onnx" in name:
    
    #train loader
    train_dataset = CommaDataset(comma_recordings_basedir, train_split=split_per, seq_len=seq_len,
                                shuffle=True, single_frame_batches=single_frame_batches, seed=42)
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=num_workers, shuffle=False,
                        prefetch_factor=prefetch_factor, persistent_workers=True, collate_fn=None)
    train_loader = BatchDataLoader(train_loader, batch_size=batch_size)
    train_loader = BackgroundGenerator(train_loader)

    #val_lodaer 
    val_dataset = CommaDataset(comma_recordings_basedir, train_split=split_per, seq_len=seq_len, validation=True,
                                shuffle=True, single_frame_batches=single_frame_batches, seed=42)
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=num_workers, shuffle=False,
                        prefetch_factor=prefetch_factor, persistent_workers=True, collate_fn=None)
    val_loader = BatchDataLoader(val_loader, batch_size=batch_size)
    val_loader = BackgroundGenerator(val_loader)
    
##Load model 
"""
Both the model from scratch and the onnx-pytorch model can be used 

"""
#params for scratch model
inputs_dim_outputheads = {"path": 256, "ll_pred": 32, "llprob": 16, "road_edges": 16,
                          "lead_car": 64, "leadprob": 16, "desire_state": 32, "meta": [64, 32], "pose": 32}
output_dim_outputheads = {"path": 4955, "ll_pred": 132, "llprob": 8, "road_edges": 132,
                          "lead_car": 102, "leadprob": 3, "desire_state": 8, "meta": [48, 32], "pose": 12}
filters_list = [16, 24, 48, 88, 120, 208, 352]
expansion = 6
param_scratch_model = [filters_list, expansion, inputs_dim_outputheads,
                     output_dim_outputheads ]
## only this part of the netwrok is currently trained.
pathplan_layer_names  = ["Gemm_959", "Gemm_981","Gemm_983","Gemm_1036"]

print("=>Loading the model")
print("=>model used:",args.modeltype)

def load_model(params_scratch, pathplan, batch_size):

    if args.modeltype == 'scratch':
        model = CombinedModel(params_scratch[0], params_scratch[1],
                           params_scratch[2], params_scratch[3])
    else :
        onnx_model = onnx.load(onnx_path)
        model = ConvertModel(onnx_model, experimental= True)  #pretrained_model
        
        ##hack to enable batch_size>1 for onnx2pytorch for our case :TODO-- results differ in onnxruntime--keras and onnx2pytorch
        model.Constant_1047.constant = torch.tensor((batch_size,2,66))
        model.Reshape_1048.shape = (batch_size,2,66)
        model.Constant_1049.constant = torch.tensor((batch_size,2,66))
        model.Reshape_1050.shape = (batch_size,2,66)
        model.Constant_1051.constant = torch.tensor((batch_size,2,66))
        model.Reshape_1052.shape = (batch_size,2,66)
        model.Constant_1053.constant = torch.tensor((batch_size,2,66))
        model.Reshape_1054.shape = (batch_size,2,66)
        model.Constant_1057.constant = torch.tensor((batch_size,2,66))
        model.Reshape_1058.shape = (batch_size,2,66)
        model.Constant_1059.constant = torch.tensor((batch_size,2,66))
        model.Reshape_1060.shape = (batch_size,2,66)
        model.Elu_907.inplace = False

        def reinitialise_weights(layer_weight):
            model.layer_weight = torch.nn.Parameter(torch.nn.init.xavier_uniform_(layer_weight))

        for name, layer in model.named_children():
            if isinstance(layer, torch.nn.Linear) and name in pathplan:
                reinitialise_weights(layer.weight)
                layer.bias.data.fill_(0.01)     
    return model 

comma_model = load_model(param_scratch_model, pathplan_layer_names)
comma_model = comma_model.to(device)

# wandb.watch(comma_model) # Log the network weight histograms

#allowing grad only for path_plan
for name, param in comma_model.named_parameters():
    name_layer= name.split(".")
    if name_layer[0] in pathplan_layer_names:
        param.requires_grad = True
    else:
        param.requires_grad = False

## Define optimizer and scheduler
#diff. learning rate for different parts of the network.
if not diff_lr:
    param_group = comma_model.parameters()

    #optimizer for onnx and when diff_lr not True
    if args.modeltype == "onnx2torch":
        optimizer = topt.Adam(param_group,lr[0], weight_decay=l2_lambda[0])
        scheduler = topt.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lrs_factor, patience=lrs_patience, 
                                                 threshold=lrs_thresh, verbose=True, min_lr=lrs_min,
                                                 cooldown=lrs_cd)  
    else :
        pass # needs to be filled 

else:
    conv, gru, outhead = comma_model.getGroupParams()    
    gru = list(itertools.chain.from_iterable(gru))
    conv = list(itertools.chain.from_iterable(conv))
    outhead = list(itertools.chain.from_iterable(outhead))
    param_group = [ { "params": conv },
                    { "params": gru, "lr": lr[1], "weight_decay": l2_lambda[1] },
                    { "params": outhead, "lr": lr[2], "weight_decay": l2_lambda[2]}]
    
    optimizer = topt.Adam(param_group,lr[0], weight_decay=l2_lambda[0])
    #choice of lr_scheduler can be changed
    scheduler = topt.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lrs_factor, patience=lrs_patience, 
                                                 threshold=lrs_thresh, verbose=True, min_lr=lrs_min,
                                                 cooldown=lrs_cd)                               

#Loss functions:

def mean_std(array):
    mean = array[:,0,:,:]
    std = array[:,1,:,:]
    return mean, std

def calcualte_path_loss(mean1, mean2, std1, std2):
    d1 = torch.distributions.normal.Normal(mean1, std1)
    d2 = torch.distributions.normal.Normal(mean2, std2)
    loss = torch.distributions.kl.kl_divergence(d1, d2).sum(dim =2).sum(dim =1).mean(dim =0)
    return loss
    
def path_plan_loss(plan_pred,plan_gt,plan_prob_gt,batch_size):
    
    path_dict = {} ### there are chances i might need to put this also into if loop as it is compliant with dummy and scratch
    path_plans =  plan_pred
    path1, path2, path3, path4, path5 =torch.split(path_plans,991,dim=1)

    path_dict["path_prob"] = []
    path_dict["path1"] = path1[:,:-1].clone().reshape(batch_size,2,33,15)
    path_dict["path2"] = path2[:,:-1].clone().reshape(batch_size,2,33,15)
    path_dict["path3"] = path3[:,:-1].clone().reshape(batch_size,2,33,15)
    path_dict["path4"] = path4[:,:-1].clone().reshape(batch_size,2,33,15)
    path_dict["path5"] = path5[:,:-1].reshape(batch_size,2,33,15)
    path_pred_prob = torch.cat((path1[:,-1].reshape(batch_size,1), path2[:,-1].reshape(batch_size,1),path3[:,-1].reshape(batch_size,1),
                    path4[:,-1].reshape(batch_size,1),path5[:,-1].reshape(batch_size,1)),dim =1).reshape(batch_size,5,1)
    
    #naive path_loss---> train all the paths together
    path1_gt = plan_gt[:,0,:,:,:] 
    path2_gt = plan_gt[:,1,:,:,:]
    path3_gt = plan_gt[:,2,:,:,:]
    path4_gt = plan_gt[:,3,:,:,:]
    path5_gt = plan_gt[:,4,:,:,:]

    mean_pred_path1, std_pred_path1 = mean_std(path_dict["path1"])
    mean_gt_path1 , std_gt_path1 =  mean_std(path1_gt)

    mean_pred_path2, std_pred_path2 = mean_std(path_dict["path2"])
    mean_gt_path2 , std_gt_path2 =  mean_std(path2_gt)

    mean_pred_path3, std_pred_path3 = mean_std(path_dict["path3"])
    mean_gt_path3 , std_gt_path3 =  mean_std(path3_gt)

    mean_pred_path4, std_pred_path4 = mean_std(path_dict["path4"])
    mean_gt_path4 , std_gt_path4 =  mean_std(path4_gt)

    mean_pred_path5, std_pred_path5 = mean_std(path_dict["path5"])
    mean_gt_path5 , std_gt_path5 =  mean_std(path5_gt)

    path1_loss = calcualte_path_loss(mean_pred_path1, mean_gt_path1, std_pred_path1, std_gt_path1)
    path2_loss = calcualte_path_loss(mean_pred_path2, mean_gt_path2, std_pred_path2, std_gt_path2)
    path3_loss = calcualte_path_loss(mean_pred_path3, mean_gt_path3, std_pred_path3, std_gt_path3)
    path4_loss = calcualte_path_loss(mean_pred_path4, mean_gt_path4, std_pred_path4, std_gt_path4)
    path5_loss = calcualte_path_loss(mean_pred_path5, mean_gt_path5, std_pred_path5, std_gt_path5)

    path_pred_prob_d = torch.distributions.bernoulli.Bernoulli(logits = path_pred_prob)
    path_gt_prob_d = torch.distributions.bernoulli.Bernoulli(logits = plan_prob_gt)
    path_prob_loss =  torch.distributions.kl.kl_divergence(path_pred_prob_d, path_gt_prob_d).sum(dim=1).mean(dim=0)

    path_plan_loss = path1_loss + path2_loss + path3_loss + path4_loss + path5_loss + path_prob_loss
    
    return path_plan_loss

"""
Note: other loss functions to be used while training other output heads
"""
# ## lanelines
# lane_pred = lane_pred.reshape(batch_size, 4,2,33,2)
# mean_lane_pred = lane_pred[:,:,0,:,:] 
# mean_lane_gt =  lane_line_gt[:,:,0,:,:]
# std_lane_pred = lane_pred[:,:,1,:,:]
# std_lane_gt = lane_line_gt[:,:,0,:,:]

# lane_pred_d = torch.distributions.normal.Normal(mean_lane_pred, std_lane_pred)
# lane_gt_d = torch.distributions.normal.Normal(mean_lane_gt, std_lane_gt)
# lane_loss = torch.distributions.kl.kl_divergence(lane_pred_d, lane_gt_d).sum(dim=3).sum(dim=2).sum(dim=1).mean(dim=0)

# ## lanelines probability
# lane_prob_pred = lane_prob_pred.reshape(batch_size,4,2)

# lane_prob_pred_d = torch.distributions.categorical.Categorical(logits = lane_prob_pred)
# lane_prob_gt_d = torch.distributions.categorical.Categorical(logits = lane_prob_gt)
# lane_prob_loss = torch.distributions.kl.kl_divergence(lane_prob_pred_d, lane_prob_gt_d).sum(dim=1).mean(dim=0)

# ## Road edges
# road_edges_pred = road_edges_pred.reshape(batch_size,2,2,33,2)

# mean_road_edges_pred = road_edges_pred[:,:,0,:,:]
# mean_road_edges_gt = road_edges_gt[:,:,0,:,:]

# std_road_edges_pred = road_edges_pred[:,:,0,:,:]
# std_road_edges_gt = road_edges_gt[:,:,0,:,:]
# edges_pred_d = torch.distributions.normal.Normal(mean_road_edges_pred, std_road_edges_pred)
# edges_gt_d = torch.distributions.normal.Normal(mean_road_edges_gt, std_road_edges_gt)
# road_edges_loss = torch.distributions.kl.kl_divergence(edges_pred_d, edges_gt_d).sum(dim=3).sum(dim=2).sum(dim=1).mean(dim=0)

# ## Lead car
# leads_pred = leads_pred.reshape(batch_size, 2, 51)
# leads_pred_other = leads_pred[:,:, :48].reshape(batch_size,2,2,6,4)
# leads_pred_prob = leads_pred[:,:, 48:]

# mean_leads_pred_other = leads_pred_other[:,:,0,:,:]
# mean_leads_gt = leads_gt[:,:,0,:,:]

# std_leads_pred_other = leads_pred_other[:,:,1,:,:]
# std_leads_gt =leads_gt[:,:,1,:,:]

# d1 = torch.distributions.normal.Normal(mean_leads_pred_other, std_leads_pred_other)
# d2 = torch.distributions.normal.Normal(mean_leads_gt, std_leads_gt)
# leads_other_loss = torch.distributions.kl.kl_divergence(d1, d2).sum(dim=3).sum(dim=2).sum(dim=1).mean(dim=0)

# d3 = torch.distributions.categorical.Categorical(logits = leads_pred_prob)
# d4 = torch.distributions.categorical.Categorical(logits = leads_prob_gt)
# leads_prob_loss = torch.distributions.kl.kl_divergence(d3, d4).sum(dim = 1).mean(dim =0)

# Leads_loss = leads_other_loss + leads_prob_loss

# # ## Lead Probabilities
# lead_prob_gt = lead_prob_gt[:,0,:]

# lead_prob_pred_d=  torch.distributions.categorical.Categorical(logits = lead_prob_pred)
# lead_prob_gt_d = torch.distributions.categorical.Categorical(logits = lead_prob_gt)
# lead_prob_loss = torch.distributions.kl.kl_divergence(lead_prob_pred_d, lead_prob_gt_d).mean(dim=0)

# ## Desire
# desire_gt = desire_gt[:,0,:]
# desire_pred_d = torch.distributions.categorical.Categorical(logits = desire_pred)
# desire_gt_d = torch.distributions.categorical.Categorical(logits = desire_gt)
# desire_loss = torch.distributions.kl.kl_divergence(desire_pred_d, desire_gt_d).mean(dim=0)

# ## meta1 
# meta_pred_engagement = meta_pred[:,0].reshape(batch_size,1)
# meta_pred_various = meta_pred[:,1:36].reshape(batch_size, 5, 7)
# meta_pred_blinkers = meta_pred[:,36:].reshape(batch_size, 6, 2)
# meta_pred_engagement_d = torch.distributions.bernoulli.Bernoulli(logits = meta_pred_engagement)
# meta_gt_engagement_d = torch.distributions.bernoulli.Bernoulli(logits = meta_eng_gt)
# meta_engagement_loss = torch.distributions.kl.kl_divergence(meta_pred_engagement_d, meta_gt_engagement_d).mean(dim=0)

# meta_pred_various_d = torch.distributions.categorical.Categorical(logits = meta_pred_various)
# meta_gt_various_d = torch.distributions.categorical.Categorical(logits = meta_various_gt)
# meta_various_loss = torch.distributions.kl.kl_divergence(meta_pred_various_d, meta_gt_various_d).sum(dim =1).mean(dim =0)

# meta_pred_blinkers_d = torch.distributions.categorical.Categorical(logits = meta_pred_blinkers)
# meta_gt_blinkers_d = torch.distributions.categorical.Categorical(logits = meta_blinkers_gt)
# meta_blinkers_loss = torch.distributions.kl.kl_divergence(meta_pred_blinkers_d, meta_gt_blinkers_d).sum(dim =1).mean(dim =0)
        
# meta1loss = meta_engagement_loss + meta_various_loss + meta_blinkers_loss

# ## meta desire
# meta_desire_pred = meta_desire_pred.reshape(batch_size,4,8)
# meta_desire_pred_d =  torch.distributions.categorical.Categorical(logits = meta_desire_pred)
# meta_desire_gt_d = torch.distributions.categorical.Categorical(logits = meta_desire_gt)
# meta_desire_loss = torch.distributions.kl.kl_divergence(meta_desirlabels_path_probe_pred_d, meta_desire_gt_d).sum(dim=1).mean(dim =0)

# ##pose
# pose_pred = pose_pred.reshape(batch_size,2,6)        
# mean_pose_pred = pose_pred[:,0,:]
# std_pose_pred = pose_pred[:,1,:]
# mean_pose_gt = pose_gt[:,0,:]
# std_pose_gt = pose_gt[:,1,:]
# pose_pred_dist_obj = torch.distributions.normal.Normal(mean_pose_pred, std_pose_pred)
# pose_gt_dist_obj =  torch.distributions.normal.Normal(mean_pose_gt, std_pose_gt)
# pose_loss = torch.distributions.kl.kl_divergence(pose_pred_dist_obj, pose_gt_dist_obj)
# pose_loss = pose_loss.sum(dim=1).mean(dim=0)

### train loop 

# batchsize in this case is 1 as init recurr, desire and traffic conv reamain same. 
# initializing recurrent state by zeros
recurr_state = torch.zeros(batch_size,512,dtype = torch.float32)
recurr_state = recurr_state.requires_grad_(True).to(device)

# desire and traffic convention is also set to zero
desire = torch.zeros(batch_size,8,dtype = torch.float32, requires_grad= True)
desire = desire.requires_grad_(True).to(device)

#LHD
traffic_convention = torch.zeros(batch_size,2, dtype = torch.float32)
traffic_convention[:,1] =1 
traffic_convention = traffic_convention.requires_grad_(True).to(device)


# with run:
#     wandb.config.lr = lr
#     wandb.config.l2 = l2_lambda
#     wandb.config.lrs = str(scheduler)
#     wandb.config.seed = seed   
for epoch in tqdm(range(epochs)):

    tr_loss = 0.0
    run_loss = 0.0

    for tr_it , batch in tqdm(enumerate(train_loader)):
        if args.datatype == "gen_gt" and args.modeltype == "onnx2torch":
            
            print("Begin training")    
            
            stacked_frames, plans, plans_probs, segment_finished, sequence_finished, bgr_frames = batch
            
            input = stacked_frames.to(device) # -- (batch_size, seq_len, 12, 128,256)
            labels_path = plans.to(device) # -- (batch_size,seq_len,5,2,33,15)
            labels_path_prob = plans_probs.to(device) # -- (batch_size,seq_len,5,1)
            
            optimizer.zero_grad()
            
            batch_loss = torch.zeros(1,dtype = torch.float32, requires_grad = True)

  
            for i in range(seq_len):
            # recurr_state = recurrent_state.clone()
            
                inputs_to_pretained_model = {"input_imgs":input[:,i,:,:,:],
                                            "desire": desire,
                                            "traffic_convention":traffic_convention,
                                            'initial_state': recurr_state
                }
   
                outputs = comma_model(**inputs_to_pretained_model) # -- > [32,6472]

                plan_predictions = outputs[:,:4955].clone() # -- > [32,4955]
                
                recurr = outputs[:,5960:].clone() #-- > [32,512] important to refeed state of GRU
                
                #labels_path_prob[:,i,:,:] -- > [32,5,1]
                # labels_path[:,i,:,:,:,:] --> [32,5,2,33,15]
                
                single_batch_loss = path_plan_loss(plan_predictions, labels_path[:,i,:,:,:,:], labels_path_prob[:,i,:,:], batch_size)
                
                if i == seq_len -1:
                    recurr = recurr
                else:   
                    recurr_state = recurr

                batch_loss += single_batch_loss
            batch_loss  = batch_loss/seq_len # mean of losses over batches of sequences
            
            # recurrent warmup
            if recurr_warmup and epoch == 0 and tr_it>10:
                batch_loss = batch_loss 
            else: 
                batch_loss.backward(retain_graph = True)

            loss_cpu = batch_loss.detach().clone().item() ## loss for one iteration
            
            recurr_state = recurr
            tr_loss += loss_cpu
            run_loss += loss_cpu
            optimizer.step()
            
            if (tr_it+1)%10 == 0:
                print("printing the losses")
                print(f'{epoch+1}/{epochs}, step [{tr_it+1}/{len(train_loader)}], loss: {tr_loss/(tr_it+1):.4f}')
                if (tr_it+1) %100 == 0:
                    # tr_logger.plotTr( run_loss /100, optimizer.param_groups[0]['lr'], time.time() - start_point ) ## add get current learning rate adjusted by the scheduler.
                    scheduler.step(run_loss/100)
                    run_loss =0.0
                
        # # validation loop  
        #     with torch.no_grad(): ## saving memory by not accumulating activations
        #         if (epoch +1) %check_val_epoch ==0:
        #             val_st_pt = time.time()
        #             val_loss_cpu = 0.0
        #             checkpoint_save_path = "./nets/checkpoints/commaitr" + date_it
        #             torch.save(comma_model.state_dict(), checkpoint_save_path + (str(epoch+1) + ".pth" ))    
        #             print(">>>>>validating<<<<<<<")

        #             for val_itr, val_data in enumerate(val_loader):
        #                 val_input,val_labels = val_data

        #                 val_input = val_input.to(device)
        #                 val_label_path = val_labels[0].to(device)
        #                 val_label_path_prob = val_labels[1].to(device)
        #                 val_batch_loss = torch.zeros(1,dtype = torch.float32, requires_grad = True)
                        
        #                 for i in range(batch_size):
        #                     val_inputs_to_pretained_model = {"input_imgs":val_input[i],
        #                                             "desire": desire,
        #                                             "traffic_convention":traffic_convention,
        #                                             "initial_state": recurr_state}
                        
        #                     val_outputs = comma_model(**val_inputs_to_pretained_model)
        #                     val_path_prediction = val_outputs[:,:4955].clone()

        #                     single_val_loss = path_plan_loss(val_path_prediction,val_label_path[i], val_label_path_prob[i], 1)
        #                     val_batch_loss += single_val_loss
                        
        #                 val_batch_loss = val_batch_loss/batch_size
        #                 val_loss_cpu += val_batch_loss.deatch().clone().cpu().item()

        #                 if (val_itr+1)%10 == 0:
        #                     print(f'Epoch:{epoch+1} ,step [{val_itr+1}/{len(val_loader)}], loss: {val_loss_cpu/(val_itr+1):.4f}')

        #             print(f"Epoch: {epoch+1}, Val Loss: {val_loss_cpu/(len(val_loader)):.4f}")
        #             val_logger.plotTr(val_loss_cpu, optimizer.param_groups[0]['lr'], time.time() - val_st_pt)
                        
    print(f"Epoch: {epoch+1}, Train Loss: {tr_loss/len(train_loader)}")
    tr_logger.plotTr(tr_loss/len(train_loader), optimizer.param_groups[0]['lr'], time.time() - start_point )

# PATH = "./nets/model_itr/" +name + ".pth" 
# torch.save(comma_model.state_dict(), PATH)
# print( "Saved trained model" )
