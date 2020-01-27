import argparse
from model.Unet_nest_5 import nestedUNetUp5 as Unet_nest_5
from model.Unet_nest_5 import nestedUNetUp5_dense as Unet_nest_5_dense
from model.dense_up import dense_up
import torch
from torchsummary import summary
model_dict={'Unet_nest_5': Unet_nest_5(3),
            'dense_up': Unet_nest_5_dense(3),
            'nestedUNetUp5_dense': dense_up(3)
        }


parser = argparse.ArgumentParser(description ='Training set up' )

parser.add_argument('-m','--model',default ='Unet_nest_5', help='model name (default:Unet_nest_5)')

parser.add_argument('-b', '--batch_size', default=110, type=int,metavar='N', help='mini-batch size (default: 110)')

args = parser.parse_args()

model = model_dict[args.model]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
summary(model, input_size=(1, 96, 96),batch_size=args.batch_size)
