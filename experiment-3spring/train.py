import torch, argparse
import numpy as np

import sys, os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)



from nn_models import MLP
from hnn import HNN

from data import get_dataset
from utils import L2_loss, rk4

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=2*2, type=int, help='dimensionality of input tensor') # will be 2*2 for q1,q2,p1,p2
    parser.add_argument('--hidden_dim', default=150, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=200, type=int, help='batch_size')
    parser.add_argument('--input_noise', default=0.0, type=float, help='std of noise added to inputs')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=1000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='3spring', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--use_rk4', dest='use_rk4', action='store_true', help='integrate derivative with RK4')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.set_defaults(feature=True)
    return parser.parse_args()


def train(args):

    #get random number generator seeds
    #torch.manual_seed(args.seed) # generate same seeds for same argument in same environment?
    #np.random.seed(args.seed) # get random numbers

    # init model and optimizer
    if args.verbose:
        print("Training baseline model:" if args.baseline else "Training HNN model:")
    
    output_dim=args.input_dim if args.baseline else 2 # will be 2*4 for q1,q2,p1,p2
    nn_model=MLP(args.input_dim,args.hidden_dim, output_dim,args.nonlinearity)
    model=HNN(args.input_dim,differentiable_model=nn_model,field_type=args.field_type,baseline=args.baseline) #make new class hnn model
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4) # updates leargnin rate, otherwise similar to SGD
    
    

    #arrange data into tet data
    data=get_dataset(args.seed, noise_std=args.input_noise)
    x = torch.tensor( data['coords'], requires_grad=True, dtype=torch.float32)
    #print(x.size())
    test_x = torch.tensor( data['test_coords'], requires_grad=True, dtype=torch.float32)
    dxdt = torch.Tensor(data['dcoords'])
    test_dxdt = torch.Tensor(data['test_dcoords'])
    
    # vanilla train loop
    stats = {'train_loss': [], 'test_loss': []} #statistics we need
    for step in range(args.total_steps+1): #how many training steps we wasnt to take
        # train step
        
        ixs = torch.randperm(x.shape[0])[:args.batch_size] # randomly premuted indices of x
        dxdt_hat = model.time_derivative(x[ixs]) # from hnn model, time derivative of randomly chosen points
        dxdt_hat += args.input_noise * torch.randn(*x[ixs].shape) # add noise, maybe
        loss = L2_loss(dxdt[ixs], dxdt_hat) # squared loss of this batch
        loss.backward() #calculate gradients
        grad = torch.cat([p.grad.flatten() for p in model.parameters()]).clone() # concatenates the tensors
        optim.step() ; optim.zero_grad()

        # run test data
        test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
        test_dxdt_hat = model.time_derivative(test_x[test_ixs])
        test_dxdt_hat += args.input_noise * torch.randn(*test_x[test_ixs].shape) # add noise, maybe
        test_loss = L2_loss(test_dxdt[test_ixs], test_dxdt_hat)

        # logging
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())
        if args.verbose and step % args.print_every == 0:
          print("step {}, train_loss {:.4e}, test_loss {:.4e}, grad norm {:.4e}, grad std {:.4e}"
              .format(step, loss.item(), test_loss.item(), grad@grad, grad.std()))

    train_dxdt_hat = model.time_derivative(x)
    train_dist = (dxdt - train_dxdt_hat)**2
    test_dxdt_hat = model.time_derivative(test_x)
    test_dist = (test_dxdt - test_dxdt_hat)**2
    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
    .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
            test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))


    return model, stats
if __name__ == "__main__":
    args = get_args()
    model, stats =train(args)


    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = '-baseline' if args.baseline else '-hnn'
    label = '-rk4' + label if args.use_rk4 else label
    path = '{}/{}{}.tar'.format(args.save_dir, args.name, label)
    torch.save(model.state_dict(), path)



