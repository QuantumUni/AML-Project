# In this notebook I am trying to generate data

# Import libraries
import argparse
import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp


# This example contains 2 masses that are connected to three springs
# |--OOOOO--M1--OOOOO--M2--OOOOO--|
#    l1,k1      l2,k2      l3,k3

# Define helper functions
#################################################################################################################

# The shape of data that I am using is [q1, q2, p1, p2]
def get_args_ham():
    return {'m1':1.0 , 
            'm2':2.0 ,
            'k1':1.0 ,
            'k2':4.0 ,
            'k3':2.0 ,
            'l1':1.0 ,
            'l2':1.5 ,
            'l3':3.0 }


def potential_energy(q, args):
    '''U=0.5 * k1 * (q1 - L1)^2 + 0.5 * k2 * (q2 - q1 - L1)^2 + 0.5 * k3 * (q2-L1-L2)^2'''
    # Potential Energy
    
    k=[args['k1'],args['k2'],args['k3']]
    L_list=[args['l1'], args['l2'], args['l3']]

    try:
        if q.shape[1]>1:
            # Convert the list to a numpy array and tile it to have 100 rows

            L= np.tile(L_list, (q.shape[0], 1))
            U = 0.5 * k[0] * (q[:,0].T - L[:,0].T)**2 + 0.5 * k[1] * (q[:,1].T - q[:,0] - L[:,1].T)**2 + 0.5*k[2]*(q[:,1].T-L[:,0].T-L[:,1].T)**2
    except:
        U = 0.5 * k[0] * (q[0] - L_list[0])**2 + 0.5 * k[1] * (q[1] - q[0] - L_list[1])**2 + 0.5*k[2]*(q[1]-L_list[0]-L_list[1])**2 # potential energy
    return U

def kinetic_energy(p,args=get_args_ham()):
    '''T=sum_i 0.5*m_i*v^2'''
    # Kinetic Energy
    m=[args['m1'], args['m2']]
    
    try:

        if p.shape[1]>1:
            T = p[:,0].T**2/(2*m[0]) + p[:,1].T**2/(2*m[1])
    except:
        T = p[0]**2 / (2 * m[0]) + p[1]**2 / (2 * m[1])  # kinetic energy
    return T

def hamiltonian_fn(q, p, args=get_args_ham()):

    T = kinetic_energy(p,args)  # kinetic energy
    V = potential_energy(q,args) # potential energy
    H = T + V
    return H

def dynamics_fn(t, coords, args=get_args_ham()):
    q, p = coords[:2], coords[2:]
    dH_dq = autograd.grad(hamiltonian_fn, 0)
    dH_dp = autograd.grad(hamiltonian_fn, 1)
    dqdt = np.array([dH_dp(q, p)[i] for i in range(2)]) 
    dpdt = -np.array([dH_dq(q, p)[i] for i in range(2)])
    return np.concatenate((dqdt, dpdt))

def integrate_models(x0=np.asarray(([0.1, 0.2, 0.0, 0.0])), t_span=[0,5], t_eval=None, noise_std=0.1):
    # integrate along ground truth vector field
    kwargs = {'t_eval': t_eval, 'rtol': 1e-12}
    true_path = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=x0, **kwargs)
    true_coords = true_path['y']
    
    return true_coords

def get_trajectory(t_span=[0,10], timescale=10, radius=None, y0=None, noise_std=0.0, update_fn=dynamics_fn, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0]))) #also 100 points
    #y0 = np.random.rand(2)*2-1
    
    y0 = np.asarray(y0).flatten()
    spring_ivp = solve_ivp(fun=update_fn, t_span=t_span, y0=y0.flatten(), t_eval=t_eval, rtol=1e-10, **kwargs)
    q1, q2, p1, p2 = spring_ivp['y'][0], spring_ivp['y'][1], spring_ivp['y'][2], spring_ivp['y'][3]
    #print("q1 shape", np.array(q1).shape)

    
    # Change row vector to column vector
    q1, q2 = np.atleast_2d(q1).T, np.atleast_2d(q2).T
    p1, p2 = np.atleast_2d(p1).T, np.atleast_2d(p2).T
    q, p = np.concatenate((q1, q2), axis=1), np.concatenate((p1, p2), axis=1)
    dydt = np.array([dynamics_fn(None, y) for y in spring_ivp['y'].T])
    dqdt, dpdt = np.split(dydt, 2, axis=1)

    settings = locals()
    
    # add noise
    q += np.random.randn(*q.shape)*noise_std
    p += np.random.randn(*p.shape)*noise_std
    return q, p, dqdt, dpdt, t_eval, settings

def get_dataset(seed=0, samples=50, test_split=.5, args=get_args_ham(),**kwargs):
    data = {'meta': locals()}
    L=args['l1'] + args['l2'] + args['l3']
    # randomly sample inputs
    #Do NOT set seed or there will not be random results!
    #np.random.seed(seed)
    coords, dcoords, t_axis = [], [], []
    for s in range(samples):
        #actually randomly sample inputs
        '''q0 = np.random.uniform(0,L,2)
        while q0[0] > q0[1]:
            q0 = np.random.uniform(0,L,2)

        p0 = np.array([0, 0])
        y0 = np.concatenate([q0, p0])'''
        y0 = get_rand_starting_position()
        #print(f"{y0=}")

        q, p, dqdt, dpdt, t, settings = get_trajectory(y0=y0, **kwargs)
        coords.append(np.concatenate((q,p), axis=1))
        dcoords.append(np.concatenate((dqdt, dpdt), axis=1))
        t_axis.append(np.array(t))


    data['coords'] = np.concatenate(coords)
    data['dcoords'] = np.concatenate(dcoords).squeeze()
    #added time column
    data['t_axis'] = t_axis
    # make a train/test split
    split_ix = int(len(data['coords']) * test_split)
    split_data = {}
    for k in ['coords', 'dcoords', 't_axis']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data


def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=15):
    field = {'meta': locals()}

    # meshgrid to get vector field
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    d, c = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    ys = np.stack([d.flatten(), b.flatten(), c.flatten(), a.flatten()])
    
    # get vector directions
    dydt = [dynamics_fn(None, y) for y in ys.T]
    dydt = np.stack(dydt).T

    field['x'] = ys.T
    field['dx'] = dydt.T
    
    return field

def get_rand_starting_position():
    #returns random starting position vector. p's are 0 by default.
    #respects physical condition for q's: q1 <= q2
    args = get_args_ham()
    L=args['l1'] + args['l2'] + args['l3']
    
    q0 = np.random.uniform(0,L,2)
    while q0[0] > q0[1]:
        q0 = np.random.uniform(0,L,2)
    
    #p0 = np.array([0, 0])
    p0 = np.random.uniform(-0.9, 0.9, 2)
    y0 = np.concatenate([q0, p0])
    
    return y0



#################################################################################################################
