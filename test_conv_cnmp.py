# %%
from model.conv_cnmp import ConvCNMP
import torch

torch.set_float32_matmul_precision('high')

def get_free_gpu():
    gpu_util = []
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)  # Switch GPU
#        gpu_util.append((i, torch.cuda.memory_stats()['reserved_bytes.all.current'] / (1024 ** 2)))
        gpu_util.append((i, torch.cuda.utilization()))
    gpu_util.sort(key=lambda x: x[1])
    return gpu_util[0][0]

if torch.cuda.is_available():
    available_gpu = get_free_gpu()
    if available_gpu == 0:
        device = torch.device("cuda:0")
    else:
        device = torch.device(f"cuda:{available_gpu}")
else:
    device = torch.device("cpu")

print("Device :", device)

# %%
timestamp = '1710407937'
data_root = f'data/synthetic/processed/{timestamp}'

train_data = torch.load(f'{data_root}/train.pt')
val_data = torch.load(f'{data_root}/val.pt')

# %%
batch_size = 64
dx, dy = 1, 1
dc, dw, dh = 3, 32, 32  # image size
t_steps = 200
num_train, num_val = len(train_data), len(val_data)
n_max, m_max = 10, 10

# %%
# # Pick 2 random nums: 1<=n<=n_max, 1<=m<=m_max
# # Define 4 tensors: envs (batch_size, dc, dw, dh), obs (batch_size, n, dx+dy), tar_x (batch_size, m, dx), tar_y (batch_size, m, dy)
# # For each traj_id:
# #   First, keep t[traj_id]['image'] in envs[traj_id].
# #   Then pick n random points from t[traj_id]['x'] and t[traj_id]['y'] and concat them to form a tensor of shape (1, n, 2). Store them in obs[traj_id].
# #   Then pick m random ids in [0, t_steps-1]. Pick corresponding x from t[traj_id]['x'] and store them in tar_x[traj_id]. Similarly, store corresponding y 
# #       from t[traj_id]['y'] in tar_x[traj_id].
# # Return 4 tensors
# def get_batch(t: list, traj_ids: list, val=False):  # t can be either train_data or val_data
#     n = torch.randint(1, n_max, (1,)).item()
#     m = torch.randint(1, m_max, (1,)).item() if not val else t_steps
#     # n=m=3

#     envs = torch.zeros((batch_size, dc, dw, dh), dtype=torch.float32, device=device)
#     obs = torch.zeros((batch_size, n, dx+dy), dtype=torch.float32, device=device)
#     tar_x = torch.zeros((batch_size, m, dx), dtype=torch.float32, device=device)
#     tar_y = torch.zeros((batch_size, m, dy), dtype=torch.float32, device=device)

#     for i, traj_id in enumerate(traj_ids):
#         traj = t[traj_id]
#         envs[i] = traj['env']

#         permuted_ids = torch.randperm(t_steps)
#         n_ids = permuted_ids[:n]
#         m_ids = permuted_ids[n:n+m] if not val else permuted_ids
        
#         obs[i, :n, :dx] = traj['x'][n_ids]
#         obs[i, :n, dx:] = traj['y'][n_ids]
        
#         tar_x[i] = traj['x'][m_ids]
#         tar_y[i] = traj['y'][m_ids]

#     return envs, obs, tar_x, tar_y

envs = torch.zeros((batch_size, dc, dw, dh), dtype=torch.float32, device=device)
obs = torch.zeros((batch_size, n_max, dx+dy), dtype=torch.float32, device=device)
tar_x = torch.zeros((batch_size, m_max, dx), dtype=torch.float32, device=device)
tar_y = torch.zeros((batch_size, m_max, dy), dtype=torch.float32, device=device)
tar_mask = torch.zeros((batch_size, m_max), dtype=torch.bool, device=device)

def prepare_masked_batch(t: list, traj_ids: list):
    # envs is completely overwritten but others are partially, so we need to zero out old values
    obs.fill_(0)
    tar_x.fill_(0)
    tar_y.fill_(0)
    tar_mask.fill_(False)

    for i, traj_id in enumerate(traj_ids):
        traj = t[traj_id]
        envs[i] = traj['env']

        n = torch.randint(1, n_max, (1,)).item()
        m = torch.randint(1, m_max, (1,)).item()

        permuted_ids = torch.randperm(t_steps)
        n_ids = permuted_ids[:n]
        m_ids = permuted_ids[n:n+m]
        
        obs[i, :n, :dx] = traj['x'][n_ids]
        obs[i, :n, dx:] = traj['y'][n_ids]
        
        tar_x[i, :m] = traj['x'][m_ids]
        tar_y[i, :m] = traj['y'][m_ids]
        tar_mask[i, :m] = True


val_envs = torch.zeros((batch_size, dc, dw, dh), dtype=torch.float32, device=device)
val_obs = torch.zeros((batch_size, n_max, dx+dy), dtype=torch.float32, device=device)
val_tar_x = torch.zeros((batch_size, t_steps, dx), dtype=torch.float32, device=device)
val_tar_y = torch.zeros((batch_size, t_steps, dy), dtype=torch.float32, device=device)

def prepare_masked_val_batch(t: list, traj_ids: list):
    val_obs.fill_(0)
    val_tar_x.fill_(0)
    val_tar_y.fill_(0)

    for i, traj_id in enumerate(traj_ids):
        traj = t[traj_id]
        val_envs[i] = traj['env']

        n = torch.randint(1, n_max, (1,)).item()

        permuted_ids = torch.randperm(t_steps)
        n_ids = permuted_ids[:n]
        m_ids = torch.arange(t_steps)
        
        val_obs[i, :n, :dx] = traj['x'][n_ids]
        val_obs[i, :n, dx:] = traj['y'][n_ids]
        
        val_tar_x[i] = traj['x'][m_ids]
        val_tar_y[i] = traj['y'][m_ids]

# %%
model_ = ConvCNMP(linear_output_sizes=[256]).to(device)
optimizer = torch.optim.Adam(lr=1e-4, params=model_.parameters())

if torch.__version__ >= "2.0":
    model = torch.compile(model_)

print(model_)

# %%
import time
import os
timestamp = int(time.time())
root_folder = f'output/synthetic/{str(timestamp)}/'

if not os.path.exists(root_folder):
    os.makedirs(root_folder)

if not os.path.exists(f'{root_folder}saved_model/'):
    os.makedirs(f'{root_folder}saved_model/')

# if not os.path.exists(f'{root_folder}img/'):
#     os.makedirs(f'{root_folder}img/')

# torch.save(y, f'{root_folder}y.pt')


epochs = 5_000_000
epoch_iter = num_train//batch_size  # number of batches per epoch (e.g. 100//32 = 3)
v_epoch_iter = num_val//batch_size  # number of batches per validation (e.g. 100//32 = 3)

val_per_epoch = 1000  # validation frequency
min_val_loss = 1_000_000

mse_loss = torch.nn.MSELoss()

training_loss, validation_error = [], []
avg_loss = 0

tr_loss_path = f'{root_folder}training_loss.pt'
val_err_path = f'{root_folder}validation_error.pt'

for epoch in range(epochs):
    epoch_loss = 0

    traj_ids = torch.randperm(num_train)[:batch_size*epoch_iter].chunk(epoch_iter)  # [:batch_size*epoch_iter] because nof_trajectories may be indivisible by batch_size

    for i in range(epoch_iter):
        optimizer.zero_grad()
        prepare_masked_batch(train_data, traj_ids[i])
        pred = model(envs, obs, tar_x)
        loss = model.loss(pred, tar_y, tar_mask)  # mean loss over the batch
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= epoch_iter  # mean loss over the epoch
    
    training_loss.append(epoch_loss)

    if epoch % val_per_epoch == 0:
        with torch.no_grad():
            v_traj_ids = torch.randperm(num_val)[:batch_size*v_epoch_iter].chunk(v_epoch_iter)
            val_loss = 0

            for j in range(v_epoch_iter):
                prepare_masked_val_batch(val_data, v_traj_ids[j])

                p = model(val_envs, val_obs, val_tar_x)
                val_loss += mse_loss(p[:, :, :dy], val_tar_y).item()

            validation_error.append(val_loss)
            if val_loss < min_val_loss and epoch > 1e3:
                min_val_loss = val_loss
                print(f'New best: {min_val_loss}')
                torch.save(model.state_dict(), f'{root_folder}saved_model/on_synth.pt')

    avg_loss += epoch_loss

    if epoch % val_per_epoch == 0:
        print("Epoch: {}, Loss: {}".format(epoch, avg_loss/val_per_epoch))
        avg_loss = 0

torch.save(torch.Tensor(training_loss), tr_loss_path)
torch.save(torch.Tensor(validation_error), val_err_path)

# %%



