from mpi4py import MPI
import itertools, math, pickle
import numpy as np
import pandas as pd

# Get MPI node information
def _get_node_info(verbose = False):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    if verbose:
        print('>> MPI: Name: {} Rank: {} Size: {}'.format(name, rank, size) )
    return int(rank), int(size), comm

# Recive all dataframes containing the result from the different jobs
def _combine_parallel_results(comm, results_, i_job, N_jobs, path, file_name):
    comm.Barrier()
    if i_job == 0:
        all_results_ = [results_]
        # Save current parameter combinations in dictinary
        for i in range(1, N_jobs):
            all_results_.append(comm.recv(source = i,
                                          tag    = 11))
        all_results_ = pd.concat(all_results_, axis = 0)
        try:
            existing_results_ = pd.read_csv(path + file_name)
            all_results_      = pd.DataFrame(np.concatenate([all_results_.to_numpy(), existing_results_.to_numpy()], axis = 0),
                                             columns = all_results_.columns.values)
        except:
            pass
        all_results_.to_csv(path + file_name, index = False)
    else:
        comm.send(results_, dest = 0,
                            tag  = 11)

# List all possible combinations of parameters
def _experiments(lists_):
    return list(itertools.product(*lists_))

# Get the experiments index to run in this job
def _random_experiments_index_batch_job(exps_, i_job, N_jobs):
    np.random.seed(0)
    # Random perdumations in the experiments index list for each batch
    idx_exps_          = list(np.random.permutation(len(exps_)))
    # idx_exps_in_batch_ = list(np.array_split(idx_exps_, N_batches)[i_batch])
    # # Get experiment indexes in Job
    idx_exps_in_job_ = list(np.array_split(idx_exps_, N_jobs)[i_job])
    return idx_exps_in_job_

# Get the experiments index to run in this job
def _experiments_index_batch_job(exps_, i_batch, N_batches, i_job, N_jobs):
    # Get experiment indexes in Job
    idx_exps_          = list(np.linspace(0, len(exps_) - 1, len(exps_), dtype = int))
    # print(idx_exps_)
    # idx_exps_in_batch_ = list(np.array_split(idx_exps_, N_batches)[i_batch])
    # print(idx_exps_in_batch_)
    # print(np.array_split(idx_exps_in_batch_, N_jobs))
    # idx_exps_in_job_   = list(np.array_split(idx_exps_in_batch_, N_jobs)[i_job])
    # print(idx_exps_in_job_)

    return [idx_exps_[i_job]]

# Save in the next row of a .csv file
def _save_val_in_csv_file(data_, meta_, assets_, path, name):
    file_name = r'{}{}-{}'.format(path, ''. join(str(e) for e in assets_), name)
    row_      = meta_ + data_.tolist()
    csv.writer(open(file_name, 'a')).writerow(row_)

# Save in the next row of a .csv file
def _save_test_in_csv_file(data_, meta_, assets_, path, name):
    file_name = r'{}{}-{}'.format(path, ''. join(str(e) for e in assets_), name)
    row_      = meta_ + data_.tolist()
    csv.writer(open(file_name, 'a')).writerow(row_)

# Save in the next row of a .csv file
def _save_baselines_in_csv_file(data_, i_resource, path, name):
    for i_asset in range(data_.shape[1]):
        file_name = r'{}{}{}-{}'.format(path, i_resource, i_asset, name)
        row_ = []
        for i_model in range(data_.shape[2]):
            row_ += data_[i_asset, :, i_model].tolist()
        csv.writer(open(file_name, 'a')).writerow(row_)

# Save in the predictions in a .pkl file
def _save_pred_in_pkl_file(data_, key, i_resource, path, name):
    file_name = r'{}predictions/{}-{}{}'.format(path, i_resource[0], key, name)
    print(file_name)
    with open(file_name, 'wb') as _f:
        pickle.dump(data_, _f, protocol = pickle.HIGHEST_PROTOCOL)

# Flatten a DataFrame combining index and columns names
def _flatten_DataFrame(df_):
    new_df_ = df_.unstack()
    new_df_.index = ["_".join(i) for i in new_df_.index]
    return new_df_.to_frame().T

__all__ = ['_experiments',
           '_get_node_info',
           '_combine_parallel_results',
           '_experiments_index_batch_job',
           '_random_experiments_index_batch_job',
           '_flatten_DataFrame']
