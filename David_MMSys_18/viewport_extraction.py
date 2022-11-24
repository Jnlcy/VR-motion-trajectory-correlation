


import Read_Dataset as david
david.create_and_store_sampled_dataset()




'''def load_dataset_xyz( n_traces=100):
    dataset = david.load_sampled_dataset()
    data = [('david',
                     'david_' + user,
                    'david_' + video,
                    # np.around(dataset[user][video][:n_traces, 0], decimals=2),
                     dataset[user][video][:n_traces, 1:]
                     ) for user in dataset.keys() for video in dataset[user].keys()]
    tmpdf = pd.DataFrame(data, columns=['ds', 'ds_user', 'ds_video','traces'])
    return tmpdf
'''


