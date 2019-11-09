import os, shutil, time

'''
    Script to reduce the number of saved checkpoints and to delete the data recorded for tensorboard
    evaluation in order to preserve manageable size of evaluation data.
'''

def get_complete_list_of_checkpoints_to_be_removed(path):
    """
        Remove unnecessary data blowing up the repo.
    :return:    -
    """
    models = os.listdir(path)
    to_be_removed = []
    to_be_kept = []
    print()

    for model in models:
        print(model)
        print('Path + dir: ' + path+model)
        test_runs = os.listdir(path+model)
        for check_point_dir in test_runs:
            if 'checkpoint_' in check_point_dir:
                print('Checkpoint: ' + check_point_dir)
                checkpoint_nr = check_point_dir.replace('checkpoint_', '').replace('.zip', '')
                if not int(checkpoint_nr) % 500 == 0:
                    to_be_removed.append(path+model+"/"+check_point_dir)
                else:
                    to_be_kept.append(path+model+"/"+check_point_dir)

            if 'tensorboard' in check_point_dir:
                to_be_removed.append(path + model + "/" + check_point_dir)
            else:
                to_be_kept.append(path + model + "/" + check_point_dir)
    return to_be_removed, to_be_kept



to_be_removed, to_be_kept = get_complete_list_of_checkpoints_to_be_removed('Results/PPO2/')  # 'Results_Daniel/PPO2/'


print('##### TO BE KEPT: #####')
for d in to_be_kept:
    pass
    print('To be kept: ' + d)

print('##### TO BE REMOVED: #####')
for d in to_be_removed:
    print('To be removed: ' + d)

confirmation = input('--- Do you intend to remove the files/folders listed above??? Type \'YeSyEs\': ')

if confirmation == 'YeSyEs':
    print('Going to remove in 5 seconds...')
    time.sleep(5)
    for d in to_be_removed:
        try:
            os.remove(d)
        except IsADirectoryError:
            shutil.rmtree(d)
else:
    print('Failed to confirm. No action taken.')

print('Done.')

