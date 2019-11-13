import os, sys
import json, csv
import numpy as np
from array import array

def create_dir(direct):
    """
        Ensure that a given path exists.
        :param direct: Directory to be created when necessary.
        :return: -
    """
    if not os.path.exists(direct):
        os.makedirs(direct)

parentDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PATH_READ = os.path.join(parentDirectory, 'Results', 'PPO2')
PATH_READ = '../Results/PPO2/'
PATH_WRITE = "../TrainingProgressEvaluation/"
print(PATH_READ)


def get_complete_trials(path):
    """
        Takes a path and checks which of the folders/archives inside given folder (specified by path) contain a
        final model generated during the training progress, and hence determines which data sets in terms of archives
        are complete (due to complete training process).
    :param path: Path to archive(s)-location
    :return:    dirs: List of directories/archives only containing those with complete data sets.
                list_outtakes: List of data directories/archives containing not completed test runs indicated by
                               absence of final model.
    """
    dirs = os.listdir(path)
    list_outtakes = []
    print('All dirs:')
    print(dirs)
    print()

    for direct in dirs:
        f = []
        for (dirpath, dirnames, filenames) in os.walk(path+direct):
            f.extend(filenames)
            break
        if 'final_model.zip' in f:
            # print('DATA SET COMPLETE')
            pass
        else:
            print('Incomplete data at: ' + direct)
            list_outtakes.append(direct)
            dirs.remove(direct)

    return dirs, list_outtakes

def return_parameter_specification_id_for_file(file_name):
    """
        Given a folder name of a folder containing data created during model's training, the method returns which
        parameter specification file was used to specify the parameters as a function of which the agent's training
        proceeded.
    :param file_name: Folder containing data created during training.
    :return: -
    """
    with open(file_name) as json_file:
        data = json.load(json_file)
    try:
        # Data might not contain requested element
        return data['provided_params_file']
    except KeyError:
        return None

def remove_redundant_runs(path, data_directories):
    """
        Filter out redundant back-up-test runs which are too many.
        :param path: Path to directories potentially to be included
        :param data_directories: Candidate directories to potentially be included in analysis
        :return:
            During training, each train run was called with a given parameter specification provided in a named file.
            This function reads out from the data created during training which parameter specification file was
            used for generating the respective data set specified by a given data_directory.
    """
    parameter_specification_ids = dict()
    discarded_data_directories = []
    maxElements = 5
    #print('Potential data directories: ')
    #print(data_directories)
    for directory in data_directories:
        #print('To be checked: ' + path+directory+'/params.json')
        parameter_specification_id = return_parameter_specification_id_for_file(path+directory+'/params.json')
        if parameter_specification_id is not None:
            if parameter_specification_id not in parameter_specification_ids.keys():
                id_list = [directory]
                parameter_specification_ids[parameter_specification_id] = id_list
            else:
                prev_records = parameter_specification_ids[parameter_specification_id]
                if len(prev_records) < maxElements:
                    prev_records.append(directory)
                    parameter_specification_ids[parameter_specification_id] = prev_records
                else:
                    discarded_data_directories.append(directory)
        else:
            discarded_data_directories.append(directory)

    #print(parameter_specification_ids)
    #print('Filtered: ')
    #print(discarded_data_directories)
    return parameter_specification_ids, discarded_data_directories


def clean_parameter_specification_id_string(param_id):
    # Clean id which was itself a directory+id beforehand
    param_id = param_id.replace('ParameterSettings/', '')
    param_id = param_id.replace('.json', '')
    param_id = param_id.replace('.', '_')
    param_id = param_id.replace('/', '_')
    return param_id


def save_which_data_was_used(direct, used_dict, not_used_lists):
    """
        Specify which folders containing data were used for evaluation. Each folder contains all data generated during
        a single training run.
    :param direct: Path to a folder containing all used/non-used folders/directories.
    :param used_dict: Dictionary containing the list of folders used for evaluation per param_setting_specification_id
    :param not_used_lists: Two list containing folders excluded from evaluation;
                           1. Excluded due to being incomplete
                           2. Excluded due to number of data directories to be included per param setting being exceeded
    :return: -
    """
    create_dir(direct)

    # Save as csv which data was used (nicer to read)
    with open(direct + "/" + "used_data.csv", "w") as f:
        w = csv.writer(f, dialect='excel', quoting=csv.QUOTE_ALL)

        # Save overview: for param-specification-id: [param-specification-id, count-of-used-models]
        w.writerow(['-- Overview about how many models per parameter-specification were used for evaluation:'])
        w.writerow(['Parameter specification id:', 'Count of models:'])
        for key, val_list in used_dict.items():
            key = clean_parameter_specification_id_string(key)
            w.writerow([key, len(val_list)])
        w.writerow(['-- End of list.'])
        w.writerow([''])

        # Save actual data pairs [param-specification-id, used-data-set-containing-trained-model]
        w.writerow(['-- Which models were used per parameter-specification:'])
        w.writerow(['Parameter specification id:', 'Model:'])
        for key, val_list in used_dict.items():
            key = clean_parameter_specification_id_string(key)
            for val in val_list:
                w.writerow([key, val])
        w.writerow(['-- End of list.'])
    f.close()

    # Save as csv which data was not used (nicer to read)
    with open(direct + "/" + "not_used_data.csv", "w") as f:
        w = csv.writer(f, dialect='excel', quoting=csv.QUOTE_ALL)
        w.writerow(['-- Incomplete data sets:'])
        for val in not_used_lists[0]:
            w.writerow([val])
        w.writerow(['-- End of list.'])
        for _ in range(2):
            w.writerow([''])
        w.writerow(['-- Excluded due to number of data directories to be included per param setting being exceeded:'])
        for val in not_used_lists[1]:
            w.writerow([val])
        w.writerow(['-- End of list.'])

    f.close()


def save_data_package_to_file(direct, name, data_arr):
    """
        General purpose data saving. Takes either list or Numpy array as input for saving.
    :param direct: Folder where to store emission file
    :param name: Indication how to call resulting file
    :param data_arr: List/Numpy-array containing data (arrays saved row-wise to file)
    :return: -
    """
    create_dir(direct)
    # Clean name which was itself a directory+name beforehand
    name = name.replace('ParameterSettings/', '')
    name = name.replace('.json', '')
    name = name.replace('.', '_')
    name = name.replace('/', '_')

    # Save as csv
    with open(direct + "/" + name + ".csv", "w") as f:
        w = csv.writer(f, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
        for row in data_arr:
            w.writerow(row)
        pass

    f.close()


def evaluate_measurements_per_param_specification(path, params):
    """
        First, over a given number of an agent's weight updates, the number of successful grasps is recorded via the
        callback function. Particularly, after 10 weight updates have been performed, the number of total grasps
        performed over the last 10 weight updates, where each weight update is executed after 128 simulated time steps
        have been executed, is saved to file for a given trained model.
        Similarly, the number of time steps it took the robotic arm to perform a successful grasp is recorded for each
        successful grasp over the given period of 10 weight updates.
        After that period, i.e. once 10 weight updates have been performed, the callback function not only saves the
        total number of grasps performed during the last 10 weight updates, but also the mean number of time steps it
        took the model to perform these grasps, where the mean number of time steps is computed as the mean over all the
        numbers of time steps recorded over the last 10 weight updates.
        Additionally, also the standard deviation corresponding to the mean number of time steps it took the model to
        perform a successful grasp is computed as the standard deviation over all the numbers of time steps recorded
        over the last 10 weight updates. This also gets saved to file.
        All these information get saved row-wise to a file called training_eval.csv, which is included in a models's
        respective folder.

        Afterwards, this function collects the training_eval.csv files from all the models trained given a certain
        parameter-setting and creates summarizing files. Most importantly, the following files get created:
            1. MeansOverGrasps.csv: This file contains the mean number of grasps achieved across all 5 models within 10
            weight updates trained on a specific parameter setting as a function of the number of performed weight
            updates. The mean number of grasps computed over the 5 models per parameter setting as a function of weight
            updates is provided as a function of the performed weight updates.
            2. StdOverGrasps.csv: Contains for each parameter-setting as a function of performed weight updates the
            standard deviation corresponding to the mean number of grasps achieved over 10 weight updates across all the
            models belonging to a given parameter-setting.
            3. MeanOverMeanGraspingTimes.csv: Given the mean grasp times it took a model to perform a successful grasp
            (computed over 10 weight updates of the model) as a function of the number of performed weight updates, this
            file contains for each parameter-setting the mean number of time steps it took to perform a successful grasp
            per parameter-setting, computed over all models trained on a given parameter setting. The result is, again,
            a function of performed weight updates.
            4. MeanOverStdOfGraspingTimes.csv: Given the standard deviation corresponding to the mean grasp times it
            took a model to perform a successful grasp as a function of the number of performed weight updates, this
            file contains for each parameter-setting as a function of performed weight updated the mean standard
            deviation per parameter-setting computed over all the respective standard deviations of all models trained
            on a given parameter setting.

    :param path: Path to where saved models are located.
    :param params: Dictionary; Key: parameter-setting-id; Value: list of model names belonging to models trained given a
    certain parameter setting specified by the parameter-setting-id encoded in a list's respective key.
    :return: -
    """


    print('PARAMS----')
    print(params)
    rows_assigned = False
    summary_grasping_means = []     # Mean of number of grasps over 5 test runs per parameter specification
    summary_grasping_stds = []      # Std of number of grasps over 5 test runs per parameter specification
    # Average over [mean number of time steps needed for performing a successful grasp action over 10 updates] over 5 test runs per param specification:
    summary_grasping_time_mean_means = []
    # Average over [std of time steps needed for performing a successful grasp action over 10 updates] over 5 test runs per param specification:
    summary_grasping_time_std_means = []
    header_summary_graspings_m = ['Updates']
    header_summary_graspings_s = ['Updates']
    header_summary_grasping_t_m_m = ['Updates']
    header_summary_grasping_t_s_m = ['Updates']

    for key in params.keys():
        init = True
        header_graspings = []

        header_summary_graspings_m.append(key.replace('ParameterSettings/', '').replace('.json', ''))
        header_summary_graspings_s.append(key.replace('ParameterSettings/', '').replace('.json', ''))
        header_summary_grasping_t_m_m.append(key.replace('ParameterSettings/', '').replace('.json', ''))
        header_summary_grasping_t_s_m.append(key.replace('ParameterSettings/', '').replace('.json', ''))

        # Collect data from all files that made use of that param setting defined by the given key
        for file_name in params[key]:
            header_graspings.append(file_name)
            eval_file_name = path + file_name + '/training_eval.csv'
            with open(eval_file_name, 'r') as dest_f:
                # Read in a model's 'private' training log file
                data_iter = csv.reader(dest_f,
                                       quotechar='"',
                                       dialect='excel',
                                       quoting=csv.QUOTE_ALL)
                data = [data for data in data_iter]     # Concatenate rows...
            data_array = np.asarray(data)               # ... and transform them to array

            cropped = data_array[1:, :]             # Remove header-row
            cropped = cropped.astype(np.float)      # By default, floats are read in as strings from file. Convert back.

            # In summary file: first column contains the number of performed weight-updates row-wise...
            if not rows_assigned:
                summary_grasping_means = [cropped[:, 0]]
                summary_grasping_stds = [cropped[:, 0]]
                summary_grasping_time_mean_means = [cropped[:, 0]]
                summary_grasping_time_std_means = [cropped[:, 0]]
                rows_assigned = True

            # From model's 'personal' file: read in the
            #   - total number of grasps obtained over every 10 weight updates,
            #   - the mean time needed to get into grasping pose over last 10 weight updates,
            #   - the std corresponding to the mean time needed to get into grasping pose over last 10 weight updates,
            # respectively.
            grasps = cropped[:, 1]   # Grasps achieved by a read-in model over every 10 last weight updates
            avg_time = cropped[:, 2] # Avg time it took a model over every 10 last weight updates to get into grasp pose
            std_time = cropped[:, 3] # Std corresponding to mean/average time for every 10 weight updates above.

            if init:
                combined_measurements_grasps = [grasps]     # Grasp counts combined over all files created given
                                                            # key's-param-settings
                combined_measurements_avg = [avg_time]
                combined_measurements_std = [std_time]
                init = False
            else:
                combined_measurements_grasps.append(grasps)
                combined_measurements_avg.append(avg_time)
                combined_measurements_std.append(std_time)

        # Log data collected from all models trained on a given parameter setting specified by key.

        # Analyze data
        # # Graspings...
        # combined_grasps_arr contains the total number of grasps obtained for each individual model over every last 10
        # weight updates (listed row-wise) for all models trained on a given parameter-setting specified by key (listed
        # column-wise)
        combined_grasps_arr = np.array(combined_measurements_grasps).transpose()

        try:
            combined_grasps_mean_arr = np.array([np.nanmean(combined_grasps_arr, axis=1)])  # mean computed row-wise
        except RuntimeWarning:
            pass  # Attempt to suppress printing of these warnings
        combined_grasps_stds_arr = np.array([np.nanstd(combined_grasps_arr, axis=1)])

        header_mean_times_mean = header_graspings.copy()
        header_mean_times_std = header_graspings.copy()
        header_graspings.extend(['Mean_over_grasps', 'Std_over_grasps'])

        grasps_total_mean_std = np.append(combined_grasps_arr, combined_grasps_mean_arr.transpose(), axis=1)  # add mean
        grasps_total_mean_std = np.append(grasps_total_mean_std, combined_grasps_stds_arr.transpose(), axis=1)  #add std
        header_graspings = np.array(header_graspings)
        grasps_total_mean_std = np.append([header_graspings], grasps_total_mean_std, axis=0)  # Adding header line
        # Save summary
        save_data_package_to_file(PATH_WRITE+"Grasps", key + "_mean_std", grasps_total_mean_std)

        # # Mean mean grasp times...
        combined_mean_times_arr = np.array(combined_measurements_avg).transpose()

        mean_combined_mean_times_arr = np.array([np.nanmean(combined_mean_times_arr, axis=1)])

        grasp_times_total_mean = np.append(combined_mean_times_arr, mean_combined_mean_times_arr.transpose(), axis=1)  # add mean

        header_mean_times_mean.extend(['Mean_over_mean_grasp_times'])
        header_mean_times_mean = np.array(header_mean_times_mean)
        grasp_times_mean_total_mean = np.append([header_mean_times_mean], grasp_times_total_mean, axis=0)  # Adding header line

        # Save summaries
        save_data_package_to_file(PATH_WRITE+"MeansOfGraspMeanTimes", key + "_mean", grasp_times_mean_total_mean)

        # # Mean std grasp times...
        combined_std_times_arr = np.array(combined_measurements_std).transpose()

        mean_combined_std_times_arr = np.array([np.nanmean(combined_std_times_arr, axis=1)])

        grasp_times_total_std = np.append(combined_std_times_arr, mean_combined_std_times_arr.transpose(),
                                           axis=1)  # add mean

        header_mean_times_std.extend(['Mean_over_std_grasp_times'])
        header_mean_times_stds = np.array(header_mean_times_std)
        grasp_times_stds_total_mean = np.append([header_mean_times_stds], grasp_times_total_std,
                                                axis=0)  # Adding header line

        # Save summary
        save_data_package_to_file(PATH_WRITE+"MeansOfGraspTimeStds", key + "_mean", grasp_times_stds_total_mean)

        summary_grasping_means.append(combined_grasps_mean_arr[0].tolist())
        summary_grasping_stds.append(combined_grasps_stds_arr[0].tolist())
        summary_grasping_time_mean_means.append(mean_combined_mean_times_arr[0].tolist())
        summary_grasping_time_std_means.append(mean_combined_std_times_arr[0].tolist())


    print('*********************SUMMARY*********************')
    summary_grasping_means = np.array(summary_grasping_means).transpose()
    print(header_summary_graspings_m)
    print(summary_grasping_means)
    summary_grasping_means = np.append([header_summary_graspings_m], summary_grasping_means, axis=0)
    print(summary_grasping_means)

    # Save summary
    save_data_package_to_file(PATH_WRITE+"Summary", "MeansOverGrasps", summary_grasping_means)

    summary_grasping_stds = np.array(summary_grasping_stds).transpose()
    summary_grasping_stds = np.append([header_summary_graspings_s], summary_grasping_stds, axis=0)
    print(summary_grasping_stds)

    # Save summary
    save_data_package_to_file(PATH_WRITE+"Summary", "StdOverGrasps", summary_grasping_stds)

    summary_grasping_time_mean_means = np.array(summary_grasping_time_mean_means).transpose()
    summary_grasping_time_mean_means = np.append([header_summary_grasping_t_m_m], summary_grasping_time_mean_means, axis=0)
    print(summary_grasping_time_mean_means)

    # Save summary
    save_data_package_to_file(PATH_WRITE+"Summary", "MeanOverMeanGraspingTimes", summary_grasping_time_mean_means)

    summary_grasping_time_std_means = np.array(summary_grasping_time_std_means).transpose()
    summary_grasping_time_std_means = np.append([header_summary_grasping_t_s_m], summary_grasping_time_std_means, axis=0)
    print(summary_grasping_time_std_means)

    # Save summary
    save_data_package_to_file(PATH_WRITE+"Summary", "MeanOverStdOfGraspingTimes", summary_grasping_time_mean_means)

    print()


candidate_dirs, list_outtakes_failure = get_complete_trials(PATH_READ)
used_dict, filtered_out = remove_redundant_runs(PATH_READ, candidate_dirs)
evaluate_measurements_per_param_specification(PATH_READ, used_dict)

not_used_lists = [list_outtakes_failure, filtered_out]

print('Complete and sufficient runs:')
print(candidate_dirs)
print('Outtakes:')
print(not_used_lists)
print()
print('Used parameter id\'s and the test runs that used them:')
print(used_dict)


save_which_data_was_used(PATH_WRITE+"EvaluatedData", used_dict, not_used_lists)
