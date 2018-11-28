import os


def recursive_dir_scan(current_dir):
    all_files = []
    all_dirs = []
    for root, dirs, files in os.walk(current_dir):
        for file_name in files:
            all_files.append(os.path.join(root, file_name))
        for dir_name in dirs:
            sub_dir = os.path.join(root, dir_name)
            all_dirs.append(sub_dir)
    return all_files, all_dirs
