import os
import csv
import glob
import shutil
from contextlib import ExitStack


labels1 = ['Fr', 'An', 'Hp', 'Sd', 'Dg', 'Ct', 'Am']
labels2 = ['A', 'V', 'I']


ROOT_DIR = '../data/SEMAINE_ROOT'


def parse_files():

    data_path = os.path.join(ROOT_DIR, 'SEMAINE')
    store_path = os.path.join(ROOT_DIR, 'SEMAINE_STAT')

    dirs = os.listdir(data_path)

    for d in dirs:

        if not os.path.isdir(os.path.join(data_path, d)):
            continue

        print('Start processing directory {}'.format(os.path.join(data_path, d)))

        stat_dir_path = os.path.join(store_path, d)
        if not os.path.exists(stat_dir_path):
            os.mkdir(stat_dir_path)

        files = glob.glob(os.path.join(data_path, d, '*.txt'))

        group_by_rater = {}

        for f in files:
            name = os.path.basename(f)

            if not name.startswith('R'):
                print('{}: is this shit??\n skip'.format(f))
                continue

            idx = name.find('S')
            rater = name[1:idx]

            idx2 = name.find('D')
            emo_class = name[idx2+1:-4]

            if emo_class not in labels1 and emo_class not in labels2:
                continue

            if rater in group_by_rater.keys():
                group_by_rater[rater].append(f)
            else:
                group_by_rater[rater] = [f]

        if len(group_by_rater) == 0:
            print('yes! folder {} is shit!'.format(d))
            shutil.rmtree(stat_dir_path)
            continue

        num_raters = len(group_by_rater)
        num_useless = 0

        for key, val in group_by_rater.items():
            stat_file_path = os.path.join(stat_dir_path, 'R{}.csv'.format(key))

            names = [os.path.basename(f_path) for f_path in val]
            all_emo_classes = [f_name[f_name.find('D') + 1:-4] for f_name in names]

            if len(set(all_emo_classes) & set(labels2)) == 0 or len(set(all_emo_classes) & set(labels1)) == 0:
                print('No target classes for this rater: {}'.format(key))
                num_useless += 1
                continue

            with ExitStack() as stack:
                files = [stack.enter_context(open(f_path)).readlines() for f_path in val]
                num_lines = list(map(lambda x: len(x), files))

                if len(set(num_lines)) > 1:
                    print('Files num lines don\'t match: {}!'.format(num_lines))
                    num_useless += 1
                    continue

                with open(stat_file_path, 'w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=',')
                    csv_writer.writerow(['time'] + list(map(lambda x: 'D{}'.format(x), all_emo_classes)))

                    for i in range(num_lines[0]):
                        line = [lines[i].strip().split(' ')[1] for lines in files]
                        csv_writer.writerow([files[0][i].strip().split(' ')[0]] + line)

        if num_raters == num_useless:
            shutil.rmtree(stat_dir_path)


if __name__ == '__main__':
    parse_files()
