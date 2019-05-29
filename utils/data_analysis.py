import os
import csv
import glob
import shutil
from contextlib import ExitStack


labels1 = ['Fr', 'An', 'Hp', 'Sd', 'Dg', 'Ct', 'Am']
labels2 = ['A', 'V', 'I']


ROOT_DIR = '../data'


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


def get_main_table():
    data_path = os.path.join(ROOT_DIR, 'SEMAINE')
    store_path = os.path.join(ROOT_DIR, 'SEMAINE_STAT')

    main_table_file = os.path.join(store_path, 'common_stat.csv')

    dirs = os.listdir(data_path)

    sess_dict = {}
    for d in dirs:

        if not os.path.isdir(os.path.join(data_path, d)):
            continue

        # print('Start processing directory {}'.format(os.path.join(data_path, d)))

        files = glob.glob(os.path.join(data_path, d, '*.txt'))

        for file in files:
            name = os.path.basename(file)
            if not name.startswith('R'):
                # print(name)
                continue
            idx = name.find('S')
            rater = name[1:idx]

            idx1 = name.find('T')
            session_num = name[idx+1:idx1]

            target = name[idx1+1]

            idx2 = name.find('D')
            emo_class = name[idx2 + 1:-4]

            idx3 = name.find('C')
            char_class = name[idx3 + 1:idx3 + 3]

            if emo_class not in labels1 and emo_class not in labels2:
                # print('Annotation file for {} class, skip this.'.format(emo_class))
                continue

            if session_num not in sess_dict.keys():
                sess_dict[session_num] = {
                    target: {
                        char_class: {}
                    }
                }
                for label in labels1 + labels2:
                    sess_dict[session_num][target][char_class][label] = 0
            if target not in sess_dict[session_num].keys():
                sess_dict[session_num][target] = {
                    char_class: {}
                }
                for label in labels1 + labels2:
                    sess_dict[session_num][target][char_class][label] = 0
            if char_class not in sess_dict[session_num][target].keys():
                sess_dict[session_num][target][char_class] = {}
                for label in labels1 + labels2:
                    sess_dict[session_num][target][char_class][label] = 0
            sess_dict[session_num][target][char_class][emo_class] += 1

    with open(main_table_file, 'w') as log_file:
        pref_labels1 = list(map(lambda x: 'D{}'.format(x), labels1))
        pref_labels2 = list(map(lambda x: 'D{}'.format(x), labels2))
        log_file.write('S,T,C,{},{}\n'.format(','.join(pref_labels1), ','.join(pref_labels2)))

        for session in sess_dict.keys():
            for target in sess_dict[session].keys():
                for char_class in sess_dict[session][target].keys():
                    line = '{},{},{}'.format(session, target, char_class)
                    for label in labels1 + labels2:
                        line += ',{}'.format(sess_dict[session][target][char_class][label])
                    line += '\n'
                    log_file.write(line)


if __name__ == '__main__':
    # parse_files()
    get_main_table()
