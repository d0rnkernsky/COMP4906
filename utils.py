import numpy as np
from classes import ParticipantScan, ParticipantsData, Scan, TransformationRecord, ProficiencyLabel
import mhadatareader as p


def remove_spec_char(string):
    """
        Removes \n, \r, \\n, \\r, ", ' and strips the string
    """
    return string.replace('\n', '').replace('\r', '') \
        .replace('\\n', '').replace('\\r', '') \
        .replace("'", '').replace('"', '').strip()


def add_path_len(data: ParticipantsData):
    """

    """
    part: ParticipantScan
    for part in data:
        total_path_len = 0

        for reg in Scan:
            if reg == Scan.ALL:
                continue

            origin = np.zeros((4, 1))
            origin[3, :] = 1
            reg_path_len = 0
            reg_transf = part.get_region(reg)

            prev_point = reg_transf.transformations[0].trans_mat.dot(origin)
            for i in range(1, len(reg_transf.transformations)):
                record: TransformationRecord = reg_transf.transformations[i]
                next_point = record.trans_mat.dot(origin)
                record.path_length = np.linalg.norm(next_point - prev_point)
                reg_path_len = reg_path_len + record.path_length
                prev_point = next_point

            reg_transf.path_len = reg_path_len
            total_path_len = total_path_len + reg_path_len

        part.path_length = total_path_len


def add_angular_speed(data: ParticipantsData):
    part: ParticipantScan
    for part in data:
        for reg in Scan:
            if reg == Scan.ALL:
                continue

            reg_transf = part.get_region(reg)
            reg_ang_delta = 0
            for i in range(1, len(reg_transf.transformations)):
                cur: TransformationRecord = reg_transf.transformations[i]
                prev: TransformationRecord = reg_transf.transformations[i - 1]

                t_delta = (cur.time_stamp - prev.time_stamp)
                angle_speed = (rotation_len(cur, prev) / t_delta)
                cur.angular_speed = angle_speed

                reg_ang_delta = reg_ang_delta + angle_speed

            reg_transf.angular_speed = reg_ang_delta / len(reg_transf.transformations)


def rotation_len(probe_to_ref_1: TransformationRecord, probe_to_ref_0: TransformationRecord):
    probe_to_ref_1 = probe_to_ref_1.trans_mat[:3, :3]
    probe_to_ref_0 = probe_to_ref_0.trans_mat[:3, :3]

    rotation_delta = probe_to_ref_1.dot(probe_to_ref_0.T)
    arg = (np.trace(rotation_delta) - 1) / 2
    arg = min(arg, 1)
    arg = max(arg, -1)

    return np.arccos(arg)


def add_linear_speed(data: ParticipantsData):
    part: ParticipantScan
    for part in data:
        part.linear_speed = part.path_length / part.time

        for reg in Scan:
            if reg == Scan.ALL:
                continue

            reg_rec = part.get_region(reg)
            reg_rec.linear_speed = reg_rec.path_len / reg_rec.time
            for i in range(1, len(reg_rec.transformations)):
                rec: TransformationRecord
                rec = reg_rec.transformations[i]
                rec.linear_speed = rec.path_length / rec.time_stamp


def data_slicing(data, slice_len: int, label: ProficiencyLabel):
    slice_ratio = 0.9
    res = []
    slice_stride = int(np.floor(slice_len * slice_ratio))
    zero_vec = np.zeros((data[0].shape[0], 1))

    for i in range(len(data)):
        j = 0
        while j + slice_stride < data[i].shape[1]:
            rec = data[i][:, j:j + slice_stride]
            if np.allclose(zero_vec, rec[:, rec.shape[1] - 1]):
                break

            res.append(rec)
            j = j + 1

    return res, [label.value] * len(res)


def form_folds(novices, intermed, experts):
    return [(novices[:5], intermed[:5], [experts[0]]),
            (novices[5:10], intermed[5:10], [experts[1]]),
            (novices[10:], intermed[10:], [experts[2]])]


def shuffle(x_data, y_data):
    idx = np.random.permutation(len(x_data))
    x_shuffled = []
    y_shuffled = []
    for i in range(len(x_data)):
        x_shuffled.append(x_data[idx[i]])
        y_shuffled.append(y_data[idx[i]])

    return np.array(x_shuffled), np.array(y_shuffled),


def slice_sequence(novices, intermeds, experts, slice_window) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    x_novice, y_novice = data_slicing(novices, slice_window, ProficiencyLabel.Novice)
    x_intermed, y_intermed = data_slicing(intermeds, slice_window, ProficiencyLabel.Intermediate)
    x_expert, y_expert = data_slicing(experts, slice_window, ProficiencyLabel.Expert)

    x = np.array(x_novice + x_intermed + x_expert)
    y = np.array(y_novice + y_intermed + y_expert)

    return x, y


def prepare_data(novices, intermediates, experts):
    regions = dict()

    for _, reg in enumerate([Scan.LUQ, Scan.RUQ, Scan.PERICARD, Scan.PELVIC, Scan.ALL]):
        x_novice = prepare_data_reg(novices, reg)
        x_intermed = prepare_data_reg(intermediates, reg)
        x_expert = prepare_data_reg(experts, reg)

        regions[reg] = (x_novice, x_intermed, x_expert)

    return regions


def prepare_data_reg(data: ParticipantsData, reg: Scan):
    records = []
    part: ParticipantScan
    for part in data:
        part_records = None
        tr: TransformationRecord
        for tr in part.get_transforms(reg):
            tr: TransformationRecord
            to_vector = np.reshape(tr.trans_mat, (tr.trans_mat.shape[0] * tr.trans_mat.shape[1], 1))
            to_vector[to_vector.shape[0] - 4, 0] = tr.path_length
            to_vector[to_vector.shape[0] - 3, 0] = tr.angular_speed
            to_vector[to_vector.shape[0] - 2, 0] = tr.linear_speed
            to_vector[to_vector.shape[0] - 1, 0] = tr.time_stamp

            if type(part_records) != np.ndarray:
                part_records = to_vector
            else:
                part_records = np.append(part_records, to_vector, 1)

        records.append(part_records)

    return records


def load_data(dir_name):
    parser = p.MhaDataReader()

    # read novices
    novices = parser.read_data(f'{dir_name}/Novices/')
    sanity_check(novices)

    add_path_len(novices)
    add_linear_speed(novices)
    add_angular_speed(novices)

    # read intermediates
    intermediates = parser.read_data(f'{dir_name}/Intermediates/')
    sanity_check(intermediates)

    add_path_len(intermediates)
    add_linear_speed(intermediates)
    add_angular_speed(intermediates)

    # read experts
    experts = parser.read_data(f'{dir_name}/Experts/')
    sanity_check(experts)

    add_path_len(experts)
    add_linear_speed(experts)
    add_angular_speed(experts)

    return novices, intermediates, experts


def sanity_check(data: ParticipantsData):
    for part in data:
        assert len(part.get_transforms(Scan.ALL)) == \
               len(part.get_transforms(Scan.LUQ)) + len(part.get_transforms(Scan.RUQ)) + \
               len(part.get_transforms(Scan.PERICARD)) + len(part.get_transforms(Scan.PELVIC))

        assert part.get_time() == part.get_reg_time(Scan.RUQ) + part.get_reg_time(Scan.LUQ) + \
               part.get_reg_time(Scan.PERICARD) + part.get_reg_time(Scan.PELVIC)


def prepare_folds(train_fold, valid_fold, test_fold, slice_window):
    x_novice, x_intermed, x_expert = train_fold
    val_novice, val_intermed, val_expert = valid_fold
    test_novice, test_intermed, test_expert = test_fold

    x_train, y_train = slice_sequence(x_novice, x_intermed, x_expert, slice_window)
    x_val, y_val = slice_sequence(val_novice, val_intermed, val_expert, slice_window)
    x_test, y_test = slice_sequence(test_novice, test_intermed, test_expert, slice_window)

    x_train, y_train = shuffle(x_train, y_train)
    x_val, y_val = shuffle(x_val, y_val)
    x_test, y_test = shuffle(x_test, y_test)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
