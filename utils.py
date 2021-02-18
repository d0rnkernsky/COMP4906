import numpy as np
from classes import ParticipantScan, ParticipantsData, Scan, TransformationRecord, ProficiencyLabel


def remove_spec_char(string):
    """
        Removes \n, \r, \\n, \\r, ", ' and strips the string
    """
    return string.replace('\n', '').replace('\r', '') \
        .replace('\\n', '').replace('\\r', '') \
        .replace("'", '').replace('"', '').strip()


def add_path_len(data: ParticipantsData):
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

                t_delta = max((cur.time_stamp - prev.time_stamp), 0.000000000001)
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


def prepare_data_all_reg(data: ParticipantsData):
    records = []
    part: ParticipantScan
    for part in data:
        shape = (16, 1)
        part_records = None
        tr: TransformationRecord
        for tr in part.get_transforms(Scan.ALL):
            tr: TransformationRecord
            to_vector = np.reshape(tr.trans_mat, shape)
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


def find_max_seq(in_data: list):
    max_len = in_data[0].shape[1]

    for i in range(len(in_data)):
        rec: np.ndarray
        rec = in_data[i]

        if rec.shape[1] > max_len:
            max_len = rec.shape[1]

    return max_len


def find_min_seq(in_seq: list):
    min_len = in_seq[0].shape[1]

    for i in range(len(in_seq)):
        rec: np.ndarray
        rec = in_seq[i]

        if rec.shape[1] < min_len:
            min_len = rec.shape[1]

    return min_len


def pad_data_to_max(in_data: list, MAX_LEN: int):
    assert MAX_LEN is not None

    for i in range(len(in_data)):
        if MAX_LEN > in_data[i].shape[1]:
            in_data[i] = np.pad(in_data[i], [(0, 0), (0, MAX_LEN - in_data[i].shape[1])], mode='constant',
                                constant_values="0")


def data_slicing(data, slice_len: int, label: ProficiencyLabel):
    SLICE_RATIO = 0.9
    res = []
    slice_len = int(np.floor(slice_len * SLICE_RATIO))

    for i in range(len(data)):
        j = 0
        while j + slice_len < data[i].shape[1]:
            res.append(data[i][:, j:j + slice_len])
            j = j + 1

    return res, np.full((len(res),), label.value)
