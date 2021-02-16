import numpy as np
from classes import ParticipantScan, ParticipantsData, Scan, TransformationRecord

appcnt = 0
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

            prev_point = np.zeros((4, 1))
            prev_point[3, :] = 1
            reg_path_len = 0

            reg_transf = part.get_region(reg)
            for i in range(len(reg_transf.transformations)):
                record: TransformationRecord = reg_transf.transformations[i]
                next_point = record.trans_mat.dot(prev_point)
                reg_path_len = reg_path_len + np.linalg.norm(next_point - prev_point)
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
                angle_delta = (rotation_len(cur, prev) / (cur.time_stamp - prev.time_stamp))
                reg_ang_delta = reg_ang_delta + angle_delta


            part.angular_speed = reg_ang_delta / len(reg_transf.transformations)


def rotation_len(probe_to_ref_1: TransformationRecord, probe_to_ref_0: TransformationRecord):
    probe_to_ref_1 = probe_to_ref_1.trans_mat[:3, :3]
    probe_to_ref_0 = probe_to_ref_0.trans_mat[:3, :3]

    rotation_delta = probe_to_ref_1.dot(probe_to_ref_0.T)
    if (np.trace(rotation_delta) - 1) / 2 > 1 or (np.trace(rotation_delta) - 1) / 2 < -1:
        global appcnt
        appcnt = appcnt + 1

    return np.arccos((np.trace(rotation_delta) - 1) / 2)


def add_linear_speed(data: ParticipantsData):
    part: ParticipantScan
    for part in data:
        part.linear_speed = part.path_length / part.time

        for reg in Scan:
            if reg == Scan.ALL:
                continue

            reg_rec = part.get_region(reg)
            reg_rec.linear_speed = reg_rec.path_len / reg_rec.time
