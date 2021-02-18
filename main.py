from mhadatareader import MhaDataReader
from classes import ParticipantScan, ParticipantsData, Scan, ProficiencyLabel
import utils as ut
import numpy as np


def sanity_check(data: ParticipantsData):
    for part in data:
        assert len(part.get_transforms(Scan.ALL)) == \
               len(part.get_transforms(Scan.LUQ)) + len(part.get_transforms(Scan.RUQ)) + \
               len(part.get_transforms(Scan.PERICARD)) + len(part.get_transforms(Scan.PELVIC))

        assert part.get_time() == part.get_reg_time(Scan.RUQ) + part.get_reg_time(Scan.LUQ) + \
               part.get_reg_time(Scan.PERICARD) + part.get_reg_time(Scan.PELVIC)


def main():
    parser = MhaDataReader()
    intermediates = parser.read_data('./data/Intermediates/')
    sanity_check(intermediates)

    ut.add_path_len(intermediates)
    ut.add_linear_speed(intermediates)
    ut.add_angular_speed(intermediates)

    x_intermed = ut.prepare_data_all_reg(intermediates)

    experts = parser.read_data('./data/Experts/')
    sanity_check(experts)

    ut.add_path_len(experts)
    ut.add_linear_speed(experts)
    ut.add_angular_speed(experts)
    x_expert = ut.prepare_data_all_reg(experts)

    novices = parser.read_data('./data/Novices/')
    sanity_check(novices)

    ut.add_path_len(novices)
    ut.add_linear_speed(novices)
    ut.add_angular_speed(novices)
    x_novice = ut.prepare_data_all_reg(novices)

    all_data = x_novice + x_intermed + x_expert
    slice_len = ut.find_min_seq(all_data)

    x_novice, y_novice = ut.data_slicing(x_novice, slice_len, ProficiencyLabel.Novice)
    x_intermed, y_intermed = ut.data_slicing(x_intermed, slice_len, ProficiencyLabel.Intermediate)
    x_expert, y_expert = ut.data_slicing(x_expert, slice_len, ProficiencyLabel.Expert)

    all_data = x_novice + x_intermed + x_expert
    # max_len = ut.find_max_seq(all_data)
    # ut.pad_data_to_max(all_data, max_len)

    X = np.array(all_data)
    Y = np.append(y_novice, np.append(y_intermed, y_expert, 0), 0)

    assert X.shape[0] == Y.shape[0]

    print('done')

if __name__ == "__main__":
    main()
