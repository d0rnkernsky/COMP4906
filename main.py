from mhadatareader import MhaDataReader
from classes import ParticipantScan, ParticipantsData, Scan
import utils as ut

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

    experts = parser.read_data('./data/Experts/')
    sanity_check(experts)

    ut.add_path_len(experts)
    ut.add_linear_speed(experts)
    ut.add_angular_speed(experts)

    novices = parser.read_data('./data/Novices/')
    sanity_check(novices)

    ut.add_path_len(novices)
    ut.add_linear_speed(novices)
    ut.add_angular_speed(novices)

    print('done')


if __name__ == "__main__":
    main()