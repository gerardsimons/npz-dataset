# An example implementation of using NPZRecord and NPZDataset to store ECG data, QRS and features
from npz_record import NPZRecord, NPZDataset


class ECGRecord(NPZRecord):
    def __init__(self, path):
        super().__init__(path, keys=['ecg', 'qrs', 'feat_x', 'feat_y'])





class ECGDataset(NPZDataset):
    """ ECGDataset contains ECG data, optional QRS complexes and any relevant features """
    pass



def try_ecg_dataset():

