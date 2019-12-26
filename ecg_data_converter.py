
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import terminalColors as tc
import os
import shutil
import wfdb


result_dir = 'results/ecg'

def visualize(df, show=True):
    fig = plt.figure()

    plt.title('Record sel102 from Physionet')
    df.V5.plot()
    # df.V2.plot()
    save_data(df)

    if show:
        plt.show()
    fig.tight_layout()
    file_path = '{}/ecg.png'.format(result_dir)
    fig.savefig(file_path)
    plt.close()
    tc.green(
        'Saved {}.'.format(file_path)
    )

def save_data(df):
    """Write data to pandas csv file.
    """
    fn = '{}/sel102.csv'.format(result_dir)
    df.to_csv(fn, index=False) 
    tc.green('Saved data in {}.'.format(fn))

# Demo 1 - Read a wfdb record using the 'rdrecord' function into a wfdb.Record object.
# Plot the signals, and show the data.
# record = wfdb.rdrecord('sample-data/a103l') 
record = wfdb.rdrecord('./data/qt-database-1.0.0/sel102')
signal = record.p_signal
df = pd.DataFrame(record.p_signal, columns=record.sig_name)
visualize(df)
# wfdb.plot_wfdb(record=record, title='Record sel102 from Physionet Challenge 2015') 
# print(record.__dict__)


# Can also read the same files hosted on Physiobank https://physionet.org/physiobank/database/
# in the challenge/2015/training/ database subdirectory. Full url = https://physionet.org/physiobank/database/challenge/2015/training/
# record2 = wfdb.rdrecord('a103l', pb_dir='challenge/2015/training/')