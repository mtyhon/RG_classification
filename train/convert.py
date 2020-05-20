import numpy as np
import pandas as pd
import os, re

def psd_to_npz():
    jie_file = 'JieData_Full2018.txt'
    jie_df = pd.read_csv(jie_file, header=0, delimiter='|')
    jie_kic = jie_df['KICID'].values
    jie_numax = jie_df['numax'].values
    jie_numax_err = jie_df['numax_err'].values
    jie_dnu = jie_df['dnu'].values
    jie_dnu_err = jie_df['dnu_err'].values

    consensus_df = pd.read_csv('Elsworth_Jie_2019_ID_Label.dat', header=0, delim_whitespace=True)
    consensus_kic = consensus_df['KIC'].values
    consensus_label = consensus_df['Label'].values
    unique_kic = []

    source_folder= 'psd_file/'
    count = 0
    for filename in os.listdir(source_folder):
        file_kic = int(re.search(r'\d+', filename).group())
        segment_number  = 0
        
        print("Working on KIC %s" % int(file_kic))

        if file_kic not in consensus_kic:
            continue
        if file_kic not in jie_kic:
            continue
        jie_index = np.where(jie_kic == file_kic)[0]

        assert len(jie_index) == 1
        file_numax = jie_numax[jie_index]
        file_numax_err=jie_numax_err[jie_index]
        file_dnu = jie_dnu[jie_index]
        file_dnu_err = jie_dnu_err[jie_index]

        merge = os.path.join(source_folder, filename)
        if (merge[-3:]) == 'csv':
            df = pd.read_csv(merge, header=None)
        else:
            df = pd.read_table(merge, delim_whitespace=True, header=None)

        df.columns = ['Frequency', 'Power']
        freq = df['Frequency'].values
        power = df['Power'].values

        spectra_kic = int(file_kic)

        m_index = np.where(consensus_kic == spectra_kic)
        pop = consensus_label[m_index]

        np.savez_compressed('npz_file/%d-%d'%(spectra_kic,segment_number),
                            freq=freq,power=power,pop=pop, numax=file_numax, numax_err = file_numax_err,
                            dnu=file_dnu, dnu_err=file_dnu_err)
        unique_kic.append(spectra_kic)
        count += 1

        print('Count: %d' %count)
    print('Unique KICs saved: ', len(np.unique(unique_kic)))

if __name__ == '__main__':
    psd_to_npz() 
