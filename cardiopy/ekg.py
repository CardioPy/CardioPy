""" This file contains the EKG class .

    All R peak detections should be manually inspected with EKG.plotpeaks method and
    false detections manually removed with rm_peaks method. After rpeak examination, 
    NaN data can be accounted for by removing false IBIs with rm_ibi method.

    TO DO:
        ** Update docstrings **
        1. Add option to extract sampling frequency & milliseconds from time column
        2. Re-add code to import previously cleaned nn data -- DONE. 11-24-19 in hrv_stats method
        3. Add range options for indices for rm peaks and rm ibis
        4. Add more descriptive error message for ValueError encountered during
            add_peaks if range is outside of data
        5. Add option for auto-determining threshold parameters (mw_size and upshift)
        6. Add threshold statistics (sensitivity & PPV) to output
        7. Update hrv_stats to assume NN
        8. Add nn attribute for data that doesn't require cleanings
        9. Fix spreadsheet alignment when smoothing used (incorp smooth & sm_wn metadata to all files)
        10. Option for 1000Hz interpolation prior to peak detection; Option for 4Hz resampling of NN tachogram
            rather than original sampling frequencys

"""

import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np 
import os
import pandas as pd 
import scipy as sp
import shapely.geometry as SG
import statistics

from mne.time_frequency import psd_array_multitaper
from pandas.plotting import register_matplotlib_converters
from scipy.signal import welch

class EKG:
    """ General class containing EKG analyses
    
    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing 'EKG' column data. First two rows are headers [Channel, Datatype]
    
    Attributes
    ----------
    """

    def __init__(self, fname, fpath, min_dur=True, epoched=True, smooth=False, sm_wn=30, mw_size=100, upshift=3.5, rm_artifacts=False, detect_peaks=True):
        """ Initialize raw EKG object

        Parameters
        ----------
        fname: str
            filename
        fpath: str
            path to file
        min_dur: bool (default:True)
            only load files that are >= 5 minutes long
        epoched: bool (default: True)
            Whether file was epoched using ioeeg
        smooth: BOOL (default: False)
            Whether raw signal should be smoothed before peak detections. Set True if raw data has consistent high frequency noise
            preventing accurate peak detection
        sm_wn: float (default: 30)
            Size of moving window for rms smoothing preprocessing (milliseconds)
        mw_size: float (default: 100)
            Moving window size for R peak detection (milliseconds)
        upshift: float (default: 3.5)
            Detection threshold upshift for R peak detection (% of signal)
        rm_artifacts: bool (default: False)
            Apply IBI artifact removal algorithm

        Returns
        -------
        EKG object w/ R peak detections and calculated inter-beat intervals
         """

        # set metadata
        filepath = os.path.join(fpath, fname)
        if epoched == False:
            in_num, start_date, slpstage, cycle = fname.split('_')[:4]
        elif epoched == True:
            in_num, start_date, slpstage, cycle, epoch = fname.split('_')[:5]
        self.metadata = {'file_info':{'in_num': in_num,
                                'fname': fname,
                                'path': filepath,
                                'start_date': start_date,
                                'sleep_stage': slpstage,
                                'cycle': cycle
                                }
                        }
        if epoched == True:
            self.metadata['file_info']['epoch'] = epoch
        
        # load the ekg
        self.load_ekg(min_dur)

        if smooth == True:
            self.rms_smooth(sm_wn)
        else:
           self.metadata['analysis_info']['smooth'] = False

        # detect R peaks
        if detect_peaks == True:
            # detect R peaks & calculate inter-beat intevals
            self.calc_RR(smooth, mw_size, upshift, rm_artifacts)

        register_matplotlib_converters()
        
        
    def load_ekg(self, min_dur):
        """ 
        Load ekg data and extract sampling frequency. 
        
        Parameters
        ----------
        min_dur: bool, default: True
            If set to True, will not load files shorter than minimum duration in length 
        """
        
        data = pd.read_csv(self.metadata['file_info']['path'], header = [0, 1], index_col = 0, parse_dates=True)['EKG']
        
        # Check cycle length against 5 minute duration minimum
        cycle_len_secs = (data.index[-1] - data.index[0]).total_seconds()
        if cycle_len_secs < 60*5-1:
            if min_dur == True:
                print('Data is shorter than minimum duration. Cycle will not be loaded.')
                print('--> To load data, set min_dur to False')
                return
            else:
                print('* WARNING: Data is shorter than 5 minutes.')
                self.data = data
        else:
            self.data = data
        
        diff = data.index.to_series().diff()[1:2]
        s_freq = 1000000/diff[0].microseconds
        nans = len(data) - data['Raw'].count()

        self.metadata['file_info']['start_time'] = data.index[0]
        self.metadata['analysis_info'] = {'s_freq': s_freq, 'cycle_len_secs': cycle_len_secs, 
                                        'NaNs(samples)': nans, 'NaNs(secs)': nans/s_freq}

        print('EKG successfully imported.')

    def rms_smooth(self, sm_wn):
        """ Smooth raw data with RMS moving window """
        self.metadata['analysis_info']['smooth'] = True
        self.metadata['analysis_info']['rms_smooth_wn'] = sm_wn
        
        mw = int((sm_wn/1000)*self.metadata['analysis_info']['s_freq'])
        self.data['raw_smooth'] = self.data.Raw.rolling(mw, center=True).mean()


    def set_Rthres(self, smooth, mw_size, upshift):
        """ set R peak detection threshold based on moving average + %signal upshift """
        print('Calculating moving average with {} ms window and a {}% upshift...'.format(mw_size, upshift))
        
        # convert moving window to sample & calc moving average over window
        mw = int((mw_size/1000)*self.metadata['analysis_info']['s_freq'])
        if smooth == False:
            mavg = self.data.Raw.rolling(mw).mean()
            ekg_avg = np.mean(self.data['Raw'])
        elif smooth == True:
            mavg = self.data.raw_smooth.rolling(mw).mean()
            ekg_avg = np.mean(self.data['raw_smooth'])

        # replace edge nans with overall average
        mavg = mavg.fillna(ekg_avg)

        # set detection threshold as +5% of moving average
        upshift_mult = 1 + upshift/100
        det_thres = mavg*upshift_mult
        # insert threshold column at consistent position in df to ensure same color for plotting regardless of smoothing
        self.data.insert(1, 'EKG_thres', det_thres) # can remove this for speed, just keep as series

        self.metadata['analysis_info']['mw_size'] = mw_size
        self.metadata['analysis_info']['upshift'] = upshift

        # create empy series for false detections removed and missed peaks added
        self.rpeak_artifacts = pd.Series()
        self.rpeaks_added = pd.Series()
        self.ibi_artifacts = pd.Series()

    def detect_Rpeaks(self, smooth):
        """ detect R peaks from raw signal """
        print('Detecting R peaks...')

        if smooth == False:
            raw = pd.Series(self.data['Raw'])
        elif smooth == True:
            raw = pd.Series(self.data['raw_smooth'])
        
        thres = pd.Series(self.data['EKG_thres'])
        
        peaks = []
        x = 0
        while x < len(raw):
            if raw[x] > thres[x]:
                roi_start = x
                # count forwards to find down-crossing
                for h in range(x, len(raw), 1):
                    # if value drops below threshold, end ROI
                    if raw[h] < thres[h]:
                        roi_end = h
                        break
                    # else if data ends before dropping below threshold, leave ROI open
                    # & advance h pointer to end loop
                    elif (raw[h] >= thres[h]) and (h == len(raw)-1):
                        roi_end = None
                        h += 1
                        break
                # if ROI is closed, get maximum between roi_start and roi_end
                if roi_end:
                    peak = raw[x:h].idxmax()
                    peaks.append(peak)
                # advance the pointer
                x = h
            else:
                x += 1

        self.rpeaks = raw[peaks]
        print('R peak detection complete')

        # get time between peaks and convert to mseconds
        self.rr = np.diff(self.rpeaks.index)/np.timedelta64(1, 'ms')
        
        # create rpeaks dataframe and add ibi columm
        rpeaks_df = pd.DataFrame(self.rpeaks)
        ibi = np.insert(self.rr, 0, np.NaN)
        rpeaks_df['ibi_ms'] = ibi
        self.rpeaks_df = rpeaks_df

        print('R-R intervals calculated')

    def rm_peaks(self, time):
        """ 
        Examine a second of interest and manually remove artifact peaks
        
        Parameters
        ----------
        time: str format 'hh:mm:ss'
        
        Returns
        -------
        Modified self.rpeaks and self.rpeaks_df attributes. Removed peaks added to 
        self.rpeak_artifacts attribute.
        """
        
        # print all rpeaks in the second of interest
        peak_idxlist = {}
        peak_num = 1
        h, m, s = time.split(':')
        print('id', '\t', 'time', '\t\t\t\t', 'ibi_ms')
        for i, x in enumerate(self.rpeaks_df.index):
            if x.hour == int(h) and x.minute == int(m) and x.second == int(s):
                peak_idxlist[peak_num] = x
                print(peak_num, '\t', x, '\t', self.rpeaks_df['ibi_ms'].loc[x])
                peak_num += 1
        
        # specify the peak to remove
        rm_peak = input('Rpeaks to remove [list ids or None]: ')
        print('\n')
        if rm_peak == 'None':
            print('No peaks removed.')
            # create nn attribute
            self.nn = self.rr
            return
        else:
            rm_peaks = rm_peak.split(',')
            rm_peaks = [int(x) for x in rm_peaks]
            for p in rm_peaks:
                peak_to_rm = pd.Series(self.rpeaks[peak_idxlist[p]])
                peak_to_rm.index = [peak_idxlist[p]]

                # add peak to rpeak_artifacts list
                self.rpeak_artifacts = self.rpeak_artifacts.append(peak_to_rm)
                self.rpeak_artifacts.sort_index(inplace=True)

                # remove peak from rpeaks list & rpeaks dataframe
                self.rpeaks.drop(peak_idxlist[p], inplace=True)
                self.rpeaks_df.drop(peak_idxlist[p], inplace=True)
                print('R peak at ', peak_to_rm.index[0], ' successfully removed.')
                
            # recalculate ibi values
            self.rr = np.diff(self.rpeaks.index)/np.timedelta64(1, 'ms')
            ibi = np.insert(self.rr, 0, np.NaN)
            self.rpeaks_df['ibi_ms'] = ibi
            print('ibi values recalculated.')

        # refresh nn values
        self.nn = self.rr


    def undo_rm_peaks(self, time):
        """ add back incorrectly removed peaks from rm_peaks() method
            NOTE: This is strictly an "undo" method. It is NOT equivalent to add_peaks().
            
            Parameters
            ----------
            time: str (format 'hh:mm:ss')
                second of incorrectly removed peak
            
            Returns
            -------
            Modified self.rpeaks, self.rpeaks_df, self.rr, self.nn, and self.rpeaks_artifacts attributes
        """
        
        if len(self.rpeak_artifacts) == 0:
            print('No rpeaks have been removed.')
            return
        
        # print all rpeaks in the second of interest
        peak_idxlist = {}
        peak_num = 1
        h, m, s = time.split(':')
        print('id', '\t', 'time')
        for i, x in enumerate(self.rpeak_artifacts.index):
            if x.hour == int(h) and x.minute == int(m) and x.second == int(s):
                peak_idxlist[peak_num] = x
                print(peak_num, '\t', x)
                peak_num += 1

        # specify the peak to add back
        add_peak = input('Removed Rpeaks to add back [list ids or None]: ')
        print('\n')
        if add_peak == 'None':
            print('No peaks added.')
            # create nn attribute
            self.nn = self.rr
            return
        else:
            add_peaks = add_peak.split(',')
            add_peaks = [int(x) for x in add_peaks]
            for p in add_peaks:
                peak_to_add = pd.Series(self.rpeak_artifacts[peak_idxlist[p]])
                peak_to_add.index = [peak_idxlist[p]]
        
                # remove peak from rpeak_artifacts list
                self.rpeak_artifacts.drop(labels=peak_to_add.index, inplace=True)
                
                # add peak back to rpeaks list
                self.rpeaks = self.rpeaks.append(peak_to_add)
                self.rpeaks.sort_index(inplace=True)

                # add peak back to rpeaks_df
                self.rpeaks_df.loc[peak_to_add.index[0]] = [peak_to_add[0], np.NaN]
                self.rpeaks_df.sort_index(inplace=True)
                print('Rpeak at ', peak_to_add.index[0], ' successfully replaced.')

            # recalculate ibi values
            self.rr = np.diff(self.rpeaks.index)/np.timedelta64(1, 'ms')
            ibi = np.insert(self.rr, 0, np.NaN)
            self.rpeaks_df['ibi_ms'] = ibi
            print('ibi values recalculated.')

        # refresh nn values
        self.nn = self.rr    

    def add_peak(self, time):
        """ manually add missed peaks 

        Parameters
        ----------
        time: str format 'hh:mm:ss'
        
        Returns
        -------
        Modified self.rpeaks, self.rpeaks_df, self.rr, and self.nn attributes. Added peaks stored in 
        self.rpeaks_added attribute.
        """
        
        # specify time range of missed peak
        h, m, s = time.split(':')
        us_rng = input('Millisecond range of missed peak [min:max]: ').split(':')
        # add zeros bc datetime microsecond precision goes to 6 figures
        us_min, us_max = us_rng[0] + '000', us_rng[1] + '000'
        
        # set region of interest for new peak
        ## can modify this to include smoothing if needed
        roi = []
        for x in self.data.index:
            if x.hour == int(h) and x.minute == int(m) and x.second == int(s) and x.microsecond >= int(us_min) and x.microsecond <= int(us_max):
                roi.append(x)

        # define new rpeak
        if self.metadata['analysis_info']['smooth'] == False:
        	peak_idx = self.data.loc[roi]['Raw'].idxmax()
        	peak_val = self.data['Raw'].loc[peak_idx]
        	new_peak = pd.Series(peak_val, [peak_idx])
        if self.metadata['analysis_info']['smooth'] == True:
            peak_idx = self.data.loc[roi]['raw_smooth'].idxmax()
            peak_val = self.data['raw_smooth'].loc[peak_idx]
            new_peak = pd.Series(peak_val, [peak_idx])

        # add peak to rpeaks list
        self.rpeaks = self.rpeaks.append(new_peak)
        self.rpeaks.sort_index(inplace=True)

        # add peak to rpeaks_df
        self.rpeaks_df.loc[peak_idx] = [peak_val, np.NaN]
        self.rpeaks_df.sort_index(inplace=True)

        # add peak to rpeaks_added list
        self.rpeaks_added = self.rpeaks_added.append(new_peak)
        self.rpeaks_added.sort_index(inplace=True)
        print('New peak added.')

        # recalculate ibi values
        self.rr = np.diff(self.rpeaks.index)/np.timedelta64(1, 'ms')
        ibi = np.insert(self.rr, 0, np.NaN)
        self.rpeaks_df['ibi_ms'] = ibi
        print('ibi values recalculated.')

        # refresh nn values
        self.nn = self.rr

    def undo_add_peak(self, time):
        """ remove an incorrect peak from add_peaks() method
            NOTE: This is strictly an "undo" method. It is NOT equivalent to rm_peaks().
            
            Parameters
            ----------
            time: str (format 'hh:mm:ss')
                second of incorrectly added peak
            
            Returns
            -------
            Modified self.rpeaks, self.rpeaks_df, self.rr, self.nn, and self.rpeaks_added attributes
        """
        
        if len(self.rpeaks_added) == 0:
            print('No rpeaks have been added.')
            return
        
        # print all rpeaks in the second of interest
        peak_idxlist = {}
        peak_num = 1
        h, m, s = time.split(':')
        print('id', '\t', 'time')
        for i, x in enumerate(self.rpeaks_added.index):
            if x.hour == int(h) and x.minute == int(m) and x.second == int(s):
                peak_idxlist[peak_num] = x
                print(peak_num, '\t', x)
                peak_num += 1

        # specify the peak to remove
        rm_peak = input('Added Rpeaks to remove [list ids or None]: ')
        print('\n')
        if rm_peak == 'None':
            print('No peaks removed.')
            # create nn attribute
            self.nn = self.rr
            return
        else:
            rm_peaks = rm_peak.split(',')
            rm_peaks = [int(x) for x in rm_peaks]
            for p in rm_peaks:
                peak_to_rm = pd.Series(self.rpeaks_added[peak_idxlist[p]])
                peak_to_rm.index = [peak_idxlist[p]]
        
                # remove peak from rpeaks_added list
                self.rpeaks_added.drop(labels=peak_to_rm.index, inplace=True)
                
                # remove peak from rpeaks list & rpeaks dataframe
                self.rpeaks.drop(peak_idxlist[p], inplace=True)
                self.rpeaks_df.drop(peak_idxlist[p], inplace=True)
                print('R peak at ', peak_to_rm.index, ' successfully removed.')

            # recalculate ibi values
            self.rr = np.diff(self.rpeaks.index)/np.timedelta64(1, 'ms')
            ibi = np.insert(self.rr, 0, np.NaN)
            self.rpeaks_df['ibi_ms'] = ibi
            print('ibi values recalculated.')

        # refresh nn values
        self.nn = self.rr    

    def rm_ibi(self, thres = 3000):
        """ Manually remove IBI's corresponding to missing data (due to cleaning) or missed beats that can't be
            manually added with ekg.add_peak() method
            NOTE: This step must be completed LAST, after removing any false peaks and adding any missed peaks
        
            Parameters
            ----------
            thres: int
                threshold in milliseconds for automatic IBI removal
        """
        
        # check for extra-long IBIs & option to auto-remove
        if any(self.rpeaks_df['ibi_ms'] > thres):
            print(f'IBIs greater than {thres} milliseconds detected')
            rm = input('Automatically remove? [y/n]: ')
            
            if rm.casefold() == 'y':
                # get indices of ibis greater than threshold
                rm_idx = [i for i, x in enumerate(self.nn) if x > thres]
                # replace ibis w/ NaN
                self.nn[rm_idx] = np.NaN
                print('{} IBIs removed.'.format(len(rm_idx), thres))
                
                # add ibi to ibi_artifacts list
                df_idx = [x+1 for x in rm_idx] # shift indices by 1 to correspond with df indices
                ibis_rmvd = pd.Series(self.rpeaks_df['ibi_ms'].iloc[df_idx])
                self.ibi_artifacts = self.ibi_artifacts.append(ibis_rmvd)
                self.ibi_artifacts.sort_index(inplace=True)
                print('ibi_artifacts series updated.') 

                # update rpeaks_df
                ibi = np.insert(self.nn, 0, np.NaN)
                self.rpeaks_df['ibi_ms'] = ibi
                print('R peaks dataframe updated.\n')    
        
        else:
            print(f'All ibis are less than {thres} milliseconds.')

        # option to specify which IBIs to remove
        rm = input('Manually remove IBIs? [y/n]: ')
        if rm.casefold() == 'n':
            print('Done.')
            return
        elif rm.casefold() == 'y':
            # print IBI list w/ IDs
            print('Printing IBI list...\n')
            print('ID', '\t', 'ibi end time', '\t', 'ibi_ms')
            for i, x in enumerate(self.rpeaks_df.index[1:]):
                    print(i, '\t',str(x)[11:-3], '\t', self.rpeaks_df['ibi_ms'].loc[x])
            rm_ids = input('IDs to remove [list or None]: ')
            if rm_ids.casefold() == 'none':
                print('No ibis removed.')
                return
            else:
                # replace IBIs in nn array
                rm_ids = [int(x) for x in rm_ids.split(',')]
                self.nn[rm_ids] = np.NaN
                print('{} IBIs removed.'.format(len(rm_ids)))

                # add ibi to ibi_artifacts list
                df_idx = [x+1 for x in rm_ids] # shift indices by 1 to correspond with df indices
                ibis_rmvd = pd.Series(self.rpeaks_df['ibi_ms'].iloc[df_idx])
                self.ibi_artifacts = self.ibi_artifacts.append(ibis_rmvd)
                self.ibi_artifacts.sort_index(inplace=True)
                print('ibi_artifacts series updated.')
                
                # update self.rpeaks_df
                ibi = np.insert(self.nn, 0, np.NaN)
                self.rpeaks_df['ibi_ms'] = ibi
                print('R peaks dataframe updated.\nDone.')


    def calc_RR(self, smooth, mw_size, upshift, rm_artifacts):
        """ Detect R peaks and calculate R-R intervals """
        
        # set R peak detection parameters
        self.set_Rthres(smooth, mw_size, upshift)
        # detect R peaks & make RR tachogram
        self.detect_Rpeaks(smooth)
        # remove artifacts
        if rm_artifacts == True:
            self.rm_artifacts()

    def export_RR(self, savedir):
        """ Export R peaks and RR interval data to .txt files """

        # set save directory
        if savedir is None:
            savedir = os.getcwd()
            chngdir = input('Files will be saved to ' + savedir + '. Change save directory? [Y/N] ')
            if chngdir == 'Y':
                savedir = input('New save directory: ')
                if not os.path.exists(savedir):
                    createdir = input(savedir + ' does not exist. Create directory? [Y/N] ')
                    if createdir == 'Y':
                        os.makedirs(savedir)
                    else:
                        savedir = input('Try again. Save directory: ')
                        if not os.path.exists(savedir):
                            print(savedir + ' does not exist. Aborting. ')
                            return
        elif not os.path.exists(savedir):
            print(savedir + ' does not exist. Creating directory...')
            os.makedirs(savedir)
        else:
            print('Files will be saved to ' + savedir)
        
        # save rpeak_artifacts list
        try:
            self.rpeak_artifacts
        except AttributeError:
            cont = input('EKG object has no artifacts attribute. Continue save without cleaning? [y/n]: ')
            if cont == 'y': 
                pass
            elif cont == 'n':
                print('Save aborted.')
                return
        else:
            savearts = self.metadata['file_info']['fname'].split('.')[0] + '_rpeak_artifacts.txt'
            art_file = os.path.join(savedir, savearts)
            self.rpeak_artifacts.to_csv(art_file, header=False)
            print('R peak artifacts exported.')

        # save rpeaks_added list
        savename = self.metadata['file_info']['fname'].split('.')[0] + '_rpeaks_added.txt'
        savefile = os.path.join(savedir, savename)
        self.rpeaks_added.to_csv(savefile, header=False)
        print('R peak additions exported.')

        # save R peak detections
        savepeaks = self.metadata['file_info']['fname'].split('.')[0] + '_rpeaks.txt'
        peaks_file = os.path.join(savedir, savepeaks)
        self.rpeaks.to_csv(peaks_file, header=False)
        print('R peaks exported.')

        # save ibi_artifact list
        savename = self.metadata['file_info']['fname'].split('.')[0] + '_ibi_artifacts.txt'
        savefile = os.path.join(savedir, savename)
        self.ibi_artifacts.to_csv(savefile, header=False)
        print('IBI artifacts exported.')

        # save RR intervals
        rr_header = 'R peak detection mw_size = {} & upshift = {}'.format(self.metadata['analysis_info']['mw_size'], self.metadata['analysis_info']['upshift'])
        saverr = self.metadata['file_info']['fname'].split('.')[0] + '_rr.txt'
        rr_file = os.path.join(savedir, saverr)
        np.savetxt(rr_file, self.rr, header=rr_header, fmt='%.0f', delimiter='\n')
        print('rr intervals exported.')

        # save NN intervals, if exists
        try: 
            self.nn
        except AttributeError:
            print('EKG object has no nn attribute. Only exporting r peaks and rr intervals.')
            pass
        else:
            # set # of artifacts removed for header
            try:
                self.rpeak_artifacts
            except AttributeError:
                arts_len = 0
            else:
                arts_len = len(self.rpeak_artifacts) + len(self.ibi_artifacts)
            nn_header = 'R peak detection mw_size = {} & upshift = {}.\nTotal artifacts removed = {} ( {} false peaks + {} false ibis).'.format(self.metadata['analysis_info']['mw_size'], self.metadata['analysis_info']['upshift'], arts_len, len(self.rpeak_artifacts), len(self.ibi_artifacts))
            savenn = self.metadata['file_info']['fname'].split('.')[0] + '_nn.txt'
            nn_file = os.path.join(savedir, savenn)
            np.savetxt(nn_file, self.nn, header=nn_header, fmt='%.0f', delimiter='\n')
            print('nn intervals exported.')

        print('Done.')



    def calc_tstats(self, itype):
        """ Calculate commonly used time domain HRV statistics. Min/max HR is determined over 5 RR intervals 

            Params
            ------
            itype: str, 
                Interval type (Options:'rr', 'nn'). 'rr' is uncleaned data. 'nn' is normal intervals (cleaned)
        """   
        print('Calculating time domain statistics...')

        if itype == 'rr':
            ii = self.rr
            ii_diff = np.diff(self.rr)
            ii_diffsq = ii_diff**2
            self.rr_diff = ii_diff
            self.rr_diffsq = ii_diffsq
        
        elif itype == 'nn':
            # remove np.NaNs for calculations
            ii = self.nn[~np.isnan(self.nn)]
            ii_diff = np.diff(ii)
            ii_diffsq = ii_diff**2
            self.nn_diff = ii_diff
            self.nn_diffsq = ii_diffsq

        # heartrate in bpm 
        hr_avg = 60/np.mean(ii)*1000
        
        rollmean_ii = pd.Series(ii).rolling(5).mean()
        mx_ii, mn_ii = np.nanmax(rollmean_ii), np.nanmin(rollmean_ii)
        hr_max = 60/mn_ii*1000
        hr_min = 60/mx_ii*1000


        # inter-beat interval & SD (ms)
        ibi = np.mean(ii)
        sdnn = np.std(ii, ddof=1)

        # SD & RMS of differences between successive II intervals (ms)
        sdsd = np.std(ii_diff)
        rmssd = np.sqrt(np.mean(ii_diffsq))

        # pNN20 & pNN50
        pxx20 = sum(np.abs(ii_diff) >= 20.0)/(len(ii_diff)-1) *100
        pxx50 = sum(np.abs(ii_diff) >= 50.0)/(len(ii_diff)-1) *100

        self.time_stats = {'linear':{'HR_avg': hr_avg, 'HR_max': hr_max, 'HR_min': hr_min, 'IBI_mean': ibi,
                                    'SDNN': sdnn, 'RMSSD': rmssd, 'pXX20': pxx20, 'pXX50': pxx50},
                            }
        print('Time domain stats stored in obj.time_stats\n')

    
    def interpolateII(self, itype):
        """ Resample tachogram to original sampling frequency (since RRs are not evenly spaced)
            and interpolate for power spectral estimation 
            *Note: adapted from pyHRV

            Params
            ------
            itype: str
                interval type (options: 'rr', 'nn')
        """
        # specify data
        if itype == 'rr':
            ii = self.rr
        elif itype == 'nn':
            # remove np.NaNs for calculations
            ii = self.nn[~np.isnan(self.nn)]

        # interpolate
        fs = self.metadata['analysis_info']['s_freq']
        t = np.cumsum(ii)
        t -= t[0]
        f_interp = sp.interpolate.interp1d(t, ii, 'cubic')
        t_interp = np.arange(t[0], t[-1], 1000./fs)
        self.ii_interp = f_interp(t_interp)
        self.metadata['analysis_info']['s_freq_interp'] = self.metadata['analysis_info']['s_freq']


    def calc_psd_welch(self, itype, window):
        """ Calculate welch power spectral density 

            Params
            ------
            itype: str
                interval type (options: 'rr', 'nn')
            window: str
                windowing function. options from scipy.signal welch. (wrapper default: 'hamming')
        """
        
        self.metadata['analysis_info']['psd_method'] = 'welch'
        self.metadata['analysis_info']['psd_window'] = window

        # specify data
        if itype == 'rr':
            ii = self.rr
        elif itype == 'nn':
            ii = self.nn[~np.isnan(self.nn)]
        
        # set nfft to guidelines of power of 2 above len(data), min 256 (based on MATLAB guidelines)
        nfft = max(256, 2**(int(np.log2(len(self.ii_interp))) + 1))
        
        # Adapt 'nperseg' according to the total duration of the II series (5min threshold = 300000ms)
        if max(np.cumsum(ii)) < 300000:
            nperseg = nfft
        else:
            nperseg = 300
        
        # default overlap = 50%
        f, Pxx = welch(self.ii_interp, fs=4, window=window, scaling = 'density', nfft=nfft, 
                        nperseg=nperseg)
        self.psd_welch = {'freqs':f, 'pwr': Pxx, 'nfft': nfft, 'nperseg': nperseg}


    def calc_psd_mt(self, bandwidth):
        """ Calculate multitaper power spectrum 

            Params
            ------
            bandwidth: float
                frequency resolution (NW)

            Returns
            -------
            psd_mt: dict
                'freqs': ndarray
                'psd': ndarray. power spectral density in (V^2/Hz). 10log10 to convert to dB.

        """
        self.metadata['analysis_info']['psd_method'] = 'multitaper'
        self.metadata['analysis_info']['psd_bandwidth'] = bandwidth
        sf_interp = self.metadata['analysis_info']['s_freq_interp']

        pwr, freqs = psd_array_multitaper(self.ii_interp, sf_interp, adaptive=True, 
                                            bandwidth=bandwidth, normalization='full', verbose=0)
        self.psd_mt = {'freqs': freqs, 'pwr': pwr}
        self.metadata['analysis_info']['psd_method'] = 'multitaper'

    def calc_fbands(self, method, bands):
        """ Calculate different frequency band measures 
            TO DO: add option to change bands
            Note: modified from pyHRV
            * normalized units are normalized to total lf + hf power, according to Heathers et al. (2014)
        """
        if method is None:
            method = input('Please enter PSD method (options: "welch", "mt"): ')
        if method == 'welch':
            psd = self.psd_welch
        elif method == 'mt':
            psd = self.psd_mt
        
        # set frequency bands
        if bands is None:
            ulf = None
            vlf = (0.000, 0.04)
            lf = (0.04, 0.15)
            hf = (0.15, 0.4)
            args = (ulf, vlf, lf, hf)
            names = ('ulf', 'vlf', 'lf', 'hf')
        freq_bands = dict(zip(names, args))
        #self.freq_bands = freq_bands
        
        # get indices and values for frequency bands in calculated spectrum
        fband_vals = {}
        for key in freq_bands.keys():
            fband_vals[key] = {}
            if freq_bands[key] is None:
                fband_vals[key]['idx'] = None
                fband_vals[key]['pwr'] = None
            else:
                # lower limit not inclusive
                fband_vals[key]['idx'] = np.where((freq_bands[key][0] < psd['freqs']) & (psd['freqs'] <= freq_bands[key][1]))[0]
                fband_vals[key]['pwr'] = psd['pwr'][fband_vals[key]['idx']]
                
        self.psd_fband_vals = fband_vals

        # calculate stats 
        total_pwr = sum(filter(None, [np.sum(fband_vals[key]['pwr']) for key in fband_vals.keys()]))
        freq_stats = {'totals':{'total_pwr': total_pwr}}
        # by band
        for key in freq_bands.keys():
            freq_stats[key] = {}
            freq_stats[key]['freq_range'] = str(freq_bands[key])
            if freq_bands[key] is None:
                freq_stats[key]['pwr_ms2'] = None
                freq_stats[key]['pwr_peak'] = None
                freq_stats[key]['pwr_log'] = None
                freq_stats[key]['pwr_%'] = None
                freq_stats[key]['pwr_nu'] = None
            else:
                freq_stats[key]['pwr_ms2'] = np.sum(fband_vals[key]['pwr'])
                peak_idx = np.where(fband_vals[key]['pwr'] == max(fband_vals[key]['pwr']))[0][0]
                freq_stats[key]['pwr_peak'] = psd['freqs'][fband_vals[key]['idx'][peak_idx]]
                freq_stats[key]['pwr_log'] = np.log(freq_stats[key]['pwr_ms2'])
                freq_stats[key]['pwr_%'] = freq_stats[key]['pwr_ms2']/freq_stats['totals']['total_pwr']*100
        
        # add normalized units to lf & hf bands
        for key in ['lf', 'hf']:
            freq_stats[key]['pwr_nu'] = freq_stats[key]['pwr_ms2']/(freq_stats['lf']['pwr_ms2'] + freq_stats['hf']['pwr_ms2'])*100
        # add lf/hf ratio
        freq_stats['totals']['lf/hf'] = freq_stats['lf']['pwr_ms2']/freq_stats['hf']['pwr_ms2']
        
        self.freq_stats = freq_stats


    def calc_fstats(self, itype, method, bandwidth, window):
        """
        Calculate commonly used frequency domain HRV statistics.

        Parameters
        ----------
        itype : str {'rr, 'nn'}
            Interval type.
            'rr' is uncleaned data. 'nn' is normal intervals (cleaned).
        method : str, {'mt, 'welch'}
            Method to compute power spectra.
            'mt' is multitaper.
        bandwith : float
            Bandwidth for multitaper power spectral estimation.
        window : str
            Window to use for welch FFT. See mne.time_frequency.psd_array_multitaper for options.

        See Also
        --------
        EKG.calc_tstats : Calculate commonly used time domain HRV statistics.
        EKG.hrv_stats : Calculate both time and frequency domain HRV statistics on IBI object.
        """
        # resample & interpolate tachogram
        print('Interpolating and resampling tachogram...')
        self.interpolateII(itype)
       
       # calculate power spectrum
        print('Calculating power spectrum...')
        if method == 'mt':
            self.calc_psd_mt(bandwidth)
        elif method == 'welch':
            self.calc_psd_welch(itype, window)
        
        #calculate frequency domain statistics
        print('Calculating frequency domain measures...')
        self.calc_fbands(method)
        print('Frequency measures stored in obj.freq_stats\n')



    def hrv_stats(self, itype='nn', nn_file=None, method='mt', bandwidth=0.01, window='hamming'):
        """ Calculate all statistics on IBI object 

            TO DO: Add freq_stats arguments to hrv_stats params? 

            Parameters
            ----------
            nn_file: str
                path to csv file containing cleaned nn values, if nn values were
                previously exported
        """

        self.metadata['analysis_info']['itype'] = itype
        
        # load nn attribute if data was cleaned previously
        if itype == 'nn' and nn_file is not None:
            # read in metadata
            with open(nn_file, 'r') as f:
                line1 = [x for x in f.readline().split(' ')]
                line2 = [x for x in f.readline().split(' ')]
                self.metadata['analysis_info']['mw_size'] = float(line1[6])
                self.metadata['analysis_info']['upshift'] = float(line1[10].split('.\n')[0])
                self.metadata['analysis_info']['artifacts_rmvd'] = int(line2[5])
            # load nn intervals
            self.nn = np.loadtxt(nn_file)

        else:
            self.metadata['analysis_info']['artifacts_rmvd'] = str(str(len(self.rpeak_artifacts)) + ' false peaks (removed); ' + str(len(self.rpeaks_added)) + ' missed peaks (added); ' + str(len(self.ibi_artifacts)) + ' ibis removed (from NaN data)')

        # calculate statistics
        self.calc_tstats(itype)
        self.calc_fstats(itype, method, bandwidth, window)
        
        print('Done.')

    def to_spreadsheet(self, spreadsheet, savedir):
        """ Append calculations as a row in master spreadsheet. Creates new spreadsheet
            if output file does not exist. 
            
            Parameters
            ----------
            ekg: EKG object
            spreadsheet: .csv
                output file (new or existing)
        """
        
        # this is from before division to two classes. 'data' and 'rpeaks' arrays shouldn't exist in IBI object.
        arrays = ['data', 'rpeaks', 'rr', 'rr_diff', 'rr_diffsq', 'rpeak_artifacts', 'rpeaks_added', 'ibi_artifacts',
        'rpeaks_df', 'nn', 'nn_diff', 'nn_diffsq', 'rr_arts', 'ii_interp', 'psd_mt', 'psd_welch', 'psd_fband_vals']
        data = {k:v for k,v in vars(self).items() if k not in arrays}
        
        reform = {(level1_key, level2_key, level3_key): values
                    for level1_key, level2_dict in data.items()
                    for level2_key, level3_dict in level2_dict.items()
                    for level3_key, values      in level3_dict.items()}
        
        df = pd.DataFrame(reform, index=[0])
        df.set_index([('metadata', 'file_info', 'in_num'), ('metadata', 'file_info', 'start_time')], inplace=True)
        savename = os.path.join(savedir, spreadsheet)
        
        if os.path.exists(savename):
            with open(savename, 'a') as f:
                df.to_csv(f, header=False, line_terminator='\n')
            print('Data added to {}'.format(spreadsheet))
        else:
            with open(savename, 'a') as f:
                df.to_csv(f, header=True, line_terminator='\n')
            print('{} does not exist. Data saved to new spreadsheet'.format(spreadsheet))

    def to_report(self, savedir=None, fmt='txt'):
        """ export statistics as a csv report 
            TO DO: add line to check if nn exists

            fmt: str (default: 'txt')
                output format (options: 'txt', 'json')
        """
        # set save directory
        if savedir is None:
            savedir = os.getcwd()
            chngdir = input('Files will be saved to ' + savedir + '. Change save directory? [Y/N] ')
            if chngdir == 'Y':
                savedir = input('New save directory: ')
                if not os.path.exists(savedir):
                    createdir = input(savedir + ' does not exist. Create directory? [Y/N] ')
                    if createdir == 'Y':
                        os.makedirs(savedir)
                    else:
                        savedir = input('Try again. Save directory: ')
                        if not os.path.exists(savedir):
                            print(savedir + ' does not exist. Aborting. ')
                            return
        elif not os.path.exists(savedir):
            print(savedir + ' does not exist. Creating directory...')
            os.makedirs(savedir)
        else:
            print('Files will be saved to ' + savedir)
        
        # export everything that isn't a dataframe, series, or array    
        arrays = ['data', 'rpeaks', 'rr', 'rr_diff', 'rr_diffsq', 'rpeak_artifacts', 'rpeaks_added', 'ibi_artifacts', 'rpeaks_df', 'nn', 'nn_diff', 'nn_diffsq', 'rr_arts', 'ii_interp', 'psd_mt', 'psd_fband_vals']
        data = {k:v for k,v in vars(self).items() if k not in arrays}
        
        # set savename info
        if 'epoch' in self.metadata['file_info'].keys():
            saveinfo = ('_'.join((self.metadata['file_info']['fname'].split('_')[:6]))).split('.')[0]
        else:
            saveinfo = ('_'.join((self.metadata['file_info']['fname'].split('_')[:5]))).split('.')[0]

        # save calculations
        if fmt == 'txt':
            savename = saveinfo + '_HRVstats.txt'
            file = os.path.join(savedir, savename)
            with open(file, 'w') as f:
                for k, v in data.items():
                    if type(v) is not dict:
                        line = k+' '+str(v) + '\n'
                        f.write(line)
                    elif type(v) is dict:
                        line = k + '\n'
                        f.write(line)
                        for kx, vx in v.items():
                            if type(vx) is not dict:
                                line = '\t'+ kx + ' ' + str(vx) + '\n'
                                f.write(line)
                            else:
                                line = '\t' + kx + '\n'
                                f.write(line)
                                for kxx, vxx in vx.items():
                                    line = '\t\t' + kxx + ' ' + str(vxx) + '\n'
                                    f.write(line)
        elif fmt == 'json':
            savename = saveinfo + '_HRVstats_json.txt'
            file = os.path.join(savedir, savename)
            with open(file, 'w') as f:
                json.dump(data, f, indent=4)   
        
        # save power spectra for later plotting
        try: 
            self.psd_mt
        except AttributeError: 
            pass
        else:
            savepsd = saveinfo + '_psd_mt.txt'
            psdfile = os.path.join(savedir, savepsd)
            psd_mt_df = pd.DataFrame(self.psd_mt)
            psd_mt_df.to_csv(psdfile, index=False)
        try:
            self.psd_welch
        except AttributeError: 
            pass
        else:
            savepsd = saveinfo + '_psd_welch.txt'
            psdfile = os.path.join(savedir, savepsd)
            psd_mt_df = pd.DataFrame(self.psd_mt)
            psd_mt_df.to_csv(psdfile, index=False)


    ## plotting methods ##
    def plotpeaks(self, rpeaks=True, ibi=True, thres = True):
        """ plot EKG class instance """
        # set number of panels
        if ibi == True:
            plots = ['ekg', 'ibi']
            if thres == True:
                data = [self.data, self.rpeaks_df['ibi_ms']]
            if thres == False:
                if self.metadata['analysis_info']['smooth'] == False:
                    data = [self.data['Raw'], self.rpeaks_df['ibi_ms']]
                if self.metadata['analysis_info']['smooth'] == True:
                    data = [self.data[['Raw', 'raw_smooth']], self.rpeaks_df['ibi_ms']]
            
        else:
            plots = ['ekg']
            if thres == True:
                data = [self.data]
            if thres == False:
                if self.metadata['analysis_info']['smooth'] == False:
                    data = [self.data['Raw']]
                if self.metadata['analysis_info']['smooth'] == True:
                    data = [self.data[['Raw', 'raw_smooth']]]

        fig, axs = plt.subplots(len(plots), 1, sharex=True, figsize = [9.5, 6])
        
        if len(plots) > 1:
            for dat, ax, plot in zip(data, axs, plots):
                if plot == 'ekg' and rpeaks == True:
                    ax.plot(dat, zorder = 1)
                    ax.scatter(self.rpeaks.index, self.rpeaks.values, color='red', zorder = 2)
                    ax.set_ylabel('EKG (mV)')
                    if thres == True:
                        if self.metadata['analysis_info']['smooth'] == True:
                            ax.legend(('raw data', 'threshold line', 'smoothed data', 'rpeak'), fontsize = 'small')
                        else:
                            ax.legend(('raw data', 'threshold line', 'rpeak'), fontsize = 'small')
                    else:
                        if self.metadata['analysis_info']['smooth'] == True:
                            ax.legend(('raw data', 'smoothed data', 'rpeak'), fontsize = 'small')
                        else:
                            ax.legend(('raw data', 'rpeak'), fontsize = 'small')


                elif plot == 'ibi':
                    ax.plot(dat, color='grey', marker='.', markersize=8, markerfacecolor=(0, 0, 0, 0.8), markeredgecolor='None')
                    ax.set_ylabel('Inter-beat interval (ms)')
                    ax.set_xlabel('Time')
                ax.margins(x=0)
                # show microseconds for mouse-over
                ax.format_xdata = lambda d: mdates.num2date(d).strftime('%H:%M:%S.%f')[:-3]
        else:
            for dat, plot in zip(data, plots):
                if plot == 'ekg' and rpeaks == True:
                    axs.plot(dat, zorder = 1)
                    axs.scatter(self.rpeaks.index, self.rpeaks.values, color='red', zorder = 2)
                    axs.set_ylabel('EKG (mV)')
                    axs.set_xlabel('Time')
                axs.margins(x=0)
                # show microseconds for mouse-over
                axs.format_xdata = lambda d: mdates.num2date(d).strftime('%H:%M:%S.%f')[:-3]




    def plotPS(self, method='mt', dB=False, bands=True, save=True, savedir=None):
        """ Plot power spectrum """
        
         # set title
        title = self.metadata['file_info']['in_num'] + ' ' + self.metadata['file_info']['start_date'] + '\n' + self.metadata['file_info']['sleep_stage'] + ' ' + self.metadata['file_info']['cycle']
        try:
            n.metadata['file_info']['epoch']
        except:
            pass
        else:
            title = title + ' ' +  n.metadata['file_info']['epoch']

        # set data to plot
        if method == 'mt':
            psd = self.psd_mt
        elif method == 'welch':
            psd = self.psd_welch
        
        # transform units
        if dB == True:
            pwr = 10 * np.log10(psd['pwr'])
            ylabel = 'Power spectral density (dB)'
        else:
            pwr = psd['pwr']/1e6 # convert to seconds
            ylabel = 'Power spectral density (s^2/Hz)'
        
        fig, ax = plt.subplots()
        
        # plot just spectrum
        if bands == False:
            ax.plot(psd['freqs'], pwr)
        
        # or plot spectrum colored by frequency band
        elif bands == True:
            # use matplotlib.patches.Patch to make objects for legend w/ data
            ax.plot(psd['freqs'], pwr, color='black')
            
            yline = SG.LineString(list(zip(psd['freqs'],pwr)))
            #ax.plot(yline, color='black')
            
            colors = [None, 'yellow', 'orange', 'tomato']
            for (key, value), color in zip(self.psd_fband_vals.items(), colors):
                if value['idx'] is not None:
                    # get intercepts & plot vertical lines for bands
                    xrange = [float(x) for x in self.freq_stats[key]['freq_range'][1:-1].split(",")] 
                    xline = SG.LineString([(xrange[1], min(pwr)), (xrange[1], max(pwr))])
                    coords = np.array(xline.intersection(yline))            
                    ax.vlines(coords[0], 0, coords[1], colors='black', linestyles='dotted')
                    
                    # fill spectra by band
                    ax.fill_between(psd['freqs'], pwr, where = [xrange[0] <= x <=xrange[1] for x in psd['freqs']], 
                                    facecolor=color, alpha=.6)    
            
        ax.set_xlim(0, 0.4)
        ax.margins(y=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(ylabel)
        plt.suptitle(title)

        if save:
            if savedir is None:
                print('ERROR: File not saved. Please specify savedir argument.')
            else:
                savename = os.path.join(savedir, fname.split('.')[0]) + '_psd.png'
                fig.savefig(savename, dpi=300)

        return fig
