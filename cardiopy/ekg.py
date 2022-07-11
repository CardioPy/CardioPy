"""
This file contains methods to visualize EKG data, clean EKG data and run EKG analyses.

Classes
-------
EKG

Notes
-----
All R peak detections should be manually inspected with EKG.plotpeaks method and
false detections manually removed with rm_peak method. After rpeak examination, 
NaN data can be accounted for by removing false IBIs with rm_ibi method.
"""

import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np 
import os
import pandas as pd 
import scipy as sp
import statistics
import biosignalsnotebooks as bsnb

from scipy import interpolate
from numpy import linspace, diff, zeros_like, arange, array
from mne.time_frequency import psd_array_multitaper
from pandas.plotting import register_matplotlib_converters
from scipy.signal import welch
from scipy.stats.distributions import chi2

class EKG:
    """
    Run EKG analyses including cleaning and visualizing data.
    
    Attributes
    ----------
    metadata : nested dict
        File information and analysis information.
        Format {str:{str:val}} with val being str, bool, float, int or pd.Timestamp.
    data : pd.DataFrame
        Raw data of the EKG signal (mV) and the threshold line (mV) at each sampled time point.
    rpeak_artifacts : pd.Series
        False R peak detections that have been removed.
    rpeaks_added : pd.Series
        R peak detections that have been added.
    ibi_artifacts : pd.Series
        Interbeat interval data that has been removed.
    rpeaks : pd.Series
        Cleaned R peaks data without removed peaks and with added peaks.
    rr : np.ndarray
        Time between R peaks (ms).
    nn : np.ndarray
        Cleaned time between R peaks (ms) without removed interbeat interval data.
    rpeaks_df : pd.DataFrame
        Raw EKG value (mV) and corresponding interbeat interval leading up to the data point (ms) at each sampled point.
    """

    def __init__(self, fname, fpath, polarity='positive', min_dur=True, epoched=True, smooth=False, sm_wn=30, mw_size=90, upshift=3.5, 
        rms_align='right', detect_peaks=True, pan_tompkins=True):
        """
        Initialize raw EKG object.

        Parameters
        ----------
        fname : str
            Filename.
        fpath : str
            Path to file.
        polarity: str, default 'positive'
            polarity of the R-peak deflection. Options: 'positive', 'negative'
        min_dur : bool, default True
            Only load files that are >= 5 minutes long.
        epoched : bool, default True
            Whether file was epoched using ioeeg.
        smooth : bool, default False
            Whether raw signal should be smoothed before peak detections. Set True if raw data has consistent high frequency noise
            preventing accurate peak detection.
        sm_wn : float, default 30
            Size of moving window for rms smoothing preprocessing (milliseconds).
        mw_size : float, default 100
            Moving window size for R peak detection (milliseconds).
        upshift : float, default 3.5
            Detection threshold upshift for R peak detection (% of signal).
        rms_align: str, default 'right'
            whether to align the mean to the right or left side of the moving window [options: 'right', 'left']
        rm_artifacts : bool, default False
            Apply IBI artifact removal algorithm.
        detect_peaks : bool, default True
            Option to detect R peaks and calculate interbeat intervals.
        pan_tompkins : bool, default True
            Option to detect R peaks using automatic pan tompkins detection method

        Returns
        -------
        EKG object. Includes R peak detections and calculated inter-beat intervals if detect_peaks is set to True.
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
                                'rpeak_polarity': polarity,
                                'start_date': start_date,
                                'sleep_stage': slpstage,
                                'cycle': cycle
                                }
                        }
        if epoched == True:
            self.metadata['file_info']['epoch'] = epoch
        
        # load the ekg
        self.load_ekg(min_dur)

        # flip the polarity if R peaks deflections are negative
        if polarity == 'negative':
            self.data = self.data*-1

        if smooth == True:
            self.rms_smooth(sm_wn)
        else:
           self.metadata['analysis_info']['smooth'] = False
           self.metadata['analysis_info']['rms_smooth_wn'] = 'N/A'

        # create empty series for false detections removed and missed peaks added
        self.rpeak_artifacts = pd.Series()
        self.rpeaks_added = pd.Series()
        self.ibi_artifacts = pd.Series()

        # detect R peaks
        if detect_peaks == True:
            if pan_tompkins == True:
                self.metadata['analysis_info']['mw_size'] = 'N/A'
                self.metadata['analysis_info']['upshift'] = 'N/A'
                self.metadata['analysis_info']['rms_align'] = 'N/A'
                self.pan_tompkins_detector()
            # detect R peaks & calculate inter-beat intevals
            else: 
                self.calc_RR(smooth, mw_size, upshift, rms_align)
                self.metadata['analysis_info']['pan_tompkins'] = False

        # initialize the nn object
        self.nn = self.rr

        register_matplotlib_converters()
        
        
    def load_ekg(self, min_dur):
        """ 
        Load EKG data from csv file and extract metadata including sampling frequency, cycle length, start time and NaN data.
        
        Parameters
        ----------
        min_dur : bool, default True
            If set to True, will not load files shorter than the minimum duration length of 5 minutes.
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

        # Set metadata 
        self.metadata['file_info']['start_time'] = data.index[0]
        self.metadata['analysis_info'] = {'s_freq': s_freq, 'cycle_len_secs': cycle_len_secs, 
                                        'NaNs(samples)': nans, 'NaNs(secs)': nans/s_freq}

        print('EKG successfully imported.')

    def rms_smooth(self, sm_wn):
        """ 
        Smooth raw data with root mean square (RMS) moving window.

        Reduce noise leading to false R peak detections.

        Parameters
        ----------
        sm_wn : float, default 30
            Size of moving window for RMS smoothing preprocessing (ms).
        """
        self.metadata['analysis_info']['smooth'] = True
        self.metadata['analysis_info']['rms_smooth_wn'] = sm_wn
        
        mw = int((sm_wn/1000)*self.metadata['analysis_info']['s_freq'])
        self.data['raw_smooth'] = self.data.Raw.rolling(mw, center=True).mean()


    def set_Rthres(self, smooth, mw_size, upshift, rms_align):
        """
        Set R peak detection threshold based on moving average shifted up by a percentage of the EKG signal.
        
        Parameters
        ----------
        smooth : bool, default False
            If set to True, raw EKG data will be smoothed using RMS smoothing window.
        mw_size : float, default 100
            Time over which the moving average of the EKG signal will be taken to calculate the R peak detection threshold (ms).
        upshift : float, default 3.5
            Percentage of EKG signal that the moving average will be shifted up by to set the R peak detection threshold.
        rms_align: str, default 'right'
            whether to align the mean to the right or left side of the moving window [options: 'right', 'left']

        See Also
        --------
        EKG.rms_smooth : Smooth raw EKG data with root mean square (RMS) moving window.

        """
        print('Calculating moving average with {} ms window and a {}% upshift...'.format(mw_size, upshift))
        
        # convert moving window to sample & calc moving average over window
        mw = int((mw_size/1000)*self.metadata['analysis_info']['s_freq'])
        #if smooth is true have the moving average calculated based off of smoothed data
        if smooth == False:
            mavg = self.data.Raw.rolling(mw).mean()
            ekg_avg = np.mean(self.data['Raw'])
        elif smooth == True:
            mavg = self.data.raw_smooth.rolling(mw).mean()
            ekg_avg = np.mean(self.data['raw_smooth'])

        if rms_align == 'left':
            # get the number of NaNs and shift the average left by that amount
            mavg = mavg.shift(-mavg.isna().sum())
        # replace edge nans with overall average
        mavg = mavg.fillna(ekg_avg)

        # set detection threshold as +upshift% of moving average
        upshift_perc = upshift/100
        det_thres = mavg + np.abs(mavg*upshift_perc)
        # insert threshold column at consistent position in df to ensure same color for plotting regardless of smoothing
        self.data.insert(1, 'EKG_thres', det_thres) # can remove this for speed, just keep as series

        #set metadata
        self.metadata['analysis_info']['mw_size'] = mw_size
        self.metadata['analysis_info']['upshift'] = upshift
        self.metadata['analysis_info']['rms_align'] = rms_align

    def detect_Rpeaks(self, smooth):
        """ 
        Detect R peaks of raw or smoothed EKG signal based on detection threshold. 
        Parameters
        ----------
        smooth : bool, default False
        If set to True, raw EKG data is smoothed using a RMS smoothing window.

        See Also
        --------
        EKG.rms_smooth : Smooth raw EKG data with root mean square (RMS) moving window
        EKG.set_Rthres : Set R peak detection threshold based on moving average shifted up by a percentage of the EKG signal.
        """
        print('Detecting R peaks...')
        #Use the raw data or smoothed data depending on bool smooth
        if smooth == False:
            raw = pd.Series(self.data['Raw'])
        elif smooth == True:
            raw = pd.Series(self.data['raw_smooth'])
        
        thres = pd.Series(self.data['EKG_thres'])
        #create empty peaks list
        peaks = []
        x = 0
        #Within the length of the data if the value of raw data (could be smoothed raw data) is less than ekg threshold keep counting forwards
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

    def rm_peak(self, time):
        """ 
        Examine a second of interest and manually remove artifact R peaks.
        
        Parameters
        ----------
        time: str {'hh:mm:ss'}
            Time in the format specified dictating the second containing the peak of interest.
        
        Modifies
        -------
        self.rpeaks : Peaks that have been removed are removed from attribute.
        self.rpeaks_df : Peaks that have been removed are removed from attribute.
        self.rpeak_artifacts : Removed peaks added to attribute.
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

    def undo_rm_peak(self, time):
        """
        Manually add back incorrectly removed peaks from EKG.rm_peak method.
            
        Parameters
        ----------
        time : str {'hh:mm:ss'}
            Second of incorrectly removed R peak.

        Notes
        -----
        This is strictly an "undo" method. It is NOT equivalent to add_peaks().

        Modifies
        -------
        self.rpeaks : Incorrectly removed R peaks added back.
        self.rpeaks_df : Incorrectly removed R peaks added back.
        self.rr : IBI values recalculated to reflect change in R peaks.
        self.nn : IBI values recalculated to reflect change in R peaks.
        self.rpeaks_artifacts : Incorrectly removed R peaks removed from attribute.

        See Also
        --------
        EKG.rm_peak : Examine a second of interest and manually remove artifact R peaks.
        EKG.add_peak : Examine a second of interest and manually add missed R peaks.
        EKG.undo_add_peak : Manually remove incorrectly added peaks from EKG.add_peak method.
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
        """
        Examine a second of interest and manually add missed R peaks.

        Parameters
        ----------
        time : str {'hh:mm:ss'}
            Second within which peak is to be added.

        Modifies
        -------
        self.rpeaks : Added peaks added to attribute.
        self.rpeaks_df : Added peaks added to attribute.
        self.rr : IBI values recalculate to reflect changed R peaks.
        self.nn : IBI values recalculate to reflect changed R peaks.
        self.rpeaks_added : Added peaks stored.

        See Also
        --------
        EKG.undo_add_peak : Manually add back incorrectly added R peaks from EKG.add_peak method.
        EKG.rm_peak : Examine a second of interest and manually remove artifact R peak.
        EKG.undo_rm_peak : Manually add back incorrectly removed R peaks from EKG.rm_peak method.
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
        """
        Manually remove incorrectly added peaks from EKG.add_peak method.

        Parameters
        ----------
        time : str {'hh:mm:ss'}
            Second of incorrectly removed R peak.
   
        Modifies
        -------
        self.rpeaks : Incorrectly added R peaks removed.
        self.rpeaks_df : Incorrectly added R peaks removed.
        self.rr : IBI values recalculated to reflect change in R peaks.
        self.nn : IBI values recalculated to reflect change in R peaks.
        self.rpeaks_added : Incorrectly added R peaks removed from attribute.

        Notes
        -----
        This is strictly an "undo" method. It is NOT equivalent to EKG.rm_peak.

        See Also
        --------
        EKG.add_peak : Examine a second of interest and manually add missed R peaks.
        EKG.rm_peak : Examine a second of interest and manually remove artifact R peaks.
        EKG.undo_rm_peak : Manually add back incorrectly removed peaks from EKG.rm_peak method. 
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
        """
        Manually remove IBI's that can't be manually added with EKG.add_peak() method.
        
        IBIs to be removed could correspond to missing data (due to cleaning) or missed beats.

        Parameters
        ----------
        thres: int, default 3000
            Threshold time for automatic IBI removal (ms).

        Notes
        -----
        This step must be completed LAST, after removing any false peaks and adding any missed peaks.

        See Also
        --------
        EKG.add_peak : Manually add missed R peaks. 
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

    def calc_RR(self, smooth, mw_size, upshift, rms_align):
        """
        Set R peak detection threshold, detect R peaks and calculate R-R intervals.

        Parameters
        ----------
        smooth : bool, default True
            If set to True, raw EKG data will be smoothed using RMS smoothing window.
        mw_size : float, default 100
            Time over which the moving average of the EKG signal will be taken to calculate the R peak detection threshold (ms).
        upshift : float, default 3.5
            Percentage of EKG signal that the moving average will be shifted up by to set the R peak detection threshold.
        rms_align: str, default 'right'
            Whether to align the mean to the right or left side of the moving window [options: 'right', 'left']

        See Also
        --------
        EKG.set_Rthres : Set R peak detection threshold based on moving average shifted up by a percentage of the EKG signal.
        EKG.detect_Rpeaks :  Detect R peaks of raw or smoothed EKG signal based on detection threshold. 
        EKG.pan_tompkins_detector : Use the Pan Tompkins algorithm to detect R peaks and calculate R-R intervals.
        """
        
        # set R peak detection parameters
        self.set_Rthres(smooth, mw_size, upshift, rms_align)
        # detect R peaks & make RR tachogram
        self.detect_Rpeaks(smooth)

    def pan_tompkins_detector(self):
        """
        Use the Pan Tompkins algorithm to detect R peaks and calculate R-R intervals.

        Jiapu Pan and Willis J. Tompkins.
        A Real-Time QRS Detection Algorithm. 
        In: IEEE Transactions on Biomedical Engineering 
        BME-32.3 (1985), pp. 230–236.

        See Also
        ----------
        EKG.calc_RR : Set R peak detection threshold, detect R peaks and calculate R-R intervals.
        """

        self.metadata['analysis_info']['pan_tompkins'] = True
        #interpolate data because has NaNs, cant for ecg band pass filter step
        data = self.data.interpolate()
        #makes our data a list because that is the format that bsnb wants it in
        signal = pd.Series.tolist(data['Raw'])
        # get sample rate 
        # must be an int
        sr = int(self.metadata['analysis_info']['s_freq'])
        
        filtered_signal = bsnb.detect._ecg_band_pass_filter(signal, sr) #Step 1 of Pan-Tompkins Algorithm - ECG Filtering (Bandpass between 5 and 15 Hz)
        differentiated_signal = diff(filtered_signal)
        squared_signal = differentiated_signal * differentiated_signal
        nbr_sampls_int_wind = int(0.080 * sr)
        # Initialisation of the variable that will contain the integrated signal samples
        integrated_signal = zeros_like(squared_signal)
        cumulative_sum = squared_signal.cumsum()
        integrated_signal[nbr_sampls_int_wind:] = (cumulative_sum[nbr_sampls_int_wind:] - cumulative_sum[:-nbr_sampls_int_wind]) / nbr_sampls_int_wind
        integrated_signal[:nbr_sampls_int_wind] = cumulative_sum[:nbr_sampls_int_wind] / arange(1, nbr_sampls_int_wind + 1)

        #R peak detection algorithm
        rr_buffer, signal_peak_1, noise_peak_1, threshold = bsnb.detect._buffer_ini(integrated_signal, sr)
        probable_peaks, possible_peaks= bsnb.detect._detects_peaks(integrated_signal, sr)
        #Identification of definitive R peaks
        definitive_peaks = bsnb.detect._checkup(probable_peaks, integrated_signal, sr, rr_buffer, signal_peak_1, noise_peak_1, threshold)

        # Conversion to integer type.
        definitive_peaks = array(list(map(int, definitive_peaks)))
        #Correcting step
        #Due to the multiple pre-processing stages there is a small lag in the determined peak positions, which needs to be corrected !
        definitive_peaks_rephase = np.array(definitive_peaks) - 30 * (sr / 1000)
        definitive_peaks_rephase = list(map(int, definitive_peaks_rephase))
        #make peaks list
        index = data.index[definitive_peaks_rephase]
        values = np.array(signal)[definitive_peaks_rephase]
        self.rpeaks = pd.Series(values, index = index)
        print('R peak detection complete')

        # get time between peaks and convert to mseconds
        self.rr = np.diff(self.rpeaks.index)/np.timedelta64(1, 'ms')
        
        # create rpeaks dataframe and add ibi columm
        rpeaks_df = pd.DataFrame(self.rpeaks)
        ibi = np.insert(self.rr, 0, np.NaN)
        rpeaks_df['ibi_ms'] = ibi
        self.rpeaks_df = rpeaks_df

        print('R-R intervals calculated')

    def export_RR(self, savedir):
        """
        Export R peaks and RR interval data to .txt files.

        Includes list of R peaks artifacts, R peaks added, R peaks detected, IBI artifacts, RR intervals and NN intervals.

        Parameters
        ----------
        savedir : str
            Path to directory where .txt files will be saved.

        See Also
        --------
        EKG.calc_RR : Set R peak detection threshold, detect R peaks and calculate R-R intervals.
        EKG.rm_ibi :  Manually remove IBI's that can't be manually added with EKG.add_peak() method.
        EKG.add_peak : Manually add missed R peak. 
        EKG.rm_peak : Examine a second of interest and manually remove artifact R peaks.
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
        if self.metadata['analysis_info']['pan_tompkins'] == False:
            rr_header = 'R peak detection mw_size = {} & upshift = {}'.format(self.metadata['analysis_info']['mw_size'], self.metadata['analysis_info']['upshift'])
        else:
            rr_header = 'R peak detection using the Pan Tompkins algorithm'
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
            if self.metadata['analysis_info']['pan_tompkins'] == False:
                nn_header = 'R peak detection mw_size = {} & upshift = {}.\nTotal artifacts removed = {} ( {} false peaks + {} false ibis).'.format(self.metadata['analysis_info']['mw_size'], self.metadata['analysis_info']['upshift'], arts_len, len(self.rpeak_artifacts), len(self.ibi_artifacts))
            else:
                nn_header = 'R peak detection using the Pan Tompkins algorithm.\nTotal artifacts removed = {} ( {} false peaks + {} false ibis).'.format(arts_len, len(self.rpeak_artifacts), len(self.ibi_artifacts))
            savenn = self.metadata['file_info']['fname'].split('.')[0] + '_nn.txt'
            nn_file = os.path.join(savedir, savenn)
            np.savetxt(nn_file, self.nn, header=nn_header, fmt='%.0f', delimiter='\n')
            print('nn intervals exported.')

        print('Done.')

    def calc_tstats(self, itype):
        """
        Calculate commonly used time domain HRV statistics.

        Time domain HRV statistics include mean, min and max HR (bpm), mean interbeat interval length, SDNN, RMSSD, pNN20 and pNN50.
        SDNN is the standard deviation of normal to normal IBI. RMSSD is the root mean squared standard deviation of normal interbeat interval length.
        pNN20 and pNN50 are the percentage of normal interbeat intervals that exceed 20ms and 50ms respectively.
        Min and max HR is determined over 5 RR intervals.

        Parameters
        ----------
        itype : str {'rr, 'nn'}
            Interval type.'rr' is uncleaned data. 'nn' is normal intervals (cleaned).

        See Also
        --------
        EKG.hrv_stats : Calculate all HRV statistics on IBI object.
        EKG.calc_fstats : Calculate frequency domain statistics.
        EKG.calc_psd_welch : Calculate welch power spectrum.
        EKG.calc_psd_mt : Calculate multitaper power spectrum. 
        EKG.calc_fbands : Calculate different frequency band measures.
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

    
    def interpolate_IBI(self, itype):
        """
        Resample tachogram to original sampling frequency and interpolate for power spectral estimation.

        This is done since RRs are not evenly placed.

        Parameters
        ----------
        itype : str {'rr', 'nn'}
            Interval type.'rr' is uncleaned data. 'nn' is normal intervals (cleaned).
        
        Note
        ----
        Adapted from pyHRV

        See Also
        --------
        EKG.calc_psd_welch : Calculate welch power spectrum.
        EKG.calc_psd_mt : Calculate multitaper power spectrum.
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

    def data_pre_processing(self,fs):
        """
        Load data to the workspace and perform pre-processing: Linearly interpolate the NN time series and zero center.

        Parameters
        ----------
        fs : int
            desired sampling frequency of time series in Hz

        Returns
        -------
        NN_intervals_interpolated : np.ndarray
            Interpolated and zero centered NN time series
        K : int 
            Length of NN_intervals_interpolated
        """
        # remove the time intervals that are not valid
        NN_intervals = self.nn
        NN_intervals = NN_intervals[~np.isnan(NN_intervals)]

        spike_times = np.cumsum(NN_intervals)/1000

        dt = 1/fs 
        T = np.floor(spike_times.max(0)) # Observation duration in seconds
        t = np.arange(1.0, T, dt) # Consider a linearly spaced time axis

        # Derive the linearly interpolated NN time series
        NN_intervals_interpolated = np.interp(t,spike_times,NN_intervals)
        K = NN_intervals_interpolated.shape[0]

        # Zero centering the time series prior to multi-tapering
        NN_intervals_interpolated = NN_intervals_interpolated - np.mean(NN_intervals_interpolated)
        NN_intervals_interpolated = NN_intervals_interpolated.reshape((K,1))
        return NN_intervals_interpolated, K
    
    def denoised_mt_spectral_estimation(self, NN_intervals_interpolated, N, NW, no_of_tapers, K, fs):
        """
        Peform expectation maximization to estimate the denoised Eigen coefficients and denoised Multitaper spectral estimates.

        Parameters
        ----------
        NN_intervals_interpolated : np.ndarray
            The interpolated inter heart beat interval time series
        N : int
            Total number of frequency bins
        NW : int 
            Time half bandwidth of multitapering
        no_of_tapers : int
            Number of tapers considered for Multitapering
        K : int 
            Observation duration in samples
        fs : int
            Desired sampling frequency of the time series in Hz

        Returns
        -------
        denoised_MT_est : np.ndarray
            denoised multitaper estimate of the Power Spectral Density.
        denoised_w_est_tapers : np.ndarray
            The real and imaginary components of the denoised Eigen-coefficients for each taper.

        Modifies
        --------
        self.psd_mt_denoised : Dict created containing power spectral density at respective frequencies.
                     'freqs' : np.ndarray
                       'pwr' : np.ndarray. Power spectral density in (V^2/Hz). 10log10 to convert to dB.

        See Also
        --------
        EKG.data_pre_processing : Load data to the workspace and perform pre-processing: Linearly interpolate the NN time series and zero center.
        EKG.direct_MT_Spectral_Estimation : Produce the classical multitaper estimate of the Power Spectral Density.
        EKG.plot_estimates : Plot the final PSD estimates with the confidence levels
        EKG.generate_PS : Generate power spectrum with desired confidence levels.
        """
        # Initializing the parameters
        iter_EM = 50 # Maximum EM iterations
        Q_initial_factor = 10**(5) # Initialization of Q
        sigma_observation = 1*10**(4) # Initialization of Observation noise variance
        Sequences = sp.signal.windows.dpss(K, NW, Kmax=no_of_tapers) # Generate the data tapers used for multitapering
        self.metadata['analysis_info']['psd_method'] = 'denoised multitaper'
        # constructing the inverse FFT matrix (A)
        A=np.zeros((K,2*N))
        for k in range(0,K):
            for n in range(0,N):
                A[k,2*n] = np.cos((k+1)*np.pi*(n)/N)
                A[k,2*n+1] = -np.sin((k+1)*np.pi*(n)/N)
        A= A/N
        A = np.delete(A, 1, 1)
        
        denoised_MT_est_tapers = np.zeros((N, no_of_tapers)) # Stores the denoised Eigen spectra for each taper
        denoised_w_est_tapers = np.zeros(((2*N - 1),no_of_tapers)) # Stores the denoised Eigen coefficients for each taper

        # Derive the denoised Eigen coefficients for each taper using Expectation Maximization
        for taper in range(0,no_of_tapers):
            print("Estimating denoised Eigen-coefficients for taper", taper+1)
            Q = Q_initial_factor*np.eye((2*N - 1)) # Initialize the Q matrix
            taper_sequence = Sequences[taper,:]
            taper_sequence = taper_sequence.reshape((K,1))    
            tapered_NN_intervals = taper_sequence*NN_intervals_interpolated # Obtain the tapered time series
            tapered_NN_intervals = tapered_NN_intervals.reshape((K,1))
            w_est = np.zeros((2*N-1, 1))
            P_est = np.zeros((2*N-1, 2*N-1))
            regularizer = (10**(-10))*np.eye((2*N-1))

            # Expectation Maximization
            for r in range(0,iter_EM):
                # Expectation step (E - step)
                w_est = np.linalg.inv(regularizer+A.T@A + np.linalg.inv(regularizer+Q)*sigma_observation)@(A.T@tapered_NN_intervals) # Update the expected value of the denoised Eigen coefficients
                P_est =  ((np.linalg.inv(regularizer+A.T@A/(regularizer[0,0]+sigma_observation) + np.linalg.inv(regularizer+Q)))) # Update the covariance of the denoised Eigen coefficients
                # Maximization (M - step)
                Q = np.diag(np.diag(P_est + w_est@w_est.T)) # Update the Q matrix
                sigma_observation = (tapered_NN_intervals.T@tapered_NN_intervals - 2*tapered_NN_intervals.T@A@w_est + np.trace((A.T)@(A)@(P_est + w_est@w_est.T)))/K # Update the observation noise variance
               # print(sigma_observation)                               

            # Store estimated denoised Eigen coefficients for all tapers
            denoised_w_est_tapers[:,taper] = w_est[:,0]

            # Derive and store the denoised Eigen spectra
            final_eigen_spectra = w_est@w_est.T
            denoised_MT_est_tapers[0,taper] = final_eigen_spectra[0,0]
            for n in range(1,N):
                denoised_MT_est_tapers[n,taper] =  final_eigen_spectra[2*n-1,2*n-1] + final_eigen_spectra[2*n,2*n]

        # The final multi-taper estimates are the average spectral estimates across all tapers
        denoised_MT_est = np.squeeze(np.mean(np.absolute(denoised_MT_est_tapers), axis=1, keepdims = True))
        freq_vector = np.arange(0.0, 0.5*fs, 0.5*fs/N)
        self.psd_mt_denoised = {'freqs': freq_vector, 'pwr': denoised_MT_est}

        return denoised_MT_est, denoised_w_est_tapers   

    def direct_MT_Spectral_Estimation(self, NN_intervals_interpolated, N, NW, no_of_tapers, fs):
        """
        Produce the classical multitaper estimate of the Power Spectral Density.

        Parameters
        ----------
        NN_intervals_interpolated : np.ndarray
            The interpolated inter heart beat interval time series
        N : int
            Total number of frequency bins
        NW : int 
            Time half bandwidth of multitapering
        no_of_tapers : int
            Number of tapers considered for Multitapering
        fs : int
            Desired sampling frequency of the time series in Hz

        Returns
        -------
        direct_MT_est : np.ndarray
            The classical multitaper estimate of the Power Spectral Density
        direct_w_est_tapers : np.ndarray
            The real and imaginary components of the Eigen-coefficients for each taper.

        Modifies
        --------
        self.psd_mt : Dict created containing power spectral density at respective frequencies.
            'freqs' : np.ndarray
            'pwr' : np.ndarray. Power spectral density in (V^2/Hz). 10log10 to convert to dB.


        See Also
        --------
        EKG.data_pre_processing : Load data to the workspace and perform pre-processing: Linearly interpolate the NN time series and zero center.
        EKG.denoised_MT_Spectral_Estimation : Peform expectation maximization to estimate the denoised Eigen coefficients and denoised Multitaper spectral estimates.
        EKG.plot_estimates : Plot the final PSD estimates with the confidence levels
        EKG.generate_PS : Generate power spectrum with desired confidence levels.
        EKG.Confidence_Intervals_Bootstrapping : Perform bootstrapping to derive confidence bounds. 
        EKG.Confidence_Intervals_Chi_squared : Derive confidence bounds based on the Chi-squared approximation.
        """
        # Initializing variables
        K = NN_intervals_interpolated.shape[0] # Observation duration in samples
        Fourier_est = np.zeros((2*N, no_of_tapers),dtype=complex) # Eigen Spectra
        direct_MT_est_tapers = np.zeros((2*N,no_of_tapers)) # Classical Multi-taper spectral estimate
        direct_w_est_tapers = np.zeros((2*N-1,no_of_tapers)) # The real and imaginery components of the Eigen-coefficients for each taper
        dpss_seq = sp.signal.windows.dpss(K, NW, Kmax=no_of_tapers) # Generate the data tapers used for multitapering
        self.metadata['analysis_info']['psd_method'] = 'direct multitaper'
        # Computing the Eigen-coefficients and the Eigenspectra for each taper
        for taper in range(0,no_of_tapers):
            temp = NN_intervals_interpolated.T*dpss_seq[taper,:]
            Fourier_est[:, taper] = ((np.fft.fft(temp, n = 2*N)).T).reshape((2*N,)) # Eigen-coefficients of the tapered process
            direct_w_est_tapers[0,taper] = np.absolute(Fourier_est[0,taper]) # dc component
            direct_w_est_tapers[1:2*N-1:2,taper] = np.real(Fourier_est[1:N,taper]) # real components
            direct_w_est_tapers[2:2*N-1:2,taper] = np.imag(Fourier_est[1:N,taper]) # imaginary components
            direct_MT_est_tapers[:,taper] = (np.absolute(Fourier_est[:, taper]))**2 # Eigenspectral estimates

        # The final multi-taper estimates are the average spectral estimates across all tapers
        direct_MT_est = np.mean(direct_MT_est_tapers[0:N,:], axis=1, keepdims = True)
    
        freq_vector = np.arange(0.0, 0.5*fs, 0.5*fs/N)
        self.psd_mt_direct = {'freqs': freq_vector, 'pwr': direct_MT_est}

        return direct_MT_est, direct_w_est_tapers

    def confidence_intervals_bootstrapping(self, MT_est, w_est_tapers, CI, bootstrapping_repeats, fs, K, N):
        """
        Perform bootstrapping to derive confidence bounds. 

        Kim et al.,2018
        A Multitaper Frequency-Domain Bootstrap Method
        In: IEEE Signal Processing Letters 
        SPL-25.12 (2018), pp. 1805–1809.
        
        Parameters
        ----------
        MT_est : np.ndarray
            Multitaper estimate of the Power Spectral Density
        w_est_tapers : np.ndarray
            Real and imaginery components of the Eigen coefficients of each taper
        CI : int
            Desired confidence bounds
        bootstrapping_repeats : int
            Number of bootstrap repeats
        fs : int
            Sampling frequency
        K : int
            Observation duration in samples
        
        Returns
        -------
        Lower_confidence_PSD : np.ndarray
            The lower confidence bound of the multi-taper spectral estimates
        Upper_confidence_PSD : np.ndarray
            The upper confidence bound of the multi-taper spectral estimates

        See Also
        --------
        EKG.denoised_MT_Spectral_Estimation : Peform expectation maximization to estimate the denoised Eigen coefficients and denoised Multitaper spectral estimates.
        EKG.direct_MT_Spectral_Estimation : Produce the classical multitaper estimate of the Power Spectral Density.
        EKG.plot_estimates : Plot the final PSD estimates with the confidence levels
        EKG.generate_PS : Generate power spectrum with desired confidence levels.
        EKG.confidence_intervals_chi_squared : Derive confidence bounds based on the Chi-squared approximation
        """
        self.metadata['analysis_info']['confidence_intervals'] = 'bootstrapping'
        N = MT_est.shape[0] # Number of frequency bins
        scaling_fac = (1/fs)*(K / N) # The scaling factor of the final estimates

        no_of_tapers = w_est_tapers.shape[1] # The number of tapers used for multi-tapering

        denoised_MT_est_bootstrap = np.zeros((N,bootstrapping_repeats)) # Store bootstrapped MT estimates

        #  Step 2 (Kim et al., 2018): Scale the Eigen coefficients by the power of the MT estimates
        scaled_w_hat = w_est_tapers
        scaled_w_hat[0,:] = scaled_w_hat[0,:]/np.sqrt(MT_est[0]/2)
        for n in range(0,N-1):
            S_fac = np.sqrt(MT_est[n+1]/2)
            scaled_w_hat[2*n+1,:] = scaled_w_hat[2*n+1,:] / S_fac
            scaled_w_hat[2*n+2,:] = scaled_w_hat[2*n+2,:] / S_fac    

        # Step 3 (Kim et al., 2018): Standardize the concatanated Eigencoefficients to have zero mean and unit variance
        for taper in range(0,no_of_tapers):
            temp = scaled_w_hat[:,taper]
            scaled_w_hat[:,taper] = [temp - np.mean(temp)]/np.sqrt(np.mean((temp - np.mean(temp))**2))

        # Perform the bootstrapping
        for i in range(0,bootstrapping_repeats):    
            denoised_MT_est_bootstrap_taper = np.zeros((N,no_of_tapers)) # Bootstrapped MT estimate for each taper
            scaled_w_hat_bootstrap = np.zeros((2*N-1,no_of_tapers)) #. Bootstrapped Eigen coefficients        
            for n in range(0,scaled_w_hat.shape[0]):
                temp = scaled_w_hat[n,:]
                bootstrap_order = np.random.randint(no_of_tapers, size=no_of_tapers) # Step 4 (Kim et al., 2018): bootstrapping with replacement
                scaled_w_hat_bootstrap[n,:] = temp[bootstrap_order]
                # Step 5 (Kim et al., 2018): Re-scale the bootstrapped eigen coefficients
                if n == 0:
                    scaled_w_hat_bootstrap[n,:] = scaled_w_hat_bootstrap[n,:]*np.sqrt(MT_est[0]/2)
                else:
                    scaled_w_hat_bootstrap[n,:] = scaled_w_hat_bootstrap[n,:]*np.sqrt(MT_est[(np.ceil(n/2)).astype(int)]/2)                
            # Step 6 (Kim et al., 2018): Derive the bootstrapped Eigen spectra for each taper
            for taper in range(0,no_of_tapers):
                temp = scaled_w_hat_bootstrap[:,taper]
                temp = temp.reshape((temp.shape[0],1))
                final_eigen_spectra_bootstrap = temp@temp.T
                denoised_MT_est_bootstrap_taper[0,taper] = final_eigen_spectra_bootstrap[0,0]
                for n in range(0,N-1):
                    denoised_MT_est_bootstrap_taper[n+1,taper] =  final_eigen_spectra_bootstrap[2*n+1,2*n+1]+final_eigen_spectra_bootstrap[2*n+2,2*n+2]    
            # Derive the bootstrapped Multitaper Spectral Estimates
            temp = scaling_fac*np.mean(np.absolute(denoised_MT_est_bootstrap_taper), axis=1, keepdims = True)
            denoised_MT_est_bootstrap[:,i] = temp.reshape((temp.shape[0],))

        # Specify the lower and upper percentiles based on the desired Confidence Intervals
        lower_percentile = (np.floor(((1-CI)/2)*bootstrapping_repeats)).astype(int)
        upper_percentile = (np.ceil(((1+CI)/2)*bootstrapping_repeats)).astype(int)
        Upper_confidence_PSD = np.zeros((N,1))
        Lower_confidence_PSD = np.zeros((N,1))

        # Derive the confidence bounds using the upper and lower percentiles
        for n in range(0,N):
            temp = np.sort(denoised_MT_est_bootstrap[n,:])
            Lower_confidence_PSD[n] = temp[lower_percentile]
            Upper_confidence_PSD[n] = temp[upper_percentile]

        return Lower_confidence_PSD.reshape((N,)), Upper_confidence_PSD.reshape((N,))

    def Confidence_intervals_chi_squared(self, MT_est, CI, no_of_tapers, N):
        """
        Derive confidence bounds based on the Chi-squared approximation.

        Percival & Walden, 1993
        In: Spectral analysis for Physical Applications
        Pp. 255 & 343
        
        Parameters
        ----------
        MT_est : np.ndarray
            Multitaper estimate of the Power Spectral Density
        CI : int
            The desired confidence bounds
        no_of_tapers : int
            Number of tapers used for multitapering

        Returns
        -------
        Lower_confidence_PSD : np.ndarray
            The lower confidence bound of the multi-taper spectral estimates
        Upper_confidence_PSD : np.ndarray
            The upper confidence bound of the multi-taper spectral estimates

        See Also
        --------
        EKG.denoised_mt_spectral_estimation : Peform expectation maximization to estimate the denoised Eigen coefficients and denoised Multitaper spectral estimates.
        EKG.direct_mt_spectral_estimation : Produce the classical multitaper estimate of the Power Spectral Density.
        EKG.plot_estimates : Create figure containing the final PSD estimates with the confidence levels.
        EKG.generate_PS : Generate power spectrum with desired confidence levels.
        EKG.confidence_intervals_bootstrapping : Perform bootstrapping to derive confidence bounds. 
        """
        self.metadata['analysis_info']['confidence_intervals'] = 'chi_sq'
        # The Degrees of freedom of the Chi-squared distribution equals to twice the number of tapers used in multi-tapering
        Degree_of_freedom = 2*no_of_tapers
        
        Lower_confidence_PSD = (Degree_of_freedom / chi2.ppf((1+CI)/2, df=Degree_of_freedom)) * abs(MT_est);
        Upper_confidence_PSD = (Degree_of_freedom / chi2.ppf((1-CI)/2, df=Degree_of_freedom)) * abs(MT_est);

        return Lower_confidence_PSD.reshape((N,)), Upper_confidence_PSD.reshape((N,)) 

    def plot_estimates(self, MT_PSD_est, Lower_confidence_PSD, Upper_confidence_PSD, fs):
        """
        Create figure containing the final PSD estimates with the confidence levels.

        Parameters
        ----------
        MT_est : np.ndarray
            Multitaper estimate of the Power Spectral Density
        Lower_confidence_PSD : np.array
            The lower confidence bound of the multi-taper spectral estimates
        Upper_confidence_PSD : np.array
            The upper confidence bound of the multi-taper spectral estimates
        fs : int
            Sampling frequency

        Returns
        -------
        fig : figure
            Plot of the final PSD estimates with confidence levels

        See Also
        --------
        EKG.generate_PS : Generate power spectrum with desired confidence levels
        """
        N = MT_PSD_est.shape[0]
        freq_vector = np.arange(0.0, 0.5*fs, 0.5*fs/N)
        y_axis_upper_bound = 20*10**4

        fig, ax = plt.subplots(figsize=(15,3))
        
        freq_1 = np.max(freq_vector[freq_vector<= 0.04])
        ax.plot([freq_1,freq_1],[0,y_axis_upper_bound], 'b--')
        freq_2 = np.max(freq_vector[freq_vector<= 0.15])
        ax.plot([freq_2,freq_2],[0,y_axis_upper_bound], 'b--')
        freq_3 = np.max(freq_vector[freq_vector<= 0.40])
        ax.plot([freq_3,freq_3],[0,y_axis_upper_bound], 'b--')

        ax.plot(freq_vector, MT_PSD_est, color="black")
        ax.fill_between(freq_vector, Lower_confidence_PSD, Upper_confidence_PSD, color='k', alpha=.4)
        plt.xlabel("frequency ($Hz$)")
        plt.ylabel("Power ($ms^2/Hz$)")
        ax.set_xlim(0, 0.4)
        plt.subplots_adjust(bottom=0.15)
       # plt.xlim([np.min(freq_vector), np.max(freq_vector)])
        #plt.ylim([0, y_axis_upper_bound])      
        return fig               

    def calc_psd_welch(self, itype, window):
        """ 
        Calculate welch power spectrum.

        Parameters
        ----------
        itype : str {'rr', 'nn'}
            Interval type with which to calculate the power spectrum.
            'rr' is uncleaned data. 'nn' is normal intervals (cleaned).
        window : str
            Windowing function.
            Options from scipy.signal welch
            Wrapper default 'hamming'

        See Also
        --------
        EKG.calc_psd_mt : Calculate multitaper power spectrum.
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
 
    def calc_fbands(self, method):
        """
        Calculate frequency band measures.

        Parameters
        ----------
        method : str {'welch', 'mt'}
            Method to be used to calculate frequency band measures.
 
        Notes
        -----
        Modified from pyHRV
        Normalized units are normalized to total lf + hf power, according to Heathers et al. (2014)
        """
        if method is None:
            method = input('Please enter PSD method (options: "welch", "mt_direct", "mt_denoised"): ')
        if method == 'welch':
            psd = self.psd_welch
        if method == 'mt_direct':
            psd = self.psd_mt_direct
        elif method == 'mt_denoised':
            psd = self.psd_mt_denoised
        
        # set frequency bands
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
        self.interpolate_IBI(itype)
       
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

    def hrv_stats(self, itype='nn', nn_file=None, method='mt_denoised', bandwidth=0.01, window='hamming'):
        """
        Calculate both time and frequency domain HRV statistics on IBI object.

        Parameters
        ----------
        itype : str {'nn', 'rr'}
            Interbeat interval object type to use for calculations. 
            'rr' is uncleaned data. 'nn' is normal intervals (cleaned)
        nn_file : str, optional
            Path to csv file containing cleaned nn values, if nn values were previously exported.
        method : str, {'mt_denoised', 'mt_direct', 'welch'}
            Method to use when calculating power spectrum. 
            'mt' is multitaper
        bandwidth : float, default 0.01
            Bandwidth used when calculating frequency domain statistics.
        window : str , default 'hamming'
            Window type used for welch power spectral analysis.
            Options from scipy.signal welch.
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

        # create nn variable if it doesn't exist
        try:
            self.nn
        except AttributeError:
            self.nn = self.rr

        # calculate statistics
        self.calc_tstats(itype)
        self.calc_fstats(itype, method, bandwidth, window)
        
        print('Done.')

    def to_spreadsheet(self, spreadsheet, savedir):
        """
        Append calculations as a row in master spreadsheet.

        Information exported includes arrays 'data', 'rpeaks', 'rr', 'rr_diff', 'rr_diffsq', 'rpeak_artifacts', 'rpeaks_added', 'ibi_artifacts',
        'rpeaks_df', 'nn', 'nn_diff', 'nn_diffsq', 'rr_arts', 'ii_interp', 'psd_mt_direct', 'psd_mt_denoised', psd_welch', 'psd_fband_vals' if calculated. 

        Parameters
        ----------
        savedir : str
            Path to directory where spreadsheet will be saved. 
        spreadsheet : str
            Name of output file. 

        Notes
        -----
        Creates new spreadsheet if output file does not exist. 
        """
        # this is from before division to two classes. 'data' and 'rpeaks' arrays shouldn't exist in IBI object.
        arrays = ['data', 'rpeaks', 'rr', 'rr_diff', 'rr_diffsq', 'rpeak_artifacts', 'rpeaks_added', 'ibi_artifacts',
        'rpeaks_df', 'nn', 'nn_diff', 'nn_diffsq', 'rr_arts', 'ii_interp', 'psd_mt_direct', 'psd_mt_denoised', 'psd_welch', 'psd_fband_vals']
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
        """ 
        Export HRV statistics as a csv report.

        Parameters
        ----------
        savedir : str, optional
            Path to directory in which to save report.
        fmt: str, {'txt', 'json'}
            Output format.

        See Also
        --------
        EKG.hrv_stats : Calculate both time and frequency domain HRV statistics on IBI object.
        EKG.calc_fstats : Calculate commonly used frequency domain HRV statistics.
        EKG.calc_tstats : Calculate commonly used time domain HRV statistics.
        EKG.denoised_mt_spectral_estimation : Peform expectation maximization to estimate the denoised Eigen coefficients and denoised Multitaper spectral estimates.
        EKG.direct_mt_spectral_estimation : Produce the classical multitaper estimate of the Power Spectral Density.
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
        arrays = ['data', 'rpeaks', 'rr', 'rr_diff', 'rr_diffsq', 'rpeak_artifacts', 'rpeaks_added', 'ibi_artifacts', 'rpeaks_df', 'nn', 'nn_diff', 'nn_diffsq', 'rr_arts', 'ii_interp', 'psd_mt_direct', 'psd_mt_denoised', 'psd_fband_vals']
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
            self.psd_mt_denoised
        except AttributeError: 
            pass
        else:
            savepsd = saveinfo + '_psd_mt_denoised.txt'
            psdfile = os.path.join(savedir, savepsd)
            psd_mt_df = pd.DataFrame(self.psd_mt_denoised)
            psd_mt_df.to_csv(psdfile, index=False)
        try:
            self.psd_mt_direct
        except AttributeError: 
            pass
        else:
            savepsd = saveinfo + '_psd_mt_direct.txt'
            psdfile = os.path.join(savedir, savepsd)
            psd_mt_df = pd.DataFrame(self.psd_mt_direct)
            psd_mt_df.to_csv(psdfile, index=False)

        try:
            self.psd_welch
        except AttributeError: 
            pass
        else:
            savepsd = saveinfo + '_psd_welch.txt'
            psdfile = os.path.join(savedir, savepsd)
            psd_mt_df = pd.DataFrame(self.psd_welch)
            psd_mt_df.to_csv(psdfile, index=False)

    # plotting methods
    def plotpeaks(self, rpeaks=True, ibi=True, thres = True):
        """
        Plot EKG class instance.

        Visualization of raw EKG data, smoothed EKG data, R peaks, IBI length and EKG threshold detection line.
        
        Parameters
        ----------
        rpeaks : bool, default True
            Shows r peaks on plot if set to True.
        ibi : bool, default True
            Displays plot with IBI time leading up to each r peak if set to True
        thres : bool, default True
            Shows threshold line if set to True.
        """
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
                    if self.metadata['analysis_info']['pan_tompkins'] == True:
                        ax.legend(('raw data', 'rpeak'), fontsize = 'small')
                    else:
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

    def generate_welch(self, method='welch', dB=False, bands=True, save=True, savedir=None):
        """
        Plot power spectrum with method of choice and save if appropriate. 

        Parameters
        ----------
        method : str, {'welch', 'mt'}
            Method by which power spectrum is to be calculated.
            'mt' is multitaper.
        dB : bool, default False
            If True, decibals used as unit for power spectral density instead of s^2/Hz
        bands : bool, default True
            If True, spectrum plotted colored by frequency band.
        save : bool, default True
            If True, power spectrum will be saved as well as plotted.
        savedir : str, optional
            Path to directory where spectrum is to be saved. 

        See Also
        --------
        EKG.calc_psd_welch : Calculate welch power spectrum. 
        """
        
        # set title
        title = self.metadata['file_info']['in_num'] + ' ' + self.metadata['file_info']['start_date'] + '\n' + self.metadata['file_info']['sleep_stage'] + ' ' + self.metadata['file_info']['cycle']
        try:
            n.metadata['file_info']['epoch']
        except:
            pass
        else:
            title = title + ' ' +  n.metadata['file_info']['epoch']

        # set data to plot
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
            ax.plot(psd['freqs'], pwr, color='black', zorder=10)
            
            colors = [None, 'yellow', 'darkorange', 'tomato']
            zdict = {0:0.6, 1:0.6, 2:0.4, 3:0.6}
            for (zord, alpha), (key, value), color in zip(zdict.items(), self.psd_fband_vals.items(), colors):
                if value['idx'] is not None:
                    # get intercepts & plot vertical lines for bands
                    xrange = [float(x) for x in self.freq_stats[key]['freq_range'][1:-1].split(",")] 
                    
                    # fill spectra by band
                    ax.fill_between(psd['freqs'], pwr, where = [xrange[0] <= x for x in psd['freqs']], 
                                    facecolor=color, alpha=alpha, zorder=zord)    
            
        ax.set_xlim(0, 0.4)
        ax.margins(y=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(ylabel)
        plt.suptitle(title)

        if save == True:
            if savedir is None:
                print('ERROR: File not saved. Please specify savedir argument.')
            else:
                savename = os.path.join(savedir, self.metadata['file_info']['fname'].split('.')[0]) + '_psd.png'
                fig.savefig(savename, dpi=300)

        return fig

    def generate_PS(self, denoised = True, confidence = 'bootstrapping'):
        """
        Generate power spectrum with desired confidence levels.

        Parameters
        ----------
        denoised : bool default True
            Denoise the mulitaper estimate of the power spectral density. 
        confidence : str {'bootstrapping', 'chi sq'}
            Method with which to determine the confidence intervals.
        
        See Also
        ---------
        EKG.data_pre_processing : Load data to the workspace and perform pre-processing.
        EKG.denoised_mt_spectral_estimation : Peform expectation maximization to estimate the denoised Eigen coefficients and denoised Multitaper spectral estimates.
        EKG.direct_mt_spectral_estimation :  Produce the classical multitaper estimate of the Power Spectral Density.
        EKG.plot_estimates : Create figure containing the final PSD estimates with the confidence levels.

        """

        # Main task 1. Load data into the workspace and specify the parameters - pre processing

        # Specify the desired sampling frequency of the time series in Hz
        fs = 4

        # Extract the interpolated and zero centered NN time series
        NN_intervals_interpolated, K = self.data_pre_processing(fs)
        K = NN_intervals_interpolated.shape[0]

        # Set the parameters required for Spectral analysis - multi tapering
        N = 512        # Number of frequency bins considered in the frequency interval [0,fs/2). This determines the frequency spacing.  
        NW = 2             # time half-bandwidth product of Multitapering
        no_of_tapers = 3   # the number of tapers considered for Multitapering

        # Set the parameters required for Confidence Intervals
        CI = 0.95  # Specify the Required Confidence levels
        bootstrapping_repeats = 5000 # Specify the number of bootstrap samples
        scaling_fac = (1/fs)*(K / N) # Final scaling factor of the PSD estimates
        freq_vector = np.arange(0.0, 0.5*fs, 0.5*fs/N)
        multi_tapering_spectral_resolution = NW*fs/K

        if denoised==True:
            denoised_MT_est, denoised_w_est_tapers = self.denoised_mt_spectral_estimation(NN_intervals_interpolated, N, NW, no_of_tapers,K, fs)

            # Multiply by the required scaling factors to get the final spectral estimates
            denoised_MT_est_final = scaling_fac*denoised_MT_est;                        

            if confidence == "bootstrapping":
                denoised_MT_est_Lower_confidence_bootstrap, denoised_MT_est_Upper_confidence_bootstrap = self.confidence_intervals_bootstrapping(denoised_MT_est, denoised_w_est_tapers, CI, bootstrapping_repeats, fs, K, N)
                fig = self.plot_estimates(denoised_MT_est_final, denoised_MT_est_Lower_confidence_bootstrap, denoised_MT_est_Upper_confidence_bootstrap, fs)
                plt.title('Denoised Multitaper Spectral Estimate: with %d%% Confidence Intervals - Bootstrapping'% (CI*100),fontdict = {'fontsize' : 16}) 
            if confidence == "chi sq": 
                denoised_MT_est_Lower_confidence_Chi_squared, denoised_MT_est_Upper_confidence_Chi_squared = self.confidence_intervals_chi_squared(denoised_MT_est_final, CI, no_of_tapers, N)
                fig = self.plot_estimates(denoised_MT_est_final, denoised_MT_est_Lower_confidence_Chi_squared, denoised_MT_est_Upper_confidence_Chi_squared, fs)
                plt.title('Denoised Multitaper Spectral Estimate: with %d%% Confidence Intervals - Chi - squared test'% (CI*100),fontdict = {'fontsize' : 16})
        if denoised==False:
            direct_MT_est, direct_w_est_tapers = self.direct_mt_spectral_estimation(NN_intervals_interpolated, N, NW, no_of_tapers, fs)
    
            # Multiply by the required scaling factors to get the final spectral estimates
            direct_MT_est_final = scaling_fac*direct_MT_est
            if confidence == 'bootstrapping':
                direct_MT_est_Lower_confidence_bootstrap, direct_MT_est_Upper_confidence_bootstrap = self.confidence_intervals_bootstrapping(direct_MT_est, direct_w_est_tapers, CI, bootstrapping_repeats, fs, K, N)
                fig = self.plot_estimates(direct_MT_est_final, direct_MT_est_Lower_confidence_bootstrap, direct_MT_est_Upper_confidence_bootstrap, fs)
                plt.title('Direct Multitaper Spectral Estimate: with %d%% Confidence Intervals - Bootstrapping'% (CI*100),fontdict = {'fontsize' : 16})
            if confidence == 'chi sq':
                direct_MT_est_Lower_confidence_Chi_squared, direct_MT_est_Upper_confidence_Chi_squared = self.confidence_intervals_chi_squared(direct_MT_est_final, CI, no_of_tapers, N)
                fig = self.plot_estimates(direct_MT_est_final, direct_MT_est_Lower_confidence_Chi_squared, direct_MT_est_Upper_confidence_Chi_squared, fs)
                plt.title('Direct Multitaper Spectral Estimate: with %d%% Confidence Intervals - Chi - squared test'% (CI*100),fontdict = {'fontsize' : 16})   
        plt.xlabel("frequency ($Hz$)")
        plt.show()
        # plt.xlim([np.min(freq_vector), np.max(freq_vector)])
