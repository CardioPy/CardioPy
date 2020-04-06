""" This file contains the Auto_Param and EKG classes.

    The best parameters for using the EKG class to detect R peaks can be determined and applied to creathe the EKG object with the Auto_Param class.
    All R peak detections should be manually inspected with EKG.plotpeaks method.
    False detections should be manually removed with rm_peaks method, while missed detections should be manually added with add_peaks methid.
    After rpeak examination, NaN data can be accounted for by removing false IBIs with rm_ibi method.

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

class Auto_Param:

    """ General class to determine optimal parameters for the detection of R peaks using the EKG class.
    
    Attributes
    ----------
    fname : str
        filename
    fpath : str
        path to file
    mw_size_opt : list of floats (default: [20, 130])
        fill in when better idea of what you are doing
    upshift_opt : list of floats (default: [1.0, 5.0])
        fill in when better idea of what you are doing
    sm_wn_opt : list of floats ([10, 50])
        fill in when better idea of what you are doing
    hb_range : list of floats ([40, 140])
        range of heart beats per minute that would be appropriate
    min_ibi : float (default: 500)
        minimum IBI length in ms
    max_ibi : float (default: 1400)
        maximum IBI length in ms
    smooth : BOOL (default: False)
        Whether raw signal should be smoothed before peak detections. Set True if raw data has consistent high frequency noise
        preventing accurate peak detection
    min_dur : bool (default:True)
        only load files that are >= 5 minutes long
    epoched: bool (default: True)
        whether file was epoched using ioeeg
    rm_artifacts : bool (default: False)
        apply IBI artifact removal algorithm
    detect_peaks : bool (default: True)
        whether peaks should be detected
    sampling_freq : int (default: 250)
        number of samples taken per second
    """

    def __init__(self, fname, fpath, mw_size_opt = [20, 130], upshift_opt = [1.0, 5.0], sm_wn_opt = [10, 50], hb_range = [40, 140], min_ibi = 500, max_ibi = 1400, smooth = False, min_dur=True, epoched=True, rm_artifacts=False, detect_peaks=True, sampling_freq=250):

        """ The constructor for Auto_Param class.

        Parameters
        ----------
        fname : str
            file name
        fpath : str
            path to file
        mw_size_opt : list of floats (default: [20, 130])
            moving window sizes in ms around which and in between which will be tested
        upshift_opt : list of floats (default: [1.0, 5.0])
            upshift sizes in percentages around which and in between which will be tested
        sm_wn_opt : list of floats ([10, 50])
            smoothing window sizes in ms around which and in between which will be tested
        hb_range : list of floats ([40, 140])
            range of heart beats per minute that would be appropriate
        min_ibi : float (default: 500)
            minimum IBI length in ms
        max_ibi : float (default: 1400)
            maximum IBI length in ms
        smooth : bool (default: False)
            Whether raw signal should be smoothed before peak detections. Set True if raw data has consistent high frequency noise
            preventing accurate peak detection
        min_dur : bool (default:True)
            only load files that are >= 5 minutes long
        epoched: bool (default: True)
            whether file was epoched using ioeeg
        rm_artifacts : bool (default: False)
            apply IBI artifact removal algorithm
        detect_peaks : bool (default: True)
            whether peaks should be detected
        sampling_freq : int (default: 250)
            number of samples taken per second
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
                                },
                        'testing_info':{'mw_size_opt':mw_size_opt, 
                                'min_ibi':min_ibi, 
                                'max_ibi':max_ibi,
                                'sm_wn_opt': sm_wn_opt,
                                'upshift_opt': upshift_opt,
                                'hb_range': hb_range
                                },
                        'analysis_info':{'smooth': smooth} 
                        }

        
        if epoched == True:
            self.metadata['file_info']['epoch'] = epoch

        self.prep(mw_size_opt, upshift_opt, sm_wn_opt)

        if self.halt == False:
            self.num_beats(fname, fpath, hb_range, sampling_freq)
            self.broad_test(fname, fpath, min_ibi, max_ibi, detect_peaks)
    """     if self.run_precise == True:
                self.precise_test1(fname, fpath, min_ibi, max_ibi, detect_peaks)
                if self.run_precise2 == True:
                    self.precise_test2(fname, fpath, min_ibi, max_ibi, detect_peaks)
            self.output(fname, fpath, detect_peaks) """
        
    def prep(self, mw_size_opt, upshift_opt, sm_wn_opt):
        """ The function to set variables to test and determine if the inputs for mw_size_opt, upshift_opt and sm_wn_opt are appropriate.

        Parameters:
            mw_size_opt : list of floats (default: [20, 130])
                moving window sizes in ms around which and in between which will be tested
            upshift_opt : list of floats (default: [1.0, 5.0])
                upshift sizes in percentages around which and in between which will be tested
            sm_wn_opt : list of floats ([10, 50])
                smoothing window sizes in ms around which and in between which will be tested
        """
        self.halt = False # used in init command to cause the code to not run any more methods if set to true
        if len(mw_size_opt) > 2 or len(upshift_opt) > 2 or len(sm_wn_opt) > 2: # if a list of greater than two numbers input into any of these parameters it will not run the code
            print('Please set 2 or less initial points to be tested for each parameter.')
            self.halt = True #make it stop so they can fix the input

        #find the three points to test around and set the differences between pts to a variable
        self.broad_up = np.linspace(start = upshift_opt[0], stop = upshift_opt[1], num = 3)
        self.up_diff = self.broad_up[1] - self.broad_up[0]
        self.broad_mw = np.linspace(start = mw_size_opt[0], stop = mw_size_opt[1], num = 3)
        self.mw_diff = self.broad_mw[1] - self.broad_mw[0]
        if self.metadata['analysis_info']['smooth'] == True:
            self.broad_sm = np.linspace(start = sm_wn_opt[0], stop = sm_wn_opt[1], num = 3)
            self.sm_diff = self.broad_sm[1] - self.broad_sm[0]

    def num_beats(self, fname, fpath, hb_range, sampling_freq):
        """ The function to determine if the appropriate number of detected peaks and IBIs in the segment.

        Parameters:
            fname : str
                file name
            fpath : str
                path to file
            hb_range : list of floats ([40, 140])
                range of heart beats per minute that would be appropriate
            sampling_freq : int (default: 250)
                number of samples taken per second
        """
        filepath = os.path.join(fpath, fname) 
        blanks = []
        #load EKG
        data = pd.read_csv(filepath, header = [0, 1], index_col = 0, parse_dates=True)['EKG']
        #check number of Nans and how much time there is no EKG data
        blanks.append(len(data) - pd.DataFrame.count(data))
        blank_time = len(blanks) * 1/sampling_freq
        # Check cycle length
        cycle_len_secs = (data.index[-1] - data.index[0]).total_seconds()
        min_beats = (hb_range[0] * (cycle_len_secs - blank_time) / 60) #minimum beat number based on low end of hb_range
        max_beats = (hb_range[1] * (cycle_len_secs - blank_time) / 60) #maximum beat number based on high end of hb_range
        self.beat_range = [min_beats, max_beats] #set the plausible number of beats range as a list
        self.ibi_range = [min_beats -1, max_beats -1] #set the plausible number of ibis range as a list


    def broad_test(self, fname, fpath, min_ibi, max_ibi, detect_peaks):
        """ The function to run an initial test of R peak detections using combinations of moving window and upshift parameters and determine where to test more precisely.

        Parameters:
            fname : str
                file name
            fpath : str
                path to file
            mw_size_opt : list of floats (default: [20, 130])
                fill in when better idea of what you are doing
            upshift_opt : list of floats (default: [1.0, 5.0])
                fill in when better idea of what you are doing
            min_ibi : float (default: 500)
                minimum IBI length in ms
            max_ibi : float (default: 1400)
                maximum IBI length in ms
            detect_peaks : bool (default: True)
                whether peaks should be detected
        """
        self.param_condit= [] # this list will contain lists of the parameters tested and the approximate false detection rate for each
        self.zero_val = False   #if the approximate false detection rate is zero then loop will stop below. No need to keep testing if no false detections
        self.run_precise = True # will run more precise testing unless set to false
        no_peak_count = 0 # counter of the number of times no peaks were detected
        test_count = 0 # counter of number of tests that were run
        for up in self.broad_up:
            for mw in self.broad_mw:
                if self.metadata['analysis_info']['smooth'] == True: # if smoothing go this way 
                    for sm in self.broad_sm:
                        test_count = test_count + 1 #increase counter
                        if self.zero_val == True:
                            break #if there has been a run with no false detections then break
                        e = EKG(fname, fpath, detect_peaks, upshift = up, mw_size = mw, sm_wn = sm, smooth = True) #create object with testing parameters
                        prob_false = [] #empty list that will contain the ibi's that are unlikely to be real
                        for x in e.rr: # for the values in the detected ibi's
                            if x < min_ibi: # if the ibi is less than the minimal probable ibi add to probably false
                                prob_false.append(x)
                            elif x > max_ibi: # same for max
                                prob_false.append(x) 
                        if len(e.rr) >= self.ibi_range[0] and len(e.rr) <= self.ibi_range[1]: # if the total number of ibis is within the range of probable number of ibis for the segment
                            approxparam = 100 * len(prob_false)/(len(e.rr)) # set the approximate parameter for false detection rate by dividing the number of false beats by total beats
                            self.param_condit.append([up, mw, sm, approxparam]) # make a list that when called would show the parameters that lead to which false detection rate
                            if approxparam == 0:
                                self.zero_val = True #there was a zero val for approx parameter 
                                self.run_precise = False # do not run precise if this is the case
                        if len(e.rr) == 0: # if no peaks were detected add to the no peak count
                            no_peak_count = no_peak_count + 1
                else:
                    test_count = test_count + 1 #increase counter
                    if self.zero_val == True:
                        break #if there has been a run with no false detections then break
                    e = EKG(fname, fpath, detect_peaks, upshift = up, mw_size = mw, smooth = False) #create object with testing parameters
                    prob_false = [] #empty list that will contain the ibi's that are unlikely to be real
                    for x in e.rr: # for the values in the detected ibi's
                        if x < min_ibi: # if the ibi is less than the minimal probable ibi add to probably false
                            prob_false.append(x)
                        elif x > max_ibi: # same for max
                            prob_false.append(x) 
                    if len(e.rr) >= self.ibi_range[0] and len(e.rr) <= self.ibi_range[1]: # if the total number of ibis is within the range of probable number of ibis for the segment
                        approxparam = 100 * len(prob_false)/(len(e.rr)) # set the approximate parameter for false detection rate by dividing the number of false beats by total beats
                        self.param_condit.append([up, mw, approxparam]) # make a list that when called would show the parameters that lead to which false detection rate
                        if approxparam == 0:
                            self.zero_val = True #there was a zero val for approx parameter 
                            self.run_precise = False # do not run precise if this is the case
                    if len(e.rr) == 0: # if no peaks were detected add to the no peak count
                        no_peak_count = no_peak_count + 1

        if len(self.param_condit) == 0 and test_count != no_peak_count: #if param_condit list is empty (the number of IBIS was never within the appropriate range) and they were all no peak detections
            print("Abnormal number of beats detected with all combinations of parameters tested. Plot data using plotpeaks method to check raw data.")

        if test_count == no_peak_count: #if they were all no peak detections
            print("No peaks were detected with any of these parameters. Plot data using plotpeaks method to check raw data.")

        if len(self.param_condit) == 0: #give them the option to still move forward to precise testing if they choose
            run_precise = input('Do you still want to run the more precise test? [y/n]')
            print('\n')
            if run_precise != 'y' or 'n':
                print('please enter y or n')
                run_precise = input('Do you still want to run the more precise test? [y/n]')
                print('\n')
            if run_precise == 'y':
                self.run_precise = True
            if run_precise == 'n':
                self.run_precise = False

        if self.zero_val == True or self.run_precise == True: #if there was a zero value for false detection rate or if we will run precise
            min_err = min(i[-1] for i in self.param_condit) #get the minimum "approxparam" which will be the 0
            for lst in self.param_condit:
                if min_err in lst: # if the minimum error is in the list within the lists of param condit
                    indx = self.param_condit.index(lst) # get the index number of that list
            optimal = self.param_condit[indx]
            self.optimal = optimal # the list containing the parameters and the approximate false rate will be set as optimal
            if self.zero_val == True: # if there was a 0 percent false rate then print the parameters that lead to that
                if self.metadata['analysis_info']['smooth'] == False:
                    print("The optimal upshift is " + str(optimal[0]) + "%." + " The optimal moving window size is " + str(optimal[1]) + " ms. This gave an approximate false detection rate of " + str(optimal[2]) + "%")
                else:
                    print("The optimal upshift is " + str(optimal[0]) + "%." + " The optimal moving window size is " + str(optimal[1]) + " ms. The optimal smoothing window is " + str(optimal[2]) + " ms. This gave an approximate false detection rate of " + str(optimal[-1]) + "%")
            else:
                # give each optimal paramater a variable for clarity and make it global bc will need in precise test
                self.optml_up1 = optimal[0]
                self.optml_mw1 = optimal[1]
                if self.metadata['analysis_info']['smooth'] == True:
                    self.optml_sm1 = optimal[2]

                # default set that manual input low bounds arent happening
                self.manual_low_up1 = False
                self.manual_low_mw1 = False
                self.manual_low_sm1 = False

                #show to user that optimal smoothing window is not the smallest valid smoothing window
                if self.metadata['analysis_info']['smooth'] == True and len(self.metadata['testing_info']['sm_wn_opt']) != 1:
                    #find minimum of smoothing windows in param condit
                    min_sm = min(i[2] for i in self.param_condit)
                    if self.optml_sm1 != min_sm: # if the optimal isnt the minimum of smoothing windows in param condit. not min of all smoothing windows because some may have been tested and not added to param condit because ibis werent right.
                        print('The optimal smoothing window was determined to be {}, which is not the smallest option.'.format(self.optml_sm1))
                        print('The larger the smoothing window the less precise the r peak detections.')
                        #find percentage false peaks with smallest smoothing window
                        small_sm = [] #make param condit list of just the smallest smoothing window
                        for ls in self.param_condit:
                            if ls[2] == min_sm:
                                small_sm.append(ls)
                        min_lo_sm_err = min(i[-1] for i in small_sm) #minmum error for smallest smoothing window
                        print('The percentage of false peaks with the smallest valid smoothing window of {} is {}% Compared to {} with the optimal smoothing widow of {}'.format(min_sm, min_lo_sm_err, self.optimal[-1], self.optml_sm1))
                        lrg_sm = input('Do you want to use the keep going with the larger smoothing window? [y/n]')
                        if lrg_sm == 'y':
                            self.optml_sm1 = self.optml_sm1
                        if lrg_sm == 'n':
                            self.optml_sm1 = min_sm
                            
                #for upshift set where you will test more precisely
                if len(self.metadata['testing_info']['upshift_opt']) == 2: # if 3 options were given
                    low_up_test = self.optml_up1 - (self.up_diff/2) # test precisely halfway above and below the upshift that gave the optimal detection (if 1,3,5 input and 5 deemed best will test 4 and 6)
                    high_up_test = self.optml_up1 + (self.up_diff/2) # up diff is from before the difference between the numbers given
        
                if len(self.metadata['testing_info']['upshift_opt']) == 1: # if only one was given that was the best one so use that
                    self.up_precisetest = [self.optml_up1]


                #for mw set where test more precisely
                if len(self.metadata['testing_info']['mw_size_opt']) == 2 :
                    low_mw_test = self.optml_mw1 - (self.mw_diff/2)
                    high_mw_test = self.optml_mw1 + (self.mw_diff/2)
                if len(self.metadata['testing_info']['mw_size_opt']) == 1:
                    self.mw_precisetest = [self.optml_mw1]

                #for sm_wn set where test more precisely
                if self.metadata['analysis_info']['smooth'] == True:
                    if len(self.metadata['testing_info']['sm_wn_opt']) == 2:
                        low_sm_test = self.optml_sm1 - (self.sm_diff/2)
                        high_sm_test = self.optml_sm1 + (self.sm_diff/2)
    
                    if len(self.metadata['testing_info']['sm_wn_opt']) == 1:
                        self.sm_precisetest = [self.optml_sm1]

                if len(self.metadata['testing_info']['upshift_opt']) != 1: #do not need to do all this if its 1
                    #set global variable so can be used in precise methods and also deal with what if upshift very low
                    if low_up_test >= 0.5:
                        self.up_precisetest = [low_up_test, high_up_test]
                    if low_up_test < 0.5: # less than this is unlikely to work 
                        is_low_up_bound = input('Current unprecise optimal upshift detection is {}%. Calculated lower value for further testing is {}% . Do you want to test more precisely around a value lower than the current optimal detection? [y/n]'.format(self.optml_up1, low_up_test))
                        print('\n')
                        if is_low_up_bound == 'y': #give option to set lower bound 
                            low_up_bound = float(input('What lower value do you want to test around? (Must be greater than 0 and less than the current optimal detection of {}.) [number]'.format(self.optml_up1)))
                            print('\n')
                            self.up_precisetest = [low_up_bound, high_up_test] #use this lower bound
                            self.manual_low_up1 = True
                        if is_low_up_bound == 'n':
                            self.up_precisetest = [high_up_test] # just use the upper test 

                if len(self.metadata['testing_info']['mw_size_opt']) != 1:
                    #set global variable for mw, deal with mw below 0
                    if low_mw_test > 0:
                        self.mw_precisetest = [low_mw_test, high_mw_test]
                    if low_mw_test <= 0:
                        is_low_mw_bound = input('Current unprecise optimal moving window detection is {} ms. Calculated lower value for further testing is {} ms . Do you want to test more precisely around a value lower than the current optimal detection? [y/n]'.format(self.optml_mw1, low_mw_test))
                        print('\n')
                        if is_low_mw_bound == 'y':
                            low_mw_bound = float(input('What lower value do you want to test around? (Must be greater than 0 and less than the current optimal detection of {} ms. Do not type ms)'.format(self.optml_mw1)))
                            print('\n')
                            #do something here so that if less than 0 will ask question again and not move fwd?
                            self.mw_precisetest = [low_mw_bound, high_mw_test] 
                            self.manual_low_mw1 = True
                        if is_low_mw_bound == 'n':
                            self.mw_precisetest = [high_mw_test]

                #set global variable for sm_wn, deal with sm_wn at or below 0
                if self.metadata['analysis_info']['smooth'] == True:
                    self.no_low_smooth = False #default set to False, this means have the "low bound" be no smoothing 
                    if len(self.metadata['testing_info']['sm_wn_opt']) != 1:
                        if low_sm_test > 0: #no harm in a low smoothing window
                            self.sm_precisetest = [low_sm_test, high_sm_test]
                        else: # if less than 0 wont work or if 0 then do they want no smooth?
                            is_low_up_bound = input('Current unprecise optimal smoothing window detection is {}%. Calculated lower value for further testing is {}%, which is below zero.  Do you want to test more precisely around a value lower than the current optimal detection? [y/n]'.format(self.optml_up1, low_up_test))
                            print('\n')
                            if is_low_up_bound == 'y': #give option to set lower bound 
                                low_up_bound = float(input('What lower value do you want to test around? (Must be greater than 0 and less than the current optimal detection of {}.) [number]'.format(self.optml_up1)))
                                print('\n')
                                self.up_precisetest = [low_up_bound, high_up_test] #use this lower bound
                                self.manual_low_up1 = True
                            if is_low_up_bound == 'n': #give option to do no smooth
                                does_smooth = input('Do you want to test no smoothing as well as your larger precise smoothing window? [y/n]')
                                self.up_precisetest = [high_up_test] #the upper test will be there either way
                                if no_smooth == 'y': #if they  want to try a no smooth
                                    self.no_low_smooth == True #test a no smoothintg as the lower bound for precise
                         
          
    def precise_test1(self, fname, fpath, min_ibi, max_ibi, detect_peaks): 
        """ The function to run a more precise test of R peak detections using combinations of moving window and upshift parameters determined in broad_test and determine where to test more precisely.

        Parameters:
            fname : str
                file name
            fpath : str
                path to file
            min_ibi : float (default: 500)
                minimum IBI length in ms
            max_ibi : float (default: 1400)
                maximum IBI length in ms
            detect_peaks : bool (default: True)
                whether peaks should be detected
        """
        test_count = 0 #this segment is essentially same code as in broad test
        no_peak_count = 0 # counter so if = the number of tests then we know that no hb were ever detected
        self.run_precise2 = True
        for up in self.up_precisetest:
            for mw in self.mw_precisetest:
                if self.metadata['analysis_info']['smooth'] == True:
                    for sm in self.sm_precisetest:
                        test_count = test_count + 1
                        if self.zero_val == True:
                            break
                        e = EKG(fname, fpath, detect_peaks, upshift= up, mw_size= mw, sm_wn = sm, smooth = True)
                        prob_false = []
                        for x in e.rr:
                            if x < min_ibi:
                                prob_false.append(x)
                            elif x > max_ibi:
                                prob_false.append(x) 
                        if len(e.rr) >= self.ibi_range[0] and len(e.rr) <= self.ibi_range[1]:
                            approxparam = 100 * len(prob_false)/(len(e.rr))
                            self.param_condit.append([up, mw, sm, approxparam]) #we want to add to the same param condit so we have a log of all that has been tested
                            if approxparam == 0:
                                self.zero_val = True
                                self.run_precise2 = False
                        if len(e.rr) == 0:
                            no_peak_count = no_peak_count + 1
                if self.metadata['analysis_info']['smooth'] == False or self.no_low_smooth == True: #have run the above for smooth with upper window we want lower to just be no smoothing so run below for lower smooth bound
                    test_count = test_count + 1
                    if self.zero_val == True:
                        break
                    e = EKG(fname, fpath, detect_peaks, upshift= up, mw_size= mw, smooth = False)
                    prob_false = []
                    for x in e.rr:
                        if x < min_ibi:
                            prob_false.append(x)
                        elif x > max_ibi:
                            prob_false.append(x) 
                    if len(e.rr) >= self.ibi_range[0] and len(e.rr) <= self.ibi_range[1]:
                        approxparam = 100 * len(prob_false)/(len(e.rr))
                        self.param_condit.append([up, mw, approxparam]) #we want to add to the same param condit so we have a log of all that has been tested
                        if approxparam == 0:
                            self.zero_val = True
                            self.run_precise2 = False
                    if len(e.rr) == 0:
                        no_peak_count = no_peak_count + 1

        if len(self.param_condit) == 0 and test_count != no_peak_count:
            print("Abnormal number of beats detected with all combinations of parameters tested. Plot data using plotpeaks method to check raw data.")

        if test_count == no_peak_count:
            print("No peaks were detected with any of these parameters. Plot data using plotpeaks method to check raw data.")

        if len(self.param_condit) == 0:
            run_precise = input('Do you still want to run the more precise test? [y/n]')
            print('\n')
            if run_precise != 'y' or 'n':
                print('please enter y or n')
                run_precise = input('Do you still want to run the more precise test? [y/n]')
                print('\n')
            if run_precise == 'y':
                self.run_precise2 = True
            if run_precise == 'n':
                self.run_precise2 = False

        if self.zero_val == True or self.run_precise2 == True: #if there was a zero value for false detection rate or if we will run precise
            min_err = min(i[-1] for i in self.param_condit) #get the minimum "approxparam" which will be the 0
            for lst in self.param_condit:
                if min_err in lst: # if the minimum error is in the list within the lists of param condit
                    indx = self.param_condit.index(lst) # get the index number of that list
            self.optimal = self.param_condit[indx] # the list containing the parameters and the approximate false rate will be set as optimal
            if self.metadata['analysis_info']['smooth'] == True and len(self.optimal) == 3: # if was testing smooth but now best is when not smoothed 
                self.metadata['analysis_info']['smooth'] = False # no longer smooth
            if self.zero_val == True: # if there was a 0 percent false rate then print the parameters that lead to that
                if self.metadata['analysis_info']['smooth']:
                    print("The optimal upshift is " + str(self.optimal[0]) + "%." + " The optimal moving window size is " + str(self.optimal[1]) + " ms. This gave an approximate false detection rate of " + str(self.optimal[2]) + "%")
                else:
                    print("The optimal upshift is " + str(self.optimal[0]) + "%." + " The optimal moving window size is " + str(self.optimal[1]) + " ms. The optimal smoothing window is " + str(self.optimal[2]) + " ms. This gave an approximate false detection rate of " + str(optimal[-1]) + "%")
            else:
                # give each optimal paramater a variable for clarity and make it global bc will need in next precise test
                self.optml_up2 = self.optimal[0]
                self.optml_mw2 = self.optimal[1]
                if self.metadata['analysis_info']['smooth'] == True:
                    self.optml_sm2 = self.optimal[2]

                #show to user that optimal smoothing window is not the smallest valid smoothing window
                if self.metadata['analysis_info']['smooth'] == True and len(self.metadata['testing_info']['sm_wn_opt']) != 1:
                    #find minimum of smoothing windows in the precise smooth test ran 
                    min_sm = min(self.sm_precisetest)
                    if self.optml_sm2 != min_sm: # if the optimal isnt the minimum of smoothing windows in param condit. not min of all smoothing windows because some may have been tested and not added to param condit because ibis werent right.
                        print('The optimal smoothing window was determined to be {}, which is not the smallest option.'.format(self.optml_sm2))
                        print('The larger the smoothing window the less precise the r peak detections.')
                        #find percentage false peaks with smallest smoothing window
                        small_sm = [] #make param condit list of just the smallest smoothing window
                        for ls in self.param_condit:
                            if ls[2] == min_sm:
                                small_sm.append(ls)
                        min_lo_sm_err = min(i[-1] for i in small_sm) #minmum error for smallest smoothing window
                        print('The percentage of false peaks with the smallest valid smoothing window of {} is {}% Compared to {} with the optimal smoothing widow of {}'.format(min_sm, min_lo_sm_err, self.optimal[-1], self.optml_sm2))
                        lrg_sm = input('Do you want to use the keep going with the larger smoothing window? [y/n]')
                        if lrg_sm == 'y':
                            self.optml_sm2 = self.optml_sm2
                        if lrg_sm == 'n':
                            self.optml_sm2 = min_sm    

                #for mw set where test more precisely
                if len(self.metadata['testing_info']['mw_size_opt']) != 1: #if there was more than one number input
                    if self.manual_low_mw1 == True and self.optml_mw1 == self.mw_precisetest[0]: #if manually lower bound set, and the mw which gave best detection is that manually input lower bound 
                        potential_low_mw = (self.mw_precisetest[0] - (self.optml_mw1 - self.mw_precisetest[0])/2) #take halfway between the optimal mw determined in broad test (probably 1, the lowest input) and the manually determined low bound for precise test and then subtract that from the manual low bound to get this potential value
                        high_mw_test = self.mw_precisetest[0] + (self.optml_mw1 - self.mw_precisetest[0])/2 #higher test set as halfway between the manual low value deemed optimal and the previous value deemed optimal
                        if potential_low_mw > 0: # if the potential value is greater than 0 you can use it
                            low_mw_test = potential_low_mw
                            self.mw_precisetest2 = [low_mw_test, high_mw_test]
                        else: #if not just dont use a lower test you have gone low enough
                            print("Value to be used as the lower test of the next moving window precise test calculated to be {} ms which is below zero and too low to continue. Will test more precisely with upper value only.".format(potential_low_mw))
                            self.mw_precisetest2 = [high_mw_test]
                    else: #if the manual lower test wasnt determined to be best
                        high_mw_test = self.optml_mw2 + (self.mw_diff/4) # further tests is the optimally determined one from this round plus or minuma quarter differnce (ex if 20, 75, 130 then 20 deemed best, then manual input and 47.5 tested, 47.5 deemed best so 33.75 and 61.75 tested)
                        low_mw_test = self.optml_mw2 - (self.mw_diff/4)
                        if low_mw_test <= self.mw_precisetest[0]: # set if calculated lower test is lower than the manually set test that was deemed not optimal dont run it. just run up test
                            self.mw_precisetest2 = [high_mw_test]
                        else:
                            self.mw_precisetest2 = [low_mw_test, high_mw_test]
                if len(self.metadata['testing_info']['mw_size_opt']) == 1: # if only one was inputed just use that one
                    self.mw_precisetest2 = [self.optml_mw2]

                #for upshift set where test more precisely, same as above
                if len(self.metadata['testing_info']['upshift_opt']) != 1:
                    if self.manual_low_up1 == True and self.optml_up1 == self.up_precisetest[0]:
                        potential_low_up = (self.up_precisetest[0] - (self.optml_up1 - self.up_precisetest[0])/2)
                        high_up_test = self.up_precisetest[0] + (self.optml_up1 - self.up_precisetest[0])/2
                        if potential_low_up > 0:
                            low_up_test = potential_low_up
                            self.up_precisetest2 = [low_up_test, high_up_test]
                        else:
                            print("Value to be used as the lower test of the next upshift precise test calculated to be {}% which is below zero and too low to continue. Will test more precisely with upper value only.".format(potential_low_up))
                            self.up_precisetest2 = [high_up_test]
                    # set if calculated lower test is lower than the manually set test that was deemed not optimal dont run it. just run up test
                    else: #if manual wasnt true or if it was but the optimal wasnt the manual input
                        low_up_test = self.optml_up2 - (self.up_diff/4)
                        high_up_test = self.optml_up2 + (self.up_diff/4)
                        if low_up_test <= self.up_precisetest[0]: #if the calculated lower test is lower than the manual that wasnt optimal
                            self.up_precisetest2 = [high_up_test]
                        else:
                            self.up_precisetest2 = [high_up_test, low_up_test]
                if len(self.metadata['testing_info']['upshift_opt']) == 1:
                    self.up_precisetest2 = [self.optml_up2] 

                #for smoothing window set where test more precisely
                if self.metadata['analysis_info']['smooth'] == True:
                    if len(self.metadata['testing_info']['sm_wn_opt']) != 1:
                        if self.manual_low_sm1 == True and self.optml_sm1 == self.sm_precisetest[0]:
                            potential_low_sm = (self.sm_precisetest[0] - (self.optml_sm1 - self.sm_precisetest[0])/2)
                            high_sm_test = self.sm_precisetest[0] + (self.optml_sm1 - self.sm_precisetest[0])/2
                            if potential_low_sm > 0:
                                low_sm_test = potential_low_sm
                                self.sm_precisetest2 = [low_sm_test, high_sm_test]
                            else:
                                print("Value to be used as the lower test of the next smoothing window precise test calculated to be {}% which is below zero and too low to continue. Will test more precisely with upper value only.".format(potential_low_up))
                                self.sm_precisetest2 = [high_sm_test]
                    # set if calculated lower test is lower than the manually set test that was deemed not optimal dont run it. just run up test
                        else:
                            low_sm_test = self.optml_sm2 - (self.sm_diff/4)
                            high_sm_test = self.optml_sm2 + (self.sm_diff/4)
                            if low_sm_test <= self.sm_precisetest[0]:
                                self.sm_precisetest2 = [high_sm_test]
                            else:
                                self.sm_precisetest2 = [high_sm_test, low_sm_test]
                    if len(self.metadata['testing_info']['sm_wn_opt']) == 1:
                        self.sm_precisetest2 = [self.optml_sm2] 

    def precise_test2(self, fname, fpath, min_ibi, max_ibi, detect_peaks):
        """ The function to run a more precise test of R peak detections using combinations of moving window and upshift parameters determined in precise_test1.

        Parameters:
            fname : str
                file name
            fpath : str
                path to file
            min_ibi : float (default: 500)
                minimum IBI length in ms
            max_ibi : float (default: 1400)
                maximum IBI length in ms
            detect_peaks : bool (default: True)
                whether peaks should be detected
        """

        no_peak_count = 0 # counter so if = the number of tests then we know that no hb were ever detected
        test_count = 0
        for up in self.up_precisetest2:
            for mw in self.mw_precisetest2:
                if self.metadata['analysis_info']['smooth'] == True:
                    for sm in self.sm_precisetest2:
                        test_count = test_count + 1
                        if self.zero_val == True:
                            break
                        e = EKG(fname, fpath, detect_peaks, upshift=up, mw_size=mw, sm_wn = sm, smooth = True)
                        prob_false = []
                        for x in e.rr:
                            if x < min_ibi:
                                prob_false.append(x)
                            elif x > max_ibi:
                                prob_false.append(x) 
                        if len(e.rr) >= self.ibi_range[0] and len(e.rr) <= self.ibi_range[1]:
                            approxparam = 100 * len(prob_false)/(len(e.rr))
                            self.param_condit.append([up, mw, sm, approxparam])
                            if approxparam == 0:
                                self.zero_val = True
                        if len(e.rr) == 0:
                            no_peak_count = no_peak_count + 1
                else:
                    test_count = test_count + 1
                    if self.zero_val == True:
                        break
                    e = EKG(fname, fpath, detect_peaks, upshift=up, mw_size=mw, smooth = False)
                    prob_false = []
                    for x in e.rr:
                        if x < min_ibi:
                            prob_false.append(x)
                        elif x > max_ibi:
                            prob_false.append(x) 
                    if len(e.rr) >= self.ibi_range[0] and len(e.rr) <= self.ibi_range[1]:
                        approxparam = 100 * len(prob_false)/(len(e.rr))
                        self.param_condit.append([up, mw, approxparam])
                        if approxparam == 0:
                            self.zero_val = True
                    if len(e.rr) == 0:
                        no_peak_count = no_peak_count + 1

        if len(self.param_condit) == 0 and test_count != no_peak_count:
            print("Abnormal number of beats detected with all combinations of parameters tested. Plot data using plotpeaks method to check raw data.")

        if test_count == no_peak_count:
            print("No peaks were detected with any of these parameters. Plot data using plotpeaks method to check raw data.")

        else:
            min_err = min(i[-1] for i in self.param_condit)
            for lst in self.param_condit:
                if min_err in lst:
                    indx = self.param_condit.index(lst)
            optimal = self.param_condit[indx]
            self.optimal = optimal #update the optimal 
            if self.metadata['analysis_info']['smooth'] == False:
                print("The optimal upshift is " + str(optimal[0]) + "%." + " The optimal moving window size is " + str(optimal[1]) + " ms. This gave an approximate false detection rate of " + str(optimal[2]) + "%")
            else:
                print("The optimal upshift is " + str(optimal[0]) + "%." + " The optimal moving window size is " + str(optimal[1]) + " ms. The optimal smoothing window is " + str(optimal[2]) + " ms. This gave an approximate false detection rate of " + str(optimal[-1]) + "%")
    def output(self, fname, fpath, detect_peaks): #creating of the object
        """ The function to create the EKG object using the parameters deemed optimal.

        Parameters:
            fname : str
                file name
            fpath : str
                path to file
            detect_peaks : bool (default: True)
                whether peaks should be detected
        """

        if self.metadata['analysis_info']['smooth'] == False:
            self.final = EKG(fname, fpath, detect_peaks, upshift=self.optimal[0], mw_size=self.optimal[1])
        if self.metadata['analysis_info']['smooth'] == True:
            self.final = EKG(fname, fpath, detect_peaks, upshift=self.optimal[0], mw_size=self.optimal[1], smooth = True, sm_wn = self.optimal[2])

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

        print('Data smoothed with smoothing window of {} ms.'.format(sm_wn)) 

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
        self.data.insert(1, 'EKG_thres', det_thres) # can remove this for speed, just keep as series

        self.metadata['analysis_info']['mw_size'] = mw_size
        self.metadata['analysis_info']['upshift'] = upshift

        # create empy series for false detections removed and missed peaks added
        self.rpeak_artifacts = pd.Series()
        self.rpeaks_added = pd.Series()
        self.ibi_artifacts = pd.Series()

    def detect_Rpeaks(self, smooth):
        """ detect R peaks from raw signal """

        if smooth == False:
            raw = pd.Series(self.data['Raw'])
        elif smooth == True:
            raw = pd.Series(self.data['raw_smooth'])
        
        thres = pd.Series(self.data['EKG_thres'])
        

        peaks = []
        x = 0
        if raw[len(raw)-1] > thres[len(raw)-1]:
            for h in range(len(raw)-1, 0, -1):
                if raw[len(raw)-(h+1)] < thres[len(raw)-(h+1)]:
                    end = len(raw) - (h+1)
        else:
        	end = len(raw)
        while x < end:
            if raw[x] > thres[x]:
                roi_start = x
                # count forwards to find down-crossing
                for h in range(x, len(raw), 1):
                    if raw[h] < thres[h]:
                        roi_end = h
                        break
    
                # get maximum between roi_start and roi_end
                peak = raw[x:h].idxmax()
                peaks.append(peak)
                # advance the pointer
                x = h
            else:
                x += 1


        self.rpeaks = raw[peaks]

        # get time between peaks and convert to mseconds
        self.rr = np.diff(self.rpeaks.index)/np.timedelta64(1, 'ms')

        # create nn so that ibis can be removed
        self.nn = self.rr
        
        # create rpeaks dataframe and add ibi columm
        rpeaks_df = pd.DataFrame(self.rpeaks)
        ibi = np.insert(self.rr, 0, np.NaN)
        rpeaks_df['ibi_ms'] = ibi
        self.rpeaks_df = rpeaks_df

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

        if self.metadata['analysis_info']['smooth'] == False:
        # define new rpeak
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
            # get indices of ibis greater than threshold
            rm_idx = [i for i, x in enumerate(self.nn) if x > thres]
            print('{} IBIs greater than {} milliseconds detected'.format(len(rm_idx), thres))
            rm = input('Automatically remove? [y/n]: ')
            
            if rm.casefold() == 'y':
                
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


    def calc_fstats(self, itype, method, bandwidth, window, bands=None):
        """ Calculate frequency domain statistics 

        Parameters
        ----------
        itype: str
            interval type (options: 'rr', 'nn')
        method: str, optional (default: 'mt')
            Method to compute power spectra. options: 'welch', 'mt' (multitaper)
        bandwith: float, optional (default: 0.02)
            Bandwidth for multitaper power spectral estimation
        window: str, optional (default: 'hamming')
            Window to use for welch FFT. See mne.time_frequency.psd_array_multitaper for options
        bands: Nonetype
            Frequency bands of interest. Leave as none for default. To do: update for custom bands
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
        self.calc_fbands(method, bands)
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