# Cardiopy

A flexibile package for R-peak detection and heart rate variability analysis of single-lead EKG data. <br>

**Note: This is a work in progress. Complete documentation and installation instructions coming soon!**

## How to use Cardiopy
Cardiopy can be used in two ways:<br>
   1. __As a preprocessing module for the import and cleaning of clinical EKG data in conjuction
		with HRV analyses by standard software packages.__ For this use, run through feature sets 1 and 2 (listed below). The exported '*_nn.txt*' file is compatible with all major HRV software packages <br>
   2. __As a stand-alone HRV analysis toolkit.__ For this use, continue through the workflow from feature set 1 through 4 (listed below). To ensure analytic reproducibilty, we highly recommend exporting cleaned nn detections at feature set 2.

## Features
__1. Data preprocessing and cleaning__<br>
   * Load single-lead EKG data<br>
   * Detect R-peaks with flexible thresholding parameters for adjustment to noisy data and varying amplitudes<br>
		- Option to filter especially noisy data prior to peak detection<br>
   * Built-in detection visualization methods<br>
   * Simple artifact removal methods for manual inspection of detected peaks<br>
  
__2. Export methods for cleaned peak detections__<br>
   * Compatible with commonly used software such as Kubios HRV and Artiifact<br>
   
__3. HRV analysis methods__<br>
   * Standard time-domain statistics<br>
   * Standard frequency domain statistics<br>
		- Option for Multitaper or Welch power spectral estimates<br>
    
__4. HRV statistics export__<br>
   * Single-file report exports in json format<br>
   * Multi-file exports into .csv spreadsheets for group statistics<br>