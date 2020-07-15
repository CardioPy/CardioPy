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

## Usage
Best when run with jupyter notebook. For detailed instructions see (will insert how to get to the example jupyter notebook file). (Section to be expanded as I see what can/can't be shown with the notebook file) 

In cases where the R peak isn't very pronounced use smaller moving windows as well as low upshift percentages.
For example in this segment with upshift of 1.7% and moving window of 50ms detection is not optimal.
![bad_example](https://github.com/CardioPy/CardioPy/tree/master/example_run/advice_images/example_bad_mw.png)

When changed to a moving window of 20ms R peak detection is accurate.
![good_example](https://github.com/CardioPy/CardioPy/tree/master/example_run/advice_images/example_good_mw.png)

## Installation
(Will edit once run it myself to insure details are clear step by step)

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install CardioPy.

```bash
pip install CardioPy
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
TBD

## Roadmap
The authors plan for the next version of CardioPy to include automatic parameter detection. This would include upshift, moving window and smoothing window suggestions for optimal peak detection.