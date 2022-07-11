# Cardiopy

A flexibile package for R-peak detection and heart rate variability analysis of single-lead EKG data. <br>
    <img src="https://github.com/CardioPy/CardioPy/blob/master/example_run/advice_images/example_detections.png">
    
Full documentation is available [here](https://www.biorxiv.org/content/10.1101/2020.10.06.328856v1). If you use CardioPy or a derivative in your work, please cite: <br>
Gottshall, J. L., Recoder, N., Schiff, N. D. (2020). CardioPy: An open-source heart rate variability analysis toolkit for single-lead EKG. boiRxv. doi: 10.1101/2020.10.06.328856

## How to use Cardiopy
Cardiopy can be used in two ways:<br>
   1. __As a preprocessing module for the import and cleaning of clinical EKG data in conjuction
      with HRV analyses by standard software packages.__ For this use, run through feature sets 1 and 2 (listed below). The exported '*_nn.txt*' file is compatible with all major HRV software packages <br>
   2. __As a stand-alone HRV analysis toolkit.__ For this use, continue through the workflow from feature set 1 through 4 (listed below). To ensure analytic reproducibilty, we highly recommend exporting cleaned nn detections at feature set 2.

## Features
__1. Data preprocessing and cleaning__<br>
   * Load single-lead EKG data<br>
   * Detect R-peaks using the Pan Tompkins method
      - Option to detect R-peaks with flexible thresholding parameters for adjustment to noisy data and varying amplitudes<br>
      - Option to filter especially noisy data prior to peak detection<br>
   * Built-in detection visualization methods<br>
   * Simple artifact removal methods for manual inspection of detected peaks<br>
  
__2. Export methods for cleaned peak detections__<br>
   * Compatible with commonly used software such as Kubios HRV and Artiifact<br>
   
__3. HRV analysis methods__<br>
   * Standard time-domain statistics<br>
   * Standard frequency domain statistics<br>
      - Option for regular and denoised Multitaper power spectral estimates<br>
   * Confidence intervals
      - Bootstrapping or Chi Squared methods
    
__4. HRV statistics export__<br>
   * Single-file report exports in json format<br>
   * Multi-file exports into .csv spreadsheets for group statistics<br>

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install CardioPy.

```bash
pip install cardiopy
```

## Usage
Best when run with jupyter notebook. For detailed instructions download the [example jupyter notebook file](https://github.com/CardioPy/CardioPy/blob/master/example_run/CardioPy_Example_Analysis.ipynb) and the [example jupyter notebook file for manual cleaning](https://github.com/CardioPy/CardioPy/blob/master/example_run/CardioPy_Example_Analysis_Manual.ipynb), as well as the [de-identified data segment](https://github.com/CardioPy/CardioPy/blob/master/example_run/HCXXX_2001-01-01_awake_cycle1_epoch1_222000.csv) from [github](https://github.com/CardioPy/CardioPy/blob/master/example_run) <br>
   *For optimal performance, close figure interactions ('off' button on the top right corner) when finished with each window.*

### Parameter Optimization & Cleaning Tips
* Remove false interbeat intervals LAST, after all cleaning (addition/removal of peaks) has been done.
* To maintain integrity of the artifact logs:
   - Only remove incorrectly added peaks with EKG.undo_add_peak NOT with EKG.rm_peak.
   - Only re-add incorrectly removed peaks with EKG.undo_rm_peak NOT with EKG.add_peak.

* If R peak detections are not accurate, try using the manual thresholding method

* If R peak detections are not accurate when using the manual thresholding method, try:
   1. changing the moving window size
   2. changing the upshift percentage
   3. both<br>
        <img src="https://github.com/CardioPy/CardioPy/blob/master/example_run/advice_images/EKG_paramshift.png">

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
BSD 3-Clause

## Roadmap
The authors plan for future versions of CardioPy to include:
* Support for additional commonly used data formats
* A graphical user interface
