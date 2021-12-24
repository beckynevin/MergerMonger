This folder is used if you end up running step #2, which involves measuring predictor values, which involves downloading frame images and saving files to this folder.

Galfit, source extractor, and statmorph will all run using images saved in this folder.

### Installing Source Extractor and Galfit
To install source extractor, I used:

```
brew tap brewsci/science
brew install sextractor
```
I recommend doing this over installing yourself because there are some complicated package dependences (fftw-3 and atlas).

This installs source extractor but does not correctly include some of the default files, like the convolutional filter and parameter files. I've included these in this directory with the proper settings for running on SDSS files.

Now test that source extractor will run by typing ```sex``` anywhere. Sorry that the developers are apparently super mature.

To install galfit:

```brew install galfit```
