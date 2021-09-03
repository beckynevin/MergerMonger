### Source Extractor 
Installation and use instructions:

To install source extractor, I used:

```
brew tap brewsci/science
brew install sextractor
```
I recommend doing this over installing yourself because there are some complicated package dependences (fftw-3 and atlas).

This installs source extractor but does not correctly include some of the default files, like the convolutional filter and parameter files. I've included these in this directory with the proper settings for running on SDSS files.

Now test that source extractor will run by typing ```sex``` anywhere. Sorry that the developers are apparently super mature.