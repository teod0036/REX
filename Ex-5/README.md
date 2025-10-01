This directory contains python source code that might be helpful when solving
the self-localization exercise.

The only file that needs editing to create a working system
is ```selflocalize.py```, but it is recommended to have a brief look at the rest
of the source code. In particular it is recommended to look at ```particle.py```
to see how particles are represented.

We indicate where you might need to make changes to ```selflocalize.py``` by lines saying:
```python
  # XXX: You do this
````
So look for 'XXX' to see where you should concentrate your efforts. But feel free to restructure 
the code as you see fit.

The program uses the camera class found in ```camera.py``` to access the camera and do the image
analysis. This version includes some roughly right camera calibrations in the constructor of the Camera class. 
If you like you can change calibration parameters based on your own estimates of the focal length.


Kim Steenstrup Pedersen, september, 2015, 2017, 2018, 2020, 2025.

Søren Hauberg, august 21st, 2006.
