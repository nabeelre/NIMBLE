### How to make a non-spherical halo with Agama:
First, we need to generate the parameters of a double power law DF

Use the agama script example_doublepowerlaw.exe to do so

**input command:**

```python
agama/exe/example_doublepowerlaw.exe density=dehnen
```

The output is used and adjusted in halo_alone.py

The flattening is performed by increasing the Jz coefficients and decreasing the Jr ones

**output:**

```python
Best-fit parameters of DoublePowerLaw DF (Kullback-Leibler distance=0.000925014):
norm          = 2.17066
J0            = 1.40824
slopeIn       = 1.58489
slopeOut      = 5.09761
steepness     = 1.34007
coefJrIn      = 1.46296
coefJzIn      = 0.76852
coefJrOut     = 1.06949
coefJzOut     = 0.96525
Jcutoff       = 2.07e10
cutoffStrength= 2.91954
```



