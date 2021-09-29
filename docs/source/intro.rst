Quickstart Guide
================

``imcascade`` is a method to fit sources in astronomical images. This is accomplished by modelling the objects using multi-Gaussian expansion which models the galaxy as a mixture of Gaussians. For full details please read our paper here: `here <https://arxiv.org/abs/2109.13262>`_.

What follows is a (very) brief introduction to the basic usage of ``imcascade``, please read the users guide for a more in depth discussion

The bare minimum needed to run ``imcascade`` is

1. ``img``: A ``numpy`` array representing a cutout of an image with the object of interest at the center

2. ``sig``: The widths of the Gaussian components used to model the galaxy. Generally these are logaritmically spaced from ~1 pixel to ~10 estimated effective radius of the object


The following inputs are technically optional but are very often used when running ``imcascade`` in a realistic setting,

1. ``psf_sig`` and ``psf_a``: the widths and fluxes for a Gaussian decomposition of the point spread function

2. ``weight``: A ``numpy`` array containing the pixel by pixel weights used in the fitting, usually the inverse variance.

3. ``mask``: A ``numpy`` array containing the pixel mask, i.e. neighbouring sources that should not be included.


Once these inputs have been assembled a ``Fitter`` instance  can be initialized. This class contains methods to run chi squared minimization and Bayesian inference,

.. code-block:: python

  from imcascade import Fitter

  fitter = Fitter(img,sig,psf_sig,psf_a, weight = weight, mask = mask)

  fitter.run_ls_min() #To run chi^2 minmization

  ### or

  fitter.run_dynesty(method = 'express') #For Bayesian inference

  fitter.save_results('./my_imcascade_results.asdf')

The analysis of results from ``imcascade`` is non-trivial as the free parameters are the fluxes of each Gaussian component, which do not easily map to common morphological quantities of interest. Therefore we have included the ``ImcascadeResults`` class to help with the analysis

.. code-block:: python

  from imcascade import ImcascadeResults

  res = ImcascadeResults(fitter)
  #Can give Fitter instance to initialize after calling run_ls_min or run_dynesty

  # Alternatively can initialize using a saved file
  # >>> res = ImcascadeResults('./my_imcascade_results.asdf')

  basic_quantities = res.run_basic_analysis()
  #calculates total flux, effective radius and other commonly used quantities

Additionally ``ImcascadeResuts`` contaains other methods to caluclate the recovered surface brightness profile, curve-of-growth and others

And with that you have fit your first object using ``imcascade``!! Please see the In depth example page to see a much more detailed example or the advanced guide for a discussion of more advanceded features.
