dust
====

.. currentmodule:: pinballrt.dust

.. autofunction:: load

.. autoclass:: Dust
   :show-inheritance:

   .. rubric:: Methods Summary

   .. autosummary::

      ~Dust.learn
      ~Dust.fit
      ~Dust.test_model
      ~Dust.run_dust_simulation
      ~Dust.save

   .. rubric:: Methods Documentation

   .. automethod:: learn
   .. automethod:: fit
   .. automethod:: test_model
   .. automethod:: run_dust_simulation
   .. automethod:: save

.. autoclass:: DustOpticalConstants
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~DustOpticalConstants.recipes

   .. rubric:: Methods Summary

   .. autosummary::

      ~DustOpticalConstants.add_coat
      ~DustOpticalConstants.calculate_dhs_opacity
      ~DustOpticalConstants.calculate_opacity
      ~DustOpticalConstants.calculate_optical_constants_on_wavelength_grid
      ~DustOpticalConstants.calculate_size_distribution_opacity
      ~DustOpticalConstants.set_density
      ~DustOpticalConstants.set_optical_constants
      ~DustOpticalConstants.set_optical_constants_from_draine
      ~DustOpticalConstants.set_optical_constants_from_henn
      ~DustOpticalConstants.set_optical_constants_from_jena
      ~DustOpticalConstants.set_optical_constants_from_oss

   .. rubric:: Methods Documentation

   .. automethod:: add_coat
   .. automethod:: calculate_dhs_opacity
   .. automethod:: calculate_opacity
   .. automethod:: calculate_optical_constants_on_wavelength_grid
   .. automethod:: calculate_size_distribution_opacity
   .. automethod:: set_density
   .. automethod:: set_optical_constants
   .. automethod:: set_optical_constants_from_draine
   .. automethod:: set_optical_constants_from_henn
   .. automethod:: set_optical_constants_from_jena
   .. automethod:: set_optical_constants_from_oss

.. autofunction:: mix_dust