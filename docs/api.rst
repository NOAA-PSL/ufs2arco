API Reference
#############

Regridding
----------

.. autosummary::
   :toctree: generated/

   ufs2arco.Layers2Pressure
   ufs2arco.transforms.layers_to_pressure


Ocean Diagnostics
-----------------

.. autosummary::
   :toctree: generated/

   ufs2arco.ocean_diagnostics.wright_density
   ufs2arco.ocean_diagnostics.hydrostatic_pressure
   ufs2arco.ocean_diagnostics.interfaces_from_centers
   ufs2arco.ocean_diagnostics.mixed_layer_depth_by_density_difference
   ufs2arco.ocean_diagnostics.ocean_heat_content
   ufs2arco.transforms.ocean_density
   ufs2arco.transforms.mixed_layer_depth
   ufs2arco.transforms.ocean_heat_content


Data Sources
------------

.. autosummary::
   :toctree: generated/

   ufs2arco.sources.Source
   ufs2arco.sources.AWSGEFSArchive
   ufs2arco.sources.AWSHRRRArchive
   ufs2arco.sources.GFSArchive

Data Targets
------------

.. autosummary::
   :toctree: generated/

   ufs2arco.targets.Target
   ufs2arco.targets.Anemoi


Utilities
---------

.. autosummary::
   :toctree: generated/

   ufs2arco.utils.expand_anemoi_dataset
   ufs2arco.utils.convert_anemoi_inference_dataset
   ufs2arco.static_masks.fit_static_mask_artifact
   ufs2arco.static_masks.load_static_mask_artifact
