Getting Started
===============

Protein_Area estimates a membrane protein's cross-sectional area as a function
of its position along the z axis. For each trajectory frame, it divides the
system into z slices and uses a two-dimensional Voronoi tessellation in the
x-y plane. The areas of cells belonging to protein atoms are summed for each
slice.

The membrane is expected to lie in the x-y plane. By default, periodic images
are included in the tessellation; this can be disabled with ``nopbc=True`` in
Python or ``--nopbc`` on the command line.

Installation
------------

From a clone of the repository, install the package and its runtime
dependencies with:

.. code-block:: console

   pip install .

The package requires Python 3.11 or newer.

Python API
----------

Create an MDAnalysis ``Universe`` from a topology and trajectory, then pass an
atom group containing the protein and its surrounding atoms to
``ProteinArea``. The analysis identifies protein atoms using the MDAnalysis
``protein`` selection, so the topology must contain the attributes needed for
that selection.

.. code-block:: python

   import MDAnalysis as mda
   from Protein_Area import ProteinArea

   universe = mda.Universe("topology.tpr", "trajectory.xtc")
   atoms = universe.select_atoms("all")

   analysis = ProteinArea(
       atoms,
       zmin=0,
       zmax=150,
       layer=0.5,
   )
   analysis.run(start=0, stop=100, step=1)

``zmin`` and ``zmax`` set the range of z coordinates to analyze, and
``layer`` sets the spacing between slice edges. These values use the
trajectory's coordinate units. For MDAnalysis' usual coordinates in
angstroms, the calculated areas are in square angstroms.

Results are available on ``analysis.results``:

* ``area_per_frame`` is a two-dimensional array with shape
  ``(number of analyzed frames, number of z slices)``.
* ``slice_edges`` contains the z-coordinate boundaries of the slices.
* ``slice_centers`` contains the midpoint of each slice.
* ``analysis.times`` and ``analysis.frames`` give the time and frame index for
  each analyzed frame.

You can select a different range or stride of frames with the standard
``start``, ``stop``, and ``step`` arguments to ``analysis.run()``. The
``ProteinArea`` analysis also supports the ``serial``, ``multiprocessing``,
and ``dask`` backends supported by MDAnalysis.

Command line
------------

The command-line interface reads GROMACS ``.tpr`` topology and ``.xtc``
trajectory files. For example, from an environment where Protein_Area is
installed, run:

.. code-block:: console

   python -m Protein_Area.ProteinArea topology.tpr trajectory.xtc \
       --zmin 0 --zmax 150 --layer 0.5

The command analyzes all atoms and writes NumPy ``.npy`` files to the current
directory. The default output names are:

* ``area_per_frame.npy``: area values, shaped as frames by z slices.
* ``slice_edges.npy`` and ``slice_centers.npy``: z positions for the slices.
* ``times.npy`` and ``frames.npy``: time and frame index for each result
  row.

Use ``--help`` to list the available options. In addition to the z bounds,
slice thickness, and output filenames, the command supports ``--start`` and
``--stop`` to choose a frame range, ``--backend`` and ``--workers`` for
parallel analysis, ``--nopbc`` to disable periodic images, and ``--verbose``
for verbose progress output.

Notes on interpretation
-----------------------

The method assumes the membrane is oriented in the x-y plane; it does not
reorient the trajectory. Choose z bounds that cover the protein throughout
the frames being analyzed. The area is calculated from the protein atoms
present in each slice, so the result depends on the system's atom
representation and the selected slice thickness.
