"""
ProteinArea.py
A package to calculate time-series protein area profile of MD data.

Use 2D voronoi cells to create realistic protein area with protein embedded
in lipids. The code assumes the membrane in MD data is oriented in X-Y plane.

"""

import MDAnalysis as mda
import numpy as np
import shapely.geometry as geo
import matplotlib.pyplot as plt
import argparse
import logging
import time


from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.results import ResultsGroup
from scipy.spatial import Voronoi
from scipy.spatial import cKDTree
from scipy.spatial import voronoi_plot_2d
from matplotlib.patches import Polygon
from numpy.linalg import norm


def calc_area_per_slice(ag, nopbc=False):
    """
    nopbc: flag to turn off pbc images, set to False.
    """

    # create coordinates
    if nopbc:
        points_p, points_np = voronoi_pbc(ag, nopbc)
        points = np.concatenate((points_p, points_np))
    else:
        points_p, points_np, points_pbc = voronoi_pbc(ag)
        points = np.concatenate((points_p, points_np, points_pbc))

    # if no protein atom is in this slice, return 0
    p_size = len(points_p)

    if p_size == 0:
        return 0.0

    # do voronoi
    vor = Voronoi(points)

    # calculate area
    area_per_slice = 0
    for region_index in vor.point_region[:p_size]:
        region_vertices = vor.vertices[vor.regions[region_index]]
        area_per_cell = geo.Polygon(region_vertices).area
        area_per_slice += area_per_cell

    return area_per_slice


def voronoi_pbc(ag, nopbc=False):
    """
    Create pbc images for ag slice.
    For current setup, there are in total 8 images as we are doing voronoi
    in 2D

    nopbc: flag to turn off pbc images, set to False.
    """
    x, y, z = ag.dimensions[0:3]

    # create protein and non-protein coordinates
    points_p = ag.select_atoms("protein").positions[:, 0:2]
    points_np = ag.select_atoms("not protein").positions[:, 0:2]

    if nopbc:
        return points_p, points_np

    # create 8 pbc images
    pbc_arrays = np.array(
        [[x, 0], [x, y], [x, -y], [-x, 0], [-x, y], [-x, -y], [0, y], [0, -y]]
    )

    # create pbc coordinates
    points_pbc = ag.copy().atoms.positions[:, 0:2] + pbc_arrays[0]

    for pbc_array in pbc_arrays[1:]:
        pos_pbc = ag.copy().atoms.positions[:, 0:2] + pbc_array
        points_pbc = np.concatenate((points_pbc, pos_pbc))

    return points_p, points_np, points_pbc


def voronoi_plot(points_p, points_np, points_pbc, name=""):
    # points_p = ag.select_atoms('protein').positions[:, 0:2]
    # points_np = ag.select_atoms('not protein').positions[:, 0:2]

    p_size = len(points_p)
    points = np.concatenate((points_p, points_np, points_pbc))

    # voronoi
    vor = Voronoi(points)
    area_per_slice = 0

    fig, ax = plt.subplots()
    voronoi_plot_2d(
        vor, ax=ax, show_points=False, show_vertices=False, point_size=5
    )

    # for region_index in vor.point_region[:p_size]:
    #     region_vertices = vor.vertices[vor.regions[region_index]]
    #     region_polygon = Polygon(region_vertices, facecolor='red', alpha=0.25)

    #     center_point = points[region_index]
    #     # print(region_vertices)

    #     ax.add_patch(region_polygon)

    plt.scatter(points_p[:, 0], points_p[:, 1], c="b", s=5, label="protein")
    plt.scatter(
        points_np[:, 0],
        points_np[:, 1],
        c="r",
        s=5,
        alpha=0.5,
        label="non-protein",
    )
    plt.scatter(
        points_pbc[:, 0],
        points_pbc[:, 1],
        c="grey",
        s=5,
        alpha=0.5,
        label="pbc",
    )
    plt.xlim([-160, 300])
    plt.ylim([-160, 300])
    plt.savefig("./voronoi/area" + name + ".pdf")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.close()


class ProteinArea(AnalysisBase):
    """
    zmin:  estimated zmin, should be lower than
        the minimal z of the whole traj. Default=0
    zmax:  estimated zmax, should be higher
        than the maximal z of the whole traj. Default=150
    layer: thickness of a layer. default=0.5
    nopbc: flag to turn off pbc images, set to False.
    showpoints: output the voronoi points at certain layer(default=-1)
    """

    @classmethod
    def get_supported_backends(cls):
        return ("serial", "multiprocessing", "dask")

    # Declares which parallelization backends that are supported.

    _analysis_algorithm_is_parallelizable = True
    # This tells MD that this analysis can be parallelized

    def __init__(
        self,
        atomgroup,
        zmin=0,
        zmax=150,
        layer=0.5,
        showpoints=False,
        nopbc=False,
        **kwargs,
    ):
        # This calls parent constructor with the traj
        # Allows use of MD frame-parallel infrastructure

        super().__init__(atomgroup.universe.trajectory, **kwargs)
        self._ag = atomgroup
        self._zmin = zmin
        self._zmax = zmax
        self._layer = layer
        self._showpoints = showpoints
        self._nopbc = nopbc
        # Stores input param for slicing in the Z-dim, optional Voronoi visualization
        # and whether to apply PBC corrections

    def _prepare(self):
        self.results.area_per_frame = []
        self._slices = np.arange(self._zmin, self._zmax, self._layer)
        # Called once before frame analysis starts
        # Initializes the result container
        # Calculates the Z-slice positions from zmin/zmax using layer thickness

    def _single_frame(self):
        for slice_index, slice in enumerate(self._slices[:-1]):
            slice = self._ag.select_atoms(
                "prop z > "
                + str(self._slices[slice_index])
                + " and prop z < "
                + str(self._slices[slice_index + 1])
            )  # Loops over each z-slice of the protein and selects atoms within each slice using z-coord
            # if self._showpoints:
            # points_p, points_np, points_pbc = voronoi_pbc(slice)
            # voronoi_plot(points_p, points_np, points_pbc, name=str(slice_index))
            # Visualizes Voronoi Tess of each slice if --showpoints is enabled
            self.results.area_per_frame.append(
                calc_area_per_slice(slice, self._nopbc)
            )
            # Calculates the Voronoi-based area of the slice and appends it to the result list
            # self._nopbc determines whether PBC corrections are applied

    def _get_aggregator(self):
        return ResultsGroup(
            lookup={"area_per_frame": ResultsGroup.flatten_sequence}
        )
        # Parallel analysis feature that tells MD how to combine per frame results
        # across parallel workers.

    def _conclude(self):
        # Converts the list of area values into a NumPy array for plt.
        self.results.area_per_frame = np.array(self.results.area_per_frame)
        self.results.slice_edges = self._slices
        self.results.slice_centers = 0.5 * (
            self._slices[:-1] + self._slices[1:]
        )

    def run(self, *args, **kwargs):
        if kwargs.get("backend", "serial") != "serial":
            kwargs.pop("progressbar", None)
            self._progress = None
        return super().run(*args, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute protein cross-sectional area per Z-slice from CG trajectory"
    )
    parser.add_argument("tpr", help="GROMACS topology file (.tpr)")
    parser.add_argument("xtc", help="GROMACS trajectory file (.xtc)")
    parser.add_argument(
        "--output",
        type=str,
        default="area_per_frame.npy",
        help="output file name",
    )
    parser.add_argument("--zmin", type=float, default=0, help="lower z bound")
    parser.add_argument("--zmax", type=float, default=150, help="upper z bound")
    parser.add_argument(
        "--layer", type=float, default=0.5, help="slice thickness"
    )
    parser.add_argument(
        "--nopbc", action="store_true", help="disable PBC images"
    )
    parser.add_argument(
        "--start", type=int, default=None, help="first frame to process"
    )    
    parser.add_argument(
        "--stop", type=int, default=None, help="last frame to process"
    )
    parser.add_argument(
        "--backend",
        choices=("serial", "multiprocessing", "dask"),
        default="serial",
        help="parallel backend",
    )
    parser.add_argument("--workers", type=int, help="number of workers")
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument(
        "--showpoints", action="store_true", help="plot Voronoi for each slice"
    )

    args = parser.parse_args()

    logging.basicConfig(
        filename="protein_area.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("ProteinArea run started! :)")
    logging.info(f"Command line arguments: {args}")

    overall_start = time.time()

    logging.info("Loading MDAnalysis Universe")
    u = mda.Universe(args.tpr, args.xtc)
    logging.info("MDAnalysis Universe loaded successfully")

    ag = u.select_atoms("all")
    pa = ProteinArea(
        ag, zmin=args.zmin, zmax=args.zmax, layer=args.layer, nopbc=args.nopbc
    )

    run_start = time.time()
    logging.info(
        f"pa.run() starting (backend={args.backend}, workers={args.workers})"
    )

    pa.run(
        start=args.start,
        stop=args.stop,
        backend=args.backend,
        n_workers=args.workers,
        verbose=args.verbose,
    )

    run_end = time.time()
    run_total = run_end - run_start
    logging.info(f"pa.run() completed in {run_total} seconds")
    logging.info(f"Data saved to {args.output}")

    overall_end = time.time()
    overall_total = overall_end - overall_start
    logging.info(f"Total script duration: {overall_total} seconds")
    logging.info("ProteinArea run finished! :)) \n")

    np.save(args.output, pa.results.area_per_frame)
    print(f"Saved cross-sectional area time series to {args.output}")
