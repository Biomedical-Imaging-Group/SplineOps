import numpy as np
import numpy.typing as npt
from typing import Sequence, Union, Tuple, cast

from bssp.bases.splinebasis import SplineBasis
from bssp.bases.utils import asbasis
from bssp.modes.extensionmode import ExtensionMode
from bssp.modes.utils import asmode
from bssp.utils.interop import is_ndarray

TSplineBasis = Union[SplineBasis, str]
TSplineBases = Union[TSplineBasis, Sequence[TSplineBasis]]
TExtensionMode = Union[ExtensionMode, str]
TExtensionModes = Union[TExtensionMode, Sequence[TExtensionMode]]


class TensorSpline:
    """
    Represents a tensor spline model for multidimensional data interpolation
    using spline bases and extension modes.

    Parameters (``__init__()``)
    ---------------------------
    * **data** (:py:obj:`numpy.typing.NDArray`)
      --
      The data array to fit the tensor spline model.
    * **coordinates** (:py:obj:`Union[numpy.typing.NDArray, Sequence[numpy.typing.NDArray]]`)
      --
      The coordinates corresponding to each dimension of the data.
    * **bases** (:py:class:`~TSplineBases`, :py:obj:`None`)
      --
      Spline bases for each dimension.
    * **modes** (:py:class:`~TExtensionModes`, :py:obj:`None`)
      --
      Extension modes for each dimension.

    Examples
    --------
    Below is an example of using `TensorSpline` to interpolate multidimensional data.
    This example demonstrates setting up the tensor spline, creating evaluation coordinates,
    and generating plots for both the original and interpolated data.

    .. code-block:: python3

        import numpy as np
        import matplotlib.pyplot as plt
        from your_package.interpolate.tensorspline import TensorSpline  # Adjust import as necessary

        # Data type (need to provide floating numbers, "float64" and "float32" are typical)
        dtype = "float32"

        # Create random data samples and corresponding coordinates
        nx, ny = 2, 5
        xmin, xmax = -3.1, +1
        ymin, ymax = 2, 6.5
        xx = np.linspace(xmin, xmax, nx, dtype=dtype)
        yy = np.linspace(ymin, ymax, ny, dtype=dtype)
        coordinates = xx, yy
        prng = np.random.default_rng(seed=5250)
        data = prng.standard_normal(size=tuple(c.size for c in coordinates))
        data = np.ascontiguousarray(data, dtype=dtype)

        # Tensor spline bases
        #  Note: Need to provide one basis per data dimension. If a single one is provided,
        #  it will be applied to all dimensions
        bases = "bspline3"  # same basis applied to all dimensions

        # Tensor spline signal extension modes (sometimes referred to as boundary condition)
        #  Note: Similar strategy as for bases.
        modes = "mirror"  # same mode applied to all dimensions

        # Create tensor spline
        tensor_spline = TensorSpline(
            data=data, coordinates=coordinates, bases=bases, modes=modes
        )

        # Create evaluation coordinates (extended and oversampled in this case)
        dx = (xx[-1] - xx[0]) / (nx - 1)
        dy = (yy[-1] - yy[0]) / (ny - 1)
        pad_fct = 1.1
        px = pad_fct * nx * dx
        py = pad_fct * ny * dy
        eval_xx = np.linspace(xx[0] - px, xx[-1] + px, 100 * nx)
        eval_yy = np.linspace(yy[0] - py, yy[-1] + py, 100 * ny)

        # Standard evaluation
        #   Note: coordinates are passed as a "grid", a sequence of regularly spaced axes.
        eval_coords = eval_xx, eval_yy
        data_eval = tensor_spline(coordinates=eval_coords)

        # Meshgrid evaluation (not the default choice but could be useful in some cases)
        eval_coords_mg = np.meshgrid(*eval_coords, indexing="ij")
        data_eval_mg = tensor_spline(coordinates=eval_coords_mg, grid=False)
        #   We can test that both evaluation strategy gives the same values
        np.testing.assert_equal(data_eval, data_eval_mg)

        # We can also pass a list of points directly (i.e., not as a grid)
        #   Note: here we just reshape the meshgrid as a list of evaluation coordinates
        eval_coords_pts = np.reshape(eval_coords_mg, newshape=(2, -1))
        data_eval_pts = tensor_spline(coordinates=eval_coords_pts, grid=False)
        #   We can test that it again results in the same evaluation (after reshaping)
        np.testing.assert_equal(data_eval, np.reshape(data_eval_pts, data_eval_mg.shape))

        # Figure
        fig: plt.Figure
        ax: plt.Axes

        extent = [xx[0] - dx / 2, xx[-1] + dx / 2, yy[0] - dy / 2, yy[-1] + dy / 2]
        eval_extent = [
            eval_xx[0] - dx / 2,
            eval_xx[-1] + dx / 2,
            eval_yy[0] - dy / 2,
            eval_yy[-1] + dy / 2,
        ]

        fig, axes = plt.subplots(
            nrows=1, ncols=2, sharex="all", sharey="all", layout="constrained"
        )
        ax = axes[0]
        ax.imshow(data.T, extent=extent)
        ax.set_title("Original data samples")
        ax = axes[1]
        ax.imshow(data_eval.T, extent=eval_extent)
        ax.set_title("Interpolated data")

        plt.show()

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from your_package.interpolate.tensorspline import TensorSpline  # Adjust import as necessary

        # Data type (need to provide floating numbers, "float64" and "float32" are typical)
        dtype = "float32"

        # Create random data samples and corresponding coordinates
        nx, ny = 2, 5
        xmin, xmax = -3.1, +1
        ymin, ymax = 2, 6.5
        xx = np.linspace(xmin, xmax, nx, dtype=dtype)
        yy = np.linspace(ymin, ymax, ny, dtype=dtype)
        coordinates = xx, yy
        prng = np.random.default_rng(seed=5250)
        data = prng.standard_normal(size=tuple(c.size for c in coordinates))
        data = np.ascontiguousarray(data, dtype=dtype)

        # Tensor spline bases
        #  Note: Need to provide one basis per data dimension. If a single one is provided,
        #  it will be applied to all dimensions
        bases = "bspline3"  # same basis applied to all dimensions

        # Tensor spline signal extension modes (sometimes referred to as boundary condition)
        #  Note: Similar strategy as for bases.
        modes = "mirror"  # same mode applied to all dimensions

        # Create tensor spline
        tensor_spline = TensorSpline(
            data=data, coordinates=coordinates, bases=bases, modes=modes
        )

        # Create evaluation coordinates (extended and oversampled in this case)
        dx = (xx[-1] - xx[0]) / (nx - 1)
        dy = (yy[-1] - yy[0]) / (ny - 1)
        pad_fct = 1.1
        px = pad_fct * nx * dx
        py = pad_fct * ny * dy
        eval_xx = np.linspace(xx[0] - px, xx[-1] + px, 100 * nx)
        eval_yy = np.linspace(yy[0] - py, yy[-1] + py, 100 * ny)

        # Standard evaluation
        #   Note: coordinates are passed as a "grid", a sequence of regularly spaced axes.
        eval_coords = eval_xx, eval_yy
        data_eval = tensor_spline(coordinates=eval_coords)

        # Meshgrid evaluation (not the default choice but could be useful in some cases)
        eval_coords_mg = np.meshgrid(*eval_coords, indexing="ij")
        data_eval_mg = tensor_spline(coordinates=eval_coords_mg, grid=False)
        #   We can test that both evaluation strategy gives the same values
        np.testing.assert_equal(data_eval, data_eval_mg)

        # We can also pass a list of points directly (i.e., not as a grid)
        #   Note: here we just reshape the meshgrid as a list of evaluation coordinates
        eval_coords_pts = np.reshape(eval_coords_mg, newshape=(2, -1))
        data_eval_pts = tensor_spline(coordinates=eval_coords_pts, grid=False)
        #   We can test that it again results in the same evaluation (after reshaping)
        np.testing.assert_equal(data_eval, np.reshape(data_eval_pts, data_eval_mg.shape))

        # Figure
        fig: plt.Figure
        ax: plt.Axes

        extent = [xx[0] - dx / 2, xx[-1] + dx / 2, yy[0] - dy / 2, yy[-1] + dy / 2]
        eval_extent = [
            eval_xx[0] - dx / 2,
            eval_xx[-1] + dx / 2,
            eval_yy[0] - dy / 2,
            eval_yy[-1] + dy / 2,
        ]

        fig, axes = plt.subplots(
            nrows=1, ncols=2, sharex="all", sharey="all", layout="constrained"
        )
        ax = axes[0]
        ax.imshow(data.T, extent=extent)
        ax.set_title("Original data samples")
        ax = axes[1]
        ax.imshow(data_eval.T, extent=eval_extent)
        ax.set_title("Interpolated data")

        plt.show()
    """

    def __init__(
        self,
        data: npt.NDArray,
        # TODO(dperdios): naming convention, used `samples` instead of `data`?
        coordinates: Union[npt.NDArray, Sequence[npt.NDArray]],
        bases: TSplineBases,
        modes: TExtensionModes,
        # TODO(dperdios): extrapolate? only at evaluation time?
        # TODO(dperdios): axis? axes? probably complex
        # TODO(dperdios): optional reduction strategy (e.g., first or last)
    ) -> None:
        # Data
        if not is_ndarray(data):
            raise TypeError("Must be an array.")
        ndim = data.ndim
        self._ndim = ndim

        # TODO(dperdios): make `coordinates` optional?
        # TODO(dperdios): `coordinates` need to define a uniform grid.
        #  Note: this is not straightforward to control (numerical errors)
        # Coordinates
        #   1-D special case (either `array` or `(array,)`)
        if is_ndarray(coordinates) and ndim == 1 and len(coordinates) == len(data):
            # Note: we explicitly cast the type to NDArray
            coordinates = cast(npt.NDArray, coordinates)
            # Convert `array` to `(array,)`
            coordinates = (coordinates,)
        if not all(bool(np.all(np.diff(c) > 0)) for c in coordinates):
            raise ValueError("Coordinates must be strictly ascending.")
        valid_data_shape = tuple([c.size for c in coordinates])
        if data.shape != valid_data_shape:
            raise ValueError(
                f"Incompatible data shape. " f"Expected shape: {valid_data_shape}"
            )
        if not all(np.isrealobj(c) for c in coordinates):
            raise ValueError("Must be sequence of real numbers.")
        # TODO(dperdios): useful to keep initial coordinates as property?

        # Pre-computation based on coordinates
        # TODO(dperdios): convert to Python float?
        bounds = tuple([(c[0], c[-1]) for c in coordinates])
        # TODO(dperdios): `bounds` as a public property?
        self._bounds = bounds
        lengths = valid_data_shape
        self._lengths = lengths
        step_seq = []
        for b, l in zip(bounds, lengths):
            if l > 1:
                step = (b[-1] - b[0]) / (l - 1)
            else:
                # Special case for single-sample signal
                step = 1
            step_seq.append(step)
        steps = tuple(step_seq)  # TODO: convert dtype? (can be promoted)
        self._steps = steps
        # TODO(dperdios): cast scalars to real_dtype?

        # DTypes
        dtype = data.dtype
        if not (
            np.issubdtype(dtype, np.floating)
            or np.issubdtype(dtype, np.complexfloating)
        ):
            raise ValueError("Data must be an array of floating point numbers.")
        real_dtype = data.real.dtype
        coords_dtype_seq = tuple(c.dtype for c in coordinates)
        if len(set(coords_dtype_seq)) != 1:
            raise ValueError(
                "Incompatible dtypes in sequence of coordinates. "
                "Expected a consistent dtype. "
                f"Received different dtypes: {tuple(d.name for d in coords_dtype_seq)}"
            )
        coords_dtype = coords_dtype_seq[0]
        if coords_dtype.itemsize != real_dtype.itemsize:
            # TODO(dperdios): maybe automatic cast in the future?
            raise ValueError("Coordinates and data have different floating precisions.")
        self._dtype = dtype
        self._real_dtype = real_dtype

        # Bases
        if isinstance(bases, (SplineBasis, str)):
            # Explicit type cast (special case)
            bases = cast(str, bases)
            bases = ndim * (bases,)
        bases = tuple(asbasis(b) for b in bases)
        if len(bases) != ndim:
            raise ValueError(f"Length of the sequence must be {ndim}.")
        self._bases = bases

        # Modes
        if isinstance(modes, (ExtensionMode, str)):
            # Explicit type cast (special case)
            modes = cast(str, modes)
            modes = ndim * (modes,)
        modes = tuple(asmode(m) for m in modes)
        if len(modes) != ndim:
            raise ValueError(f"Length of the sequence must be {ndim}.")
        self._modes = modes

        # Compute coefficients
        coefficients = self._compute_coefficients(data=data)
        self._coefficients = coefficients

    # Properties
    @property
    def coefficients(self) -> npt.NDArray:
        """
        Provides the computed coefficients of the tensor spline model.

        Returns
        -------
        npt.NDArray
            A copy of the coefficients array, ensuring that modifications do not affect the internal model.
        """
        return np.copy(self._coefficients)

    @property
    def bases(self) -> Tuple[SplineBasis, ...]:
        """
        The spline bases used for each dimension of the tensor spline model.

        Returns
        -------
        Tuple[SplineBasis, ...]
            The spline bases.
        """
        return self._bases

    @property
    def modes(self) -> Tuple[ExtensionMode, ...]:
        """
        The extension modes used for each dimension of the tensor spline model.

        Returns
        -------
        Tuple[ExtensionMode, ...]
            The extension modes.
        """
        return self._modes

    @property
    def ndim(self):
        """
        The number of dimensions of the tensor spline model, determined by the input data.

        Returns
        -------
        int
            The number of dimensions.
        """
        return self._ndim

    # Methods
    def __call__(
        self,
        coordinates: Union[npt.NDArray, Sequence[npt.NDArray]],
        grid: bool = True,
        # TODO(dperdios): extrapolate?
    ) -> npt.NDArray:
        """
        Evaluates the tensor spline model at specified coordinates.

        Parameters
        ----------
        coordinates : Union[npt.NDArray, Sequence[npt.NDArray]]
            The coordinates where the tensor spline should be evaluated.
        grid : bool, optional
            Specifies if the coordinates are in grid form (default is True).

        Returns
        -------
        npt.NDArray
            The interpolated values at the specified coordinates.
        """
        return self.eval(coordinates=coordinates, grid=grid)

    def eval(
        self, coordinates: Union[npt.NDArray, Sequence[npt.NDArray]], grid: bool = True
    ) -> npt.NDArray:
        """
        Evaluates the tensor spline at the given coordinates.

        Parameters
        ----------
        coordinates : Union[npt.NDArray, Sequence[npt.NDArray]]
            Coordinates at which to evaluate the tensor spline.
        grid : bool
            Specifies if the coordinates are structured as a grid.

        Returns
        -------
        npt.NDArray
            The interpolated values at the specified coordinates.
        """

        # Check coordinates
        ndim = self._ndim
        if grid:
            # Special 1-D case: "default" grid=True with a 1-D `coords` NDArray
            if is_ndarray(coordinates):
                # Note: we explicitly cast the type to NDArray
                coordinates = cast(npt.NDArray, coordinates)
                if ndim == 1 and coordinates.ndim == 1:
                    coordinates = (coordinates,)
            # N-D cases
            if len(coordinates) != ndim:
                # TODO(dperdios): Sequence of (..., n) arrays (batch dimensions
                #   must be the same!)
                raise ValueError(f"Must be a {ndim}-length sequence of 1-D arrays.")
            if not all([bool(np.all(np.diff(c, axis=-1) > 0)) for c in coordinates]):
                # TODO(dperdios): do they really need to be ascending?
                raise ValueError("Coordinates must be strictly ascending.")
        else:
            # If not `grid`, a sequence of arrays is expected with a length
            #  equal to the number of dimensions. Each array in the sequence
            #  must be of the same shape.
            coords_shapes = [c.shape for c in coordinates]
            if len(coordinates) != ndim or len(set(coords_shapes)) != 1:
                raise ValueError(
                    f"Incompatible sequence of coordinates. "
                    f"Must be a {ndim}-length sequence of same-shape N-D arrays. "
                    f"Current sequence of array shapes: {coords_shapes}."
                )
        if not all(np.isrealobj(c) for c in coordinates):
            raise ValueError("Must be a sequence of real numbers.")

        # Get properties
        real_dtype = self._real_dtype
        bounds_seq = self._bounds
        length_seq = self._lengths
        step_seq = self._steps
        basis_seq = self._bases
        mode_seq = self._modes
        coefficients = self._coefficients
        ndim = self._ndim

        # Rename
        coords_seq = coordinates

        # For-loop over dimensions
        indexes_seq = []
        weights_seq = []
        for coords, basis, mode, data_lim, dx, data_len in zip(
            coords_seq, basis_seq, mode_seq, bounds_seq, step_seq, length_seq
        ):

            # Data limits
            x_min, x_max = data_lim

            # Indexes
            #   Compute rational indexes
            # TODO(dperdios): no difference in using `* fs` or `/ dx`
            # fs = 1 / dx
            # rat_indexes = (coords - x_min) * fs
            rat_indexes = (coords - x_min) / dx
            #   Compute corresponding integer indexes (including support)
            indexes = basis.compute_support_indexes(x=rat_indexes)
            # TODO(dperdios): specify dtype in compute_support_indexes? cast dtype here?
            #  int32 faster than int64? probably not

            # Evaluate basis function (interpolation weights)
            # indexes_shift = np.subtract(indexes, rat_indexes, dtype=real_dtype)
            # shifted_idx = np.subtract(indexes, rat_indexes, dtype=real_dtype)
            shifted_idx = np.subtract(
                rat_indexes[np.newaxis], indexes, dtype=real_dtype
            )
            # TODO(dperdios): casting rules, do we really want it?
            weights = basis(x=shifted_idx)

            # Signal extension
            indexes_ext, weights_ext = mode.extend_signal(
                indexes=indexes, weights=weights, length=data_len
            )

            # TODO(dperdios): Add extrapolate handling?
            # weights[idx_extra] = cval ?? or within extend_signal?

            # Store
            indexes_seq.append(indexes_ext)
            weights_seq.append(weights_ext)

        # Broadcast arrays for tensor product
        if grid:
            # del_axis_base = np.arange(ndim + 1, step=ndim)
            # del_axes = [del_axis_base + ii for ii in range(ndim)]
            # exp_axis_base = np.arange(2 * ndim)
            # exp_axes = [tuple(np.delete(exp_axis_base, a)) for a in del_axes]
            # Batch-compatible axis expansions
            exp_axis_ind_base = np.arange(ndim)  # from start
            exp_axis_coeffs_base = exp_axis_ind_base - ndim  # from end TODO: reverse?
            exp_axes = []
            for ii in range(ndim):
                a = np.concatenate(
                    [
                        np.delete(exp_axis_ind_base, ii),
                        np.delete(exp_axis_coeffs_base, ii),
                    ]
                )
                exp_axes.append(tuple(a))
        else:
            exp_axis_base = np.arange(ndim)
            exp_axes = [tuple(np.delete(exp_axis_base, a)) for a in range(ndim)]

        indexes_bc = []
        weights_bc = []
        for indexes, weights, a in zip(indexes_seq, weights_seq, exp_axes):
            indexes_bc.append(np.expand_dims(indexes, axis=a))
            weights_bc.append(np.expand_dims(weights, axis=a))
        # Note: for interop (CuPy), cannot use prod with a sequence of arrays.
        #  Need explicit stacking before reduction. It is NumPy compatible.
        weights_tp = np.prod(np.stack(np.broadcast_arrays(*weights_bc), axis=0), axis=0)

        # Interpolation (convolution via reduction)
        # TODO(dperdios): might want to change the default reduction axis
        axes_sum = tuple(range(ndim))  # first axes are the indexes
        data = np.sum(coefficients[tuple(indexes_bc)] * weights_tp, axis=axes_sum)

        return data

    def _compute_coefficients(self, data: npt.NDArray) -> npt.NDArray:

        # Prepare data and axes
        # TODO(dperdios): there is probably too many copies along this process
        coefficients = np.copy(data)
        axes = tuple(range(coefficients.ndim))
        axes_roll = tuple(np.roll(axes, shift=-1))

        # TODO(dperdios): could do one less roll by starting with the initial
        #  shape
        for basis, mode in zip(self._bases, self._modes):

            # Roll data w.r.t. dimension
            coefficients = np.transpose(coefficients, axes=axes_roll)

            # Compute coefficients w.r.t. extension `mode` and `basis`
            coefficients = mode.compute_coefficients(data=coefficients, basis=basis)

        return coefficients
