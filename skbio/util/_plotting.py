# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""Plotting utilities."""

import importlib
from io import StringIO, BytesIO


class PlottableMixin:
    """A plottable object."""

    def _get_mpl_plt(self):
        """Import Matplotlib and its plotting interface."""
        msg = "Plotting requires Matplotlib installed in the system."
        if hasattr(self, "mpl"):
            if self.mpl is None:
                raise ImportError(msg)
            return
        try:
            self.mpl = importlib.import_module("matplotlib")
        except ImportError:
            self.mpl = None
            raise ImportError(msg)
        else:
            self.plt = importlib.import_module("matplotlib.pyplot")

    def _figure_data(self, format="png"):
        """Get figure data of a plottable object.

        Parameters
        ----------
        format : str, optional
            Image format supported by the plotting backend. Examples include
            'png' (default), 'svg', 'pdf', and 'eps'.

        Returns
        -------
        str or bytes or None
            Figure data, or None if the plotting backend is not available.

        """
        try:
            self._get_mpl_plt()
        except ImportError:
            return

        # call default plotting method
        fig = self.plot()
        fig.tight_layout()

        # get figure data
        # formats such as SVG are string
        try:
            fig.savefig(f := StringIO(), format=format)
        # formats such as PNG are bytes
        except TypeError:
            fig.savefig(f := BytesIO(), format=format)

        # close figure to avoid double display in IPython
        self.plt.close(fig)

        return f.getvalue()
