# Copyright (C) 2017 Thomas H. Gibson
#
# This file is part of FIAT.
#
# FIAT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FIAT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FIAT. If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import, print_function, division

import numpy as np
from FIAT.reference_element import Point
from FIAT.functional import PointEvaluation
from FIAT import FiniteElement
from FIAT import dual_set


class PointElement(FiniteElement):
    """Class implementing a 0-dimensional point element.
    It is what it is.
    """

    def __init__(self):
        ref_el = Point()
        entity_dofs = {0: {0: [0]}}
        node = [PointEvaluation(ref_el, pt) for pt in ref_el.get_vertices()]
        dual = dual_set.DualSet(node, ref_el, entity_dofs)

        super(PointElement, self).__init__(ref_el=ref_el,
                                           dual=dual,
                                           order=0,
                                           formdegree=0,
                                           mapping="affine")

    def degree(self):
        """Return the degree of the (embedding) polynomial space."""
        return 0

    def get_nodal_basis(self):
        """Return the nodal basis, encoded as a PolynomialSet object,
        for the finite element."""
        raise NotImplementedError("Not implemented for the point element.")

    def get_coeffs(self):
        """Return the expansion coefficients for the basis of the
        finite element."""
        raise NotImplementedError("Not implemented for the point element.")

    def tabulate(self, order, points, entity=None):
        """Return tabulated values of derivatives up to a given order of
        basis functions at given points.

        :arg order: The maximum order of derivative.
        :arg points: An iterable of points.
        :arg entity: Optional (dimension, entity number) pair
                     indicating which topological entity of the
                     reference element to tabulate on.  If ``None``,
                     tabulated values are computed by geometrically
                     approximating which facet the points are on.
        """
        assert order == 0, (
            "Derivatives are not well-defined on a 0-dimensional domain"
        )
        return {(): np.ones((1, len(points)))}

    def value_shape(self):
        """Return the value shape of the finite element functions."""
        return ()

    def dmats(self):
        """Return dmats: expansion coefficients for basis function
        derivatives."""
        raise NotImplementedError("Not implemented for the point element.")

    def get_num_members(self, arg):
        """Return number of members of the expansion set."""
        raise NotImplementedError("Not implemented for the point element.")
