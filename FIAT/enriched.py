# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
# Copyright (C) 2013 Andrew T. T. McRae
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

import numpy
from .finite_element import FiniteElement
from .tensor_product import TensorProductElement
from . import dual_set
from copy import copy


class EnrichedElement(FiniteElement):
    """Class implementing a finite element that combined the degrees of freedom
    of two existing finite elements."""

    def __init__(self, A, B):

        # Firstly, check it makes sense to enrich.  Elements must have:
        # - same reference element
        # - same mapping
        # - same value shape
        if not A.get_reference_element() == B.get_reference_element():
            raise ValueError("Elements must be defined on the same reference element")
        if not A.mapping()[0] == B.mapping()[0]:
            raise ValueError("Elements must have same mapping")
        if not A.value_shape() == B.value_shape():
            raise ValueError("Elements must have the same value shape")

        # Set up constituent elements
        self.A = A
        self.B = B

        # required degree (for quadrature) is definitely max
        self.polydegree = max(A.degree(), B.degree())
        # order is at least max, possibly more, though getting this
        # right isn't important AFAIK
        self.order = max(A.get_order(), B.get_order())
        # form degree is essentially max (not true for Hdiv/Hcurl,
        # but this will raise an error above anyway).
        # E.g. an H^1 function enriched with an L^2 is now just L^2.
        if A.get_formdegree() is None or B.get_formdegree() is None:
            self.formdegree = None
        else:
            self.formdegree = max(A.get_formdegree(), B.get_formdegree())

        # set up reference element and mapping, following checks above
        self.ref_el = A.get_reference_element()
        self._mapping = A.mapping()[0]

        # set up entity_ids - for each geometric entity, just concatenate
        # the entities of the constituent elements
        Adofs = A.entity_dofs()
        Bdofs = B.entity_dofs()
        offset = A.space_dimension()  # number of entities belonging to A
        entity_ids = {}

        for ent_dim in Adofs:
            entity_ids[ent_dim] = {}
            for ent_dim_index in Adofs[ent_dim]:
                entlist = copy(Adofs[ent_dim][ent_dim_index])
                entlist += [c + offset for c in Bdofs[ent_dim][ent_dim_index]]
                entity_ids[ent_dim][ent_dim_index] = entlist

        # set up dual basis - just concatenation
        nodes = A.dual_basis() + B.dual_basis()
        self.dual = dual_set.DualSet(nodes, self.ref_el, entity_ids)

    def degree(self):
        """Return the degree of the (embedding) polynomial space."""
        return self.polydegree

    def get_nodal_basis(self):
        """Return the nodal basis, encoded as a PolynomialSet object,
        for the finite element."""
        raise NotImplementedError("get_nodal_basis not implemented")

    def get_coeffs(self):
        """Return the expansion coefficients for the basis of the
        finite element."""
        raise NotImplementedError("get_coeffs not implemented")

    def space_dimension(self):
        """Return the dimension of the finite element space."""
        # number of dofs just adds
        return self.A.space_dimension() + self.B.space_dimension()

    def tabulate(self, order, points):
        """Return tabulated values of derivatives up to given order of
        basis functions at given points."""

        # Again, simply concatenate at the basis-function level
        # Number of array dimensions depends on whether the space
        # is scalar- or vector-valued, so treat these separately.

        Asd = self.A.space_dimension()
        Bsd = self.B.space_dimension()
        Atab = self.A.tabulate(order, points)
        Btab = self.B.tabulate(order, points)
        npoints = len(points)
        vs = self.A.value_shape()
        rank = len(vs)  # scalar: 0, vector: 1

        result = {}
        for index in Atab:
            if rank == 0:
                # scalar valued
                # Atab[index] and Btab[index] look like
                # array[basis_fn][point]
                # We build a new array, which will be the concatenation
                # of the two subarrays, in the first index.

                temp = numpy.zeros((Asd + Bsd, npoints),
                                   dtype=Atab[index].dtype)
                temp[:Asd, :] = Atab[index][:, :]
                temp[Asd:, :] = Btab[index][:, :]

                result[index] = temp
            elif rank == 1:
                # vector valued
                # Atab[index] and Btab[index] look like
                # array[basis_fn][x/y/z][point]
                # We build a new array, which will be the concatenation
                # of the two subarrays, in the first index.

                temp = numpy.zeros((Asd + Bsd, vs[0], npoints),
                                   dtype=Atab[index].dtype)
                temp[:Asd, :, :] = Atab[index][:, :, :]
                temp[Asd:, :, :] = Btab[index][:, :, :]

                result[index] = temp
            else:
                raise NotImplementedError("must be scalar- or vector-valued")
        return result

    def value_shape(self):
        """Return the value shape of the finite element functions."""
        return self.A.value_shape()

    def dmats(self):
        """Return dmats: expansion coefficients for basis function
        derivatives."""
        raise NotImplementedError("dmats not implemented")

    def get_num_members(self, arg):
        """Return number of members of the expansion set."""
        raise NotImplementedError("get_num_members not implemented")
