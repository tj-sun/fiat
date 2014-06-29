# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
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

"""Principal orthogonal expansion functions as defined by Karniadakis
and Sherwin.  These are parametrized over a reference element so as
to allow users to get coordinates that they want."""

import numpy
import math
import sympy
from . import reference_element
from . import jacobi

# Import AD modules from ScientificPython
try:
    import Scientific.Functions.Derivatives as Derivatives
except:
    raise Exception("""\
Unable to import the Python Scientific module required by FIAT.
Consider installing the package python-scientific.
""")


def xi_triangle(eta):
    """Maps from [-1,1]^2 to the (-1,1) reference triangle."""
    eta1, eta2 = eta
    xi1 = 0.5 * (1.0 + eta1) * (1.0 - eta2) - 1.0
    xi2 = eta2
    return (xi1, xi2)


def xi_tetrahedron(eta):
    """Maps from [-1,1]^3 to the -1/1 reference tetrahedron."""
    eta1, eta2, eta3 = eta
    xi1 = 0.25 * (1. + eta1) * (1. - eta2) * (1. - eta3) - 1.
    xi2 = 0.5 * (1. + eta2) * (1. - eta3) - 1.
    xi3 = eta3
    return xi1, xi2, xi3


class LineExpansionSet:
    """Evaluates the Legendre basis on a line reference element."""
    def __init__(self, ref_el):
        if ref_el.get_spatial_dimension() != 1:
            raise Exception("Must have a line")
        self.ref_el = ref_el
        self.base_ref_el = reference_element.DefaultLine()
        v1 = ref_el.get_vertices()
        v2 = self.base_ref_el.get_vertices()
        self.A, self.b = reference_element.make_affine_mapping(v1, v2)
        self.mapping = lambda x: numpy.dot(self.A, x) + self.b
        self.scale = numpy.sqrt(numpy.linalg.det(self.A))

    def get_num_members(self, n):
        return n+1

    def tabulate(self, n, pts):
        """Returns a numpy array A[i,j] = phi_i(pts[j])"""
        if len(pts) > 0:
            ref_pts = numpy.array([self.mapping(pt) for pt in pts])
            psitilde_as = jacobi.eval_jacobi_batch(0, 0, n, ref_pts)

            results = numpy.zeros((n+1, len(pts)), type(pts[0][0]))
            for k in range(n + 1):
                results[k, :] = psitilde_as[k, :] * math.sqrt(k + 0.5)

            return results
        else:
            return []

    def tabulate_derivatives(self, n, pts):
        """Returns a tuple of length one (A,) such that
        A[i,j] = D phi_i(pts[j]).  The tuple is returned for
        compatibility with the interfaces of the triangle and
        tetrahedron expansions."""
        ref_pts = [self.mapping(pt) for pt in pts]
        psitilde_as_derivs = jacobi.eval_jacobi_deriv_batch(0, 0, n, ref_pts)

        results = numpy.zeros((n+1, len(pts[0])), "d")
        for k in range(0, n + 1):
            results[k, :] = psitilde_as_derivs[k, :] * numpy.sqrt(k + 0.5)

        return (results,)


class TriangleExpansionSet:
    """Evaluates the orthonormal Dubiner basis on a triangular
    reference element."""
    def __init__(self, ref_el):
        if ref_el.get_spatial_dimension() != 2:
            raise Exception("Must have a triangle")
        self.ref_el = ref_el
        self.base_ref_el = reference_element.DefaultTriangle()
        v1 = ref_el.get_vertices()
        v2 = self.base_ref_el.get_vertices()
        self.A, self.b = reference_element.make_affine_mapping(v1, v2)
        self.mapping = lambda x: numpy.dot(self.A, x) + self.b
#        self.scale = numpy.sqrt(numpy.linalg.det(self.A))

    def get_num_members(self, n):
        return (n+1)*(n+2)/2

    def tabulate(self, n, pts):
        if len(pts) == 0:
            return numpy.array([])
        else:
            return numpy.array(self._tabulate(n, numpy.array(pts).T))

    def _tabulate(self, n, pts):
        '''A version of tabulate() that also works for a single point.
        '''
        m1, m2 = self.A.shape
        ref_pts = [sum(self.A[i][j] * pts[j] for j in range(m2)) + self.b[i]
                   for i in range(m1)
                   ]

        def idx(p, q):
            return (p+q)*(p+q+1)/2 + q

        def jrc(a, b, n):
            an = float((2*n+1+a+b)*(2*n+2+a+b)) \
                / float(2*(n+1)*(n+1+a+b))
            bn = float((a*a-b*b) * (2*n+1+a+b)) \
                / float(2*(n+1)*(2*n+a+b)*(n+1+a+b))
            cn = float((n+a)*(n+b)*(2*n+2+a+b)) \
                / float((n+1)*(n+1+a+b)*(2*n+a+b))
            return an, bn, cn

        results = ((n+1)*(n+2)/2) * [None]

        results[0] = 1.0 \
            + pts[0] - pts[0] \
            + pts[1] - pts[1]

        if n == 0:
            return results

        x = ref_pts[0]
        y = ref_pts[1]

        f1 = (1.0+2*x+y)/2.0
        f2 = (1.0 - y) / 2.0
        f3 = f2**2

        results[idx(1, 0)] = f1

        for p in range(1, n):
            a = (2.0*p+1)/(1.0+p)
            # b = p / (p+1.0)
            results[idx(p+1, 0)] = a * f1 * results[idx(p, 0)] \
                - p/(1.0+p) * f3 * results[idx(p-1, 0)]

        for p in range(n):
            results[idx(p, 1)] = 0.5 * (1+2.0*p+(3.0+2.0*p)*y) \
                * results[idx(p, 0)]

        for p in range(n-1):
            for q in range(1, n-p):
                (a1, a2, a3) = jrc(2*p+1, 0, q)
                results[idx(p, q+1)] = \
                    (a1 * y + a2) * results[idx(p, q)] \
                    - a3 * results[idx(p, q-1)]

        for p in range(n+1):
            for q in range(n-p+1):
                results[idx(p, q)] *= math.sqrt((p+0.5)*(p+q+1.0))

        return results
        #return self.scale * results

    def tabulate_derivatives(self, n, pts):
        N = len(pts)
        pts = numpy.array(pts)
        # First get the actual values
        values = self.tabulate(n, pts)
        # Now get the gradients. That's more difficult.
        # Tabulate symbolically
        X = sympy.DeferredVector('x')
        symbolic_tab = self._tabulate(n, X)
        D = 2
        # Symbolically compute the gradient
        grad_results = numpy.empty((len(symbolic_tab), N, D))
        for i, phi in enumerate(symbolic_tab):
            phi_gradient = [sympy.diff(phi, X[j]) for j in range(D)]
            # Evaluate the gradients numerically using lambda expressions
            grad_lambda_tmp = sympy.lambdify(X, phi_gradient)
            grad_lambda = [lambda X: grad_lambda_tmp(X)[0] + 0*X[0],
                           lambda X: grad_lambda_tmp(X)[1] + 0*X[0]
                           ]
            # Evaluate the lambda expressions for pts
            for k in range(D):
                grad_results[i, :, k] = grad_lambda[k](pts.T)
        # Finally put data and grad_results in the required data structure,
        # i.e., an array of 2-tuples the first entry of which is the data
        # value, the second the gradient.
        data = [[(values[i][j], grad_results[i][j])
                 for j in range(values.shape[1])]
                for i in range(values.shape[0])]
        return data

    def tabulate_jet(self, n, pts, order=1):
        dpts = [tuple([Derivatives.DerivVar(pt[i], i, order)
                       for i in range(len(pt))])
                for pt in pts
                ]
        dbfs = self.tabulate(n, dpts)
        result = []
        for d in range(order + 1):
            result_d = [[foo[d] for foo in bar] for bar in dbfs]
            result.append(numpy.array(result_d))

        return result


class TetrahedronExpansionSet:
    """Collapsed orthonormal polynomial expanion on a tetrahedron."""
    def __init__(self, ref_el):
        if ref_el.get_spatial_dimension() != 3:
            raise Exception("Must be a tetrahedron")
        self.ref_el = ref_el
        self.base_ref_el = reference_element.DefaultTetrahedron()
        v1 = ref_el.get_vertices()
        v2 = self.base_ref_el.get_vertices()
        self.A, self.b = reference_element.make_affine_mapping(v1, v2)
        self.mapping = lambda x: numpy.dot(self.A, x) + self.b
        self.scale = numpy.sqrt(numpy.linalg.det(self.A))

        return

    def get_num_members(self, n):
        return (n+1)*(n+2)*(n+3)/6

    def tabulate(self, n, pts):
        if len(pts) == 0:
            return numpy.array([])
        else:
            return numpy.array(self._tabulate(n, numpy.array(pts).T))

    def _tabulate(self, n, pts):
        '''A version of tabulate() that also works for a single point.
        '''
        print 'dasdas'
        m1, m2 = self.A.shape
        ref_pts = [sum(self.A[i][j] * pts[j] for j in range(m2)) + self.b[i]
                   for i in range(m1)
                   ]

        def idx(p, q, r):
            return (p+q+r)*(p+q+r+1)*(p+q+r+2)/6 + (q+r)*(q+r+1)/2 + r

        def jrc(a, b, n):
            an = float((2*n+1+a+b)*(2*n+2+a+b)) \
                / float(2*(n+1)*(n+1+a+b))
            bn = float((a*a-b*b) * (2*n+1+a+b)) \
                / float(2*(n+1)*(2*n+a+b)*(n+1+a+b))
            cn = float((n+a)*(n+b)*(2*n+2+a+b)) \
                / float((n+1)*(n+1+a+b)*(2*n+a+b))
            return an, bn, cn

        results = ((n+1)*(n+2)*(n+3)/6) * [None]
        results[0] = 1.0 \
            + pts[0] - pts[0] \
            + pts[1] - pts[1] \
            + pts[2] - pts[2]

        if n == 0:
            return results

        x = ref_pts[0]
        y = ref_pts[1]
        z = ref_pts[2]

        factor1 = 0.5 * (2.0 + 2.0*x + y + z)
        factor2 = (0.5*(y+z))**2
        factor3 = 0.5 * (1 + 2.0 * y + z)
        factor4 = 0.5 * (1 - z)
        factor5 = factor4 ** 2

        results[idx(1, 0, 0)] = factor1
        for p in range(1, n):
            a1 = (2.0 * p + 1.0) / (p + 1.0)
            a2 = p / (p + 1.0)
            results[idx(p+1, 0, 0)] = a1 * factor1 * results[idx(p, 0, 0)] \
                - a2 * factor2 * results[idx(p-1, 0, 0)]

        # q = 1
        for p in range(0, n):
            results[idx(p, 1, 0)] = results[idx(p, 0, 0)] \
                * (p * (1.0 + y) + (2.0 + 3.0 * y + z) / 2)

        for p in range(0, n-1):
            for q in range(1, n-p):
                (aq, bq, cq) = jrc(2*p+1, 0, q)
                qmcoeff = aq * factor3 + bq * factor4
                qm1coeff = cq * factor5
                results[idx(p, q+1, 0)] = qmcoeff * results[idx(p, q, 0)] \
                    - qm1coeff * results[idx(p, q-1, 0)]

        # now handle r=1
        for p in range(n):
            for q in range(n-p):
                results[idx(p, q, 1)] = results[idx(p, q, 0)] \
                    * (1.0 + p + q + (2.0 + q + p) * z)

        # general r by recurrence
        for p in range(n-1):
            for q in range(0, n-p-1):
                for r in range(1, n-p-q):
                    ar, br, cr = jrc(2*p+2*q+2, 0, r)
                    results[idx(p, q, r+1)] = \
                        (ar * z + br) * results[idx(p, q, r)] \
                        - cr * results[idx(p, q, r-1)]

        for p in range(n+1):
            for q in range(n-p+1):
                for r in range(n-p-q+1):
                    results[idx(p, q, r)] *= \
                        math.sqrt((p+0.5)*(p+q+1.0)*(p+q+r+1.5))

        return results

    def tabulate_derivatives(self, n, pts):
        from Scientific.Functions.FirstDerivatives import DerivVar
        dpts = [[DerivVar(pt[j], j) for j in range(len(pt))] for pt in pts]
        data = self.tabulate(n, dpts)
        return data

    def _tabulate_dpts(self, n, order, pts):
        X = sympy.DeferredVector('x')
        D = 3

        def form_derivative(F):
            '''Forms the derivative recursively, i.e.,
            F               -> [F_x, F_y, F_z],
            [F_x, F_y, F_z] -> [[F_xx, F_xy, F_xz],
                                [F_yx, F_yy, F_yz],
                                [F_zx, F_zy, F_zz]]
            and so forth.
            '''
            out = []
            try:
                out = [sympy.diff(F, X[j]) for j in range(D)]
            except AttributeError:
                # Intercept errors like
                #  AttributeError: 'list' object has no attribute
                #  'free_symbols'
                for f in F:
                    out.append(form_derivative(f))
            return out

        def numpy_lambdify(X, F):
            '''Unfortunately, SymPy's own lambdify() doesn't work well with
            NumPy in that simple functions like
                lambda x: 1.0,
            when evaluated with NumPy arrays, return just "1.0" instead of
            an array of 1s with the same shape as x. This function does that.
            '''
            try:
                lambda_x = [numpy_lambdify(X, f) for f in F]
            except TypeError:  # 'function' object is not iterable
                # SymPy's lambdify also works on functions that return arrays.
                # However, use it componentwise here so we can add 0*x to each
                # component individually. This is necessary to maintain shapes
                # if evaluated with NumPy arrays.
                lmbd_tmp = sympy.lambdify(X, F)
                lambda_x = lambda x: lmbd_tmp(x) + 0*x[0]
            return lambda_x

        def evaluate_lambda(lmbd, x):
            '''Properly evaluate lambda expressions recursively for iterables.
            '''
            try:
                values = [evaluate_lambda(l, x) for l in lmbd]
            except TypeError:  # 'function' object is not iterable
                values = lmbd(x)
            return values

        # Tabulate symbolically
        symbolic_tab = self._tabulate(n, X)
        # Make sure that the entries of symbolic_tab are lists so we can
        # append derivatives
        symbolic_tab = [[phi] for phi in symbolic_tab]
        #
        data = (order+1) * [None]
        for r in range(order+1):
            shape = [len(symbolic_tab), len(pts)] + r * [D]
            data[r] = numpy.empty(shape)
            for i, phi in enumerate(symbolic_tab):
                # Evaluate the function numerically using lambda expressions
                deriv_lambda = numpy_lambdify(X, phi[r])
                data[r][i] = \
                    numpy.array(evaluate_lambda(deriv_lambda, pts.T)).T
                # Symbolically compute the next derivative.
                # This actually happens once too many here; never mind for
                # now.
                phi.append(form_derivative(phi[-1]))
        return data

    def tabulate_jet(self, n, pts, order=1):
        return self._tabulate_dpts(n, order, numpy.array(pts))


def get_expansion_set(ref_el):
    """Returns an ExpansionSet instance appopriate for the given
    reference element."""
    if ref_el.get_shape() == reference_element.LINE:
        return LineExpansionSet(ref_el)
    elif ref_el.get_shape() == reference_element.TRIANGLE:
        return TriangleExpansionSet(ref_el)
    elif ref_el.get_shape() == reference_element.TETRAHEDRON:
        return TetrahedronExpansionSet(ref_el)
    else:
        raise Exception("Unknown reference element type.")


def polynomial_dimension(ref_el, degree):
    """Returns the dimension of the space of polynomials of degree no
    greater than degree on the reference element."""
    if ref_el.get_shape() == reference_element.LINE:
        return max(0, degree + 1)
    elif ref_el.get_shape() == reference_element.TRIANGLE:
        return max((degree+1)*(degree+2)/2, 0)
    elif ref_el.get_shape() == reference_element.TETRAHEDRON:
        return max(0, (degree+1)*(degree+2)*(degree+3)/6)
    else:
        raise Exception("Unknown reference element type.")


if __name__ == "__main__":
    from . import expansions

    E = reference_element.DefaultTriangle()

    k = 3

    pts = E.make_lattice(k)

    Phis = expansions.get_expansion_set(E)

    phis = Phis.tabulate(k, pts)

    dphis = Phis.tabulate_derivatives(k, pts)

#    dphis_x = numpy.array([[d[1][0] for d in dphi] for dphi in dphis])
#    dphis_y = numpy.array([[d[1][1] for d in dphi] for dphi in dphis])
#    dphis_z = numpy.array([[d[1][2] for d in dphi] for dphi in dphis])

#    print dphis_x

#    for dmat in make_dmats(E, k):
#        print dmat
#        print
