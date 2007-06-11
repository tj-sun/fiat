# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# Edited 27 Sept 2005 by RCK, 
# Edited 26 Sept 2005 by RCK

import dualbasis, polynomial, functionalset, functional, shapes, \
       quadrature, numpy, PhiK

def BDFMSpace( shape , k ):
    U = polynomial.OrthogonalPolynomialArraySet( shape , k )
    fnm = functional.FacetDirectionalMoment
    d = shapes.dimension( shape )

    phis = polynomial.OrthogonalPolynomialSet( shape - 1 , k )
    dimPkm1 = shapes.polynomial_dimension( shape -1 , k-1 )
    dimPk = shapes.polynomial_dimension( shape-1,k )
    
    # one set of constraint for each face
    constraints = []
    for e in shapes.entity_range( shape , d - 1 ):
        n = shapes.normals[ shape ][ e ]
        constraints.extend( [ fnm( U , shape , n , d-1 , e , phis[j] ) \
                              for j in range(dimPkm1,dimPk) ] )

    fset = functionalset.FunctionalSet( U , constraints )

    return polynomial.ConstrainedPolynomialSet( fset )


class BDFMDual( dualbasis.DualBasis ):
    def __init__( self , shape , k , U ):
        mdcb = functional.make_directional_component_batch
        d = shapes.dimension( shape )
        pts_per_edge = [ [ x \
                           for x in shapes.make_points( shape , \
                                                        d-1 , \
                                                        i , \
                                                        d+k-1 ) ] \
                        for i in shapes.entity_range( shape , d-1 ) ]
        nrmls = shapes.normals[shape]
        ls = reduce( lambda a,b:a+b , \
                     [ mdcb(U,nrmls[i],pts_per_edge[i]) \
                       for i in shapes.entity_range(shape,d-1) ] )
        interior_moments = []

        # internal moments against gradients of polynomials
        # of degree k-1 (only if k > 1)

        if k > 1:
            pk = polynomial.OrthogonalPolynomialSet(shape,k)
            pkm1 = pk[1:shapes.polynomial_dimension(shape,k-1)]

            pkm1grads = [ polynomial.gradient( p ) for p in pkm1 ]

            interior_moments.extend( [ functional.IntegralMoment( U , pg ) \
                                       for pg in pkm1grads ] )

        # internal moments against div-free polynomials with
        # vanishing normal component (only if n > 2)
        if k > 1:
            PHIK = PhiK.PhiK( shape , k , U )

            interior_moments.extend( [ functional.IntegralMoment( U ,  phi ) \
                                       for phi in PHIK ] )

        ls.extend( interior_moments )
        entity_ids = {}
        for i in range(d-1):
            entity_ids[i] = {}
            for j in shapes.entity_range(shape,i):
                entity_ids[i][j] = []
        pts_per_bdry = len(pts_per_edge[0])
        entity_ids[d-1] = {}
        node_cur = 0
        for j in shapes.entity_range(shape,d-1):
            for k in range(pts_per_bdry):
                entity_ids[d-1][j] = node_cur
                node_cur += 1
        entity_ids[d] = range(node_cur,\
                              node_cur+len(interior_moments))


        dualbasis.DualBasis.__init__( self , \
                                      functionalset.FunctionalSet( U , ls ) , \
                                      entity_ids )



class BDFM( polynomial.FiniteElement ):
    def __init__( self , shape , n ):
        U = BDFMSpace( shape , n )
        Udual = BDFMDual( shape , n , U )
        polynomial.FiniteElement.__init__( self , Udual , U )


