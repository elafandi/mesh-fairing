import numpy as np

def unit_cube(n):
    # Given an integer n >= 2,
    # create a triangular mesh of the unit cube,
    # where each cube face has n^2 vertices in a grid pattern.
    
    rise = np.linspace(1/(n-1), (n-2)/(n-1), n-2)
    
    P = np.zeros((6 * n**2 - 12 * n + 8, 3))
    
    # CORNERS:
    P[4,:] = [0, 0, 0]
    P[5,:] = [1, 0, 0]
    P[6,:] = [1, 1, 0]
    P[7,:] = [0, 1, 0]
    P[0,:] = [0, 0, 1]
    P[1,:] = [1, 0, 1]
    P[2,:] = [1, 1, 1]
    P[3,:] = [0, 1, 1]
    
    # EDGES:
    # Edge 0
    P[8:n+6,0] = rise
    P[8:n+6,1] = 0
    P[8:n+6,2] = 1
    # Edge 1
    P[n+6:2*n+4,0] = 1
    P[n+6:2*n+4,1] = rise
    P[n+6:2*n+4,2] = 1
    # Edge 2
    P[2*n+4:3*n+2,0] = rise
    P[2*n+4:3*n+2,1] = 1
    P[2*n+4:3*n+2,2] = 1
    # Edge 3
    P[3*n+2:4*n,0] = 0
    P[3*n+2:4*n,1] = rise
    P[3*n+2:4*n,2] = 1
    # Edge 4
    P[4*n:5*n-2,0] = 0
    P[4*n:5*n-2,1] = 0
    P[4*n:5*n-2,2] = rise
    # Edge 5
    P[5*n-2:6*n-4,0] = 1
    P[5*n-2:6*n-4,1] = 0
    P[5*n-2:6*n-4,2] = rise
    # Edge 6
    P[6*n-4:7*n-6,0] = 1
    P[6*n-4:7*n-6,1] = 1
    P[6*n-4:7*n-6,2] = rise
    # Edge 7
    P[7*n-6:8*n-8,0] = 0
    P[7*n-6:8*n-8,1] = 1
    P[7*n-6:8*n-8,2] = rise
    # Edge 8
    P[8*n-8:9*n-10,0] = rise
    P[8*n-8:9*n-10,1] = 0
    P[8*n-8:9*n-10,2] = 0
    # Edge 9
    P[9*n-10:10*n-12,0] = 1
    P[9*n-10:10*n-12,1] = rise
    P[9*n-10:10*n-12,2] = 0
    # Edge 10
    P[10*n-12:11*n-14,0] = rise
    P[10*n-12:11*n-14,1] = 1
    P[10*n-12:11*n-14,2] = 0
    # Edge 11
    P[11*n-14:12*n-16,0] = 0
    P[11*n-14:12*n-16,1] = rise
    P[11*n-14:12*n-16,2] = 0
    
    # INTERIORS OF FACES:
    tiled = np.tile(rise,n-2)
    repeated = np.repeat(rise,n-2)
    # Face 0
    P[12*n-16:n**2+8*n-12,0] = tiled
    P[12*n-16:n**2+8*n-12,1] = repeated[::-1]
    P[12*n-16:n**2+8*n-12,2] = 1
    # Face 1
    P[n**2+8*n-12:2*n**2+4*n-8,0] = repeated
    P[n**2+8*n-12:2*n**2+4*n-8,1] = 0
    P[n**2+8*n-12:2*n**2+4*n-8,2] = tiled
    # Face 2
    P[2*n**2+4*n-8:3*n**2-4,0] = 0
    P[2*n**2+4*n-8:3*n**2-4,1] = tiled[::-1]
    P[2*n**2+4*n-8:3*n**2-4,2] = repeated[::-1]
    # Face 3
    P[3*n**2-4:4*n**2-4*n,0] = repeated[::-1]
    P[3*n**2-4:4*n**2-4*n,1] = 1
    P[3*n**2-4:4*n**2-4*n,2] = tiled
    # Face 4
    P[4*n**2-4*n:5*n**2-8*n+4,0] = 1
    P[4*n**2-4*n:5*n**2-8*n+4,1] = tiled
    P[4*n**2-4*n:5*n**2-8*n+4,2] = repeated[::-1]
    # Face 5
    P[5*n**2-8*n+4:6*n**2-12*n+8,0] = tiled
    P[5*n**2-8*n+4:6*n**2-12*n+8,1] = repeated
    P[5*n**2-8*n+4:6*n**2-12*n+8,2] = 0
    
    # And now to set up T
    
    # indices of points on edges
    e0 = np.array(range(8,n+6))
    e1 = np.array(range(n+6,2*n+4))
    e2 = np.array(range(2*n+4,3*n+2))
    e3 = np.array(range(3*n+2,4*n))
    e4 = np.array(range(4*n,5*n-2))
    e5 = np.array(range(5*n-2,6*n-4))
    e6 = np.array(range(6*n-4,7*n-6))
    e7 = np.array(range(7*n-6,8*n-8))
    e8 = np.array(range(8*n-8,9*n-10))
    e9 = np.array(range(9*n-10,10*n-12))
    e10 = np.array(range(10*n-12,11*n-14))
    e11 = np.array(range(11*n-14,12*n-16))
    
    def triangulate_face(face):
        nw = np.resize(face[:n-1,:n-1], (n**2 - 2*n + 1,1))
        ne = np.resize(face[:n-1,1:n], (n**2 - 2*n + 1,1))
        sw = np.resize(face[1:n,:n-1], (n**2 - 2*n + 1,1))
        se = np.resize(face[1:n,1:n], (n**2 - 2*n + 1,1))
        print(nw.T)
        print(ne.T)
        print(sw.T)
        print(se.T)
        print()
        return np.vstack((np.hstack((nw, sw, ne)), np.hstack((sw, se, ne))))
    
    # Face 0
    face = np.vstack((
        np.hstack(( 3, e2, 2 )),
        np.hstack((
            np.resize(e3,(n-2,1))[::-1],
            np.resize(range(12*n-16, n**2+8*n-12), (n-2,n-2)),
            np.resize(e1,(n-2,1))[::-1]
        )),
        np.hstack(( 0, e0, 1 ))
    ))
    print(face)
    T = triangulate_face(face)
    
    # Face 1
    face = np.vstack((
        np.hstack(( 4, e4, 0 )),
        np.hstack((
            np.resize(e8,(n-2,1)),
            np.resize(range(n**2+8*n-12, 2*n**2+4*n-8), (n-2,n-2)),
            np.resize(e0,(n-2,1))
        )),
        np.hstack(( 5, e5, 1 ))
    ))
    T = np.vstack((T, triangulate_face(face)))
    
    # Face 2
    face = np.vstack((
        np.hstack(( 3, e3[::-1], 0 )),
        np.hstack((
            np.resize(e7,(n-2,1))[::-1],
            np.resize(range(2*n**2+4*n-8, 3*n**2-4), (n-2,n-2)),
            np.resize(e4,(n-2,1))[::-1]
        )),
        np.hstack(( 7, e11[::-1], 4 ))
    ))
    T = np.vstack((T, triangulate_face(face)))
    
    # Face 3
    face = np.vstack((
        np.hstack(( 6, e6, 2 )),
        np.hstack((
            np.resize(e10,(n-2,1))[::-1],
            np.resize(range(3*n**2-4, 4*n**2-4*n), (n-2,n-2)),
            np.resize(e2,(n-2,1))[::-1]
        )),
        np.hstack(( 7, e7, 3 ))
    ))
    T = np.vstack((T, triangulate_face(face)))
    
    # Face 4
    face = np.vstack((
        np.hstack(( 1, e1, 2 )),
        np.hstack((
            np.resize(e5,(n-2,1))[::-1],
            np.resize(range(4*n**2-4*n, 5*n**2-8*n+4), (n-2,n-2)),
            np.resize(e6,(n-2,1))[::-1]
        )),
        np.hstack(( 5, e9, 6 ))
    ))
    T = np.vstack((T, triangulate_face(face)))
    
    # Face 5
    face = np.vstack((
        np.hstack(( 4, e8, 5 )),
        np.hstack((
            np.resize(e11,(n-2,1)),
            np.resize(range(5*n**2-8*n+4, 6*n**2-12*n+8), (n-2,n-2)),
            np.resize(e9,(n-2,1))
        )),
        np.hstack(( 7, e10, 6 ))
    ))
    T = np.vstack((T, triangulate_face(face)))
    
    return P,T

# import matplotlib.pyplot as plt
# P,T = unit_cube(5)
# print(T)
# fig = plt.figure(figsize=plt.figaspect(1))
# ax = fig.add_subplot(projection='3d',autoscale_on=False)
# surf = ax.plot_trisurf(P[:,0], P[:,1], T, P[:,2], edgecolor='k',
                # color='RoyalBlue', linewidth=1.0, shade=False)
# plt.show()