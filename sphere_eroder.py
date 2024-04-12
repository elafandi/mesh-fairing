import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from cube_maker import unit_cube

epsilon = 0.1
beta = 1
alpha = 0.5
timestep = 0.01
max_time = 1

def tri_vert_adj(P, T):
    num_verts = P.shape[0]
    num_tri = T.shape[0]
    ind = np.arange(num_tri)

    I = np.hstack((T[:,0],T[:,1],T[:,2]))
    J = np.hstack((ind,ind,ind))
    adj = sparse.coo_matrix((np.ones(len(I)), (I,J)),shape=(num_verts,num_tri)).tocsr()

    return adj

def tri_vert_adj_angles(P, T):
    num_verts = P.shape[0]
    num_tri = T.shape[0]
    
    P1 = P[T[:,0],:]
    P2 = P[T[:,1],:]
    P3 = P[T[:,2],:]
    
    # Create and normalize edge vectors
    vec12 = P2 - P1
    vec23 = P3 - P2
    vec31 = P1 - P3
    vec12 = (vec12.T/np.linalg.norm(vec12,axis=1)).T
    vec23 = (vec23.T/np.linalg.norm(vec23,axis=1)).T
    vec31 = (vec31.T/np.linalg.norm(vec31,axis=1)).T
    
    # Get interior angles of each face
    theta1 = np.arccos(-np.sum(vec12 * vec31, axis=1))
    theta2 = np.arccos(-np.sum(vec23 * vec12, axis=1))
    theta3 = np.arccos(-np.sum(vec31 * vec23, axis=1))
    
    ind = np.arange(num_tri)
    I = np.hstack((T[:,0],T[:,1],T[:,2]))
    J = np.hstack((ind,ind,ind))
    data = np.hstack((theta1, theta2, theta3))
    return sparse.coo_matrix((data,(I,J)),shape=(num_verts,num_tri)).tocsr()

def face_normals(P, T):
    P1 = P[T[:,0],:]
    P2 = P[T[:,1],:]
    P3 = P[T[:,2],:]

    Fn = np.cross(P2-P1,P3-P1)
    norms = np.linalg.norm(Fn,axis=1)
    norms[norms==0] = 1
    return Fn/norms[:,np.newaxis]

def vertex_normals(adjang, Fn):
    Vn = adjang @ Fn
    norms = np.linalg.norm(Vn,axis=1)
    norms[norms==0] = 1
    return Vn/norms[:,np.newaxis]

def tri_areas(P, T):
    P1 = P[T[:,0],:]
    P2 = P[T[:,1],:]
    P3 = P[T[:,2],:]

    Fn = np.cross(P2-P1,P3-P1)
    return np.linalg.norm(Fn,axis=1)/2

def mean_curvature_times_vertex_normals(P, T):
    num_verts = P.shape[0]
    num_tri = T.shape[0]
    
    T1 = T[:,0]
    T2 = T[:,1]
    T3 = T[:,2]
    P1 = P[T[:,0],:]
    P2 = P[T[:,1],:]
    P3 = P[T[:,2],:]
    
    # Create and normalize edge vectors
    vec12 = P2 - P1
    vec23 = P3 - P2
    vec31 = P1 - P3
    vec12 = (vec12.T/np.linalg.norm(vec12,axis=1)).T
    vec23 = (vec23.T/np.linalg.norm(vec23,axis=1)).T
    vec31 = (vec31.T/np.linalg.norm(vec31,axis=1)).T
    
    # Get interior angles of each face
    theta1 = np.arccos(-np.sum(vec12 * vec31, axis=1))
    theta2 = np.arccos(-np.sum(vec23 * vec12, axis=1))
    theta3 = np.arccos(-np.sum(vec31 * vec23, axis=1))
    
    # Take cotangents
    cot1 = 1 / np.tan(theta1)
    cot2 = 1 / np.tan(theta2)
    cot3 = 1 / np.tan(theta3)
    
    # Make vector sums for all vertices
    I = np.hstack((T1, T2, T3))
    J1 = np.hstack((T3, T1, T2))
    data1 = np.hstack(( ((P2 - P1).T * cot3), ((P3 - P2).T * cot1), ((P1 - P3).T * cot2) )).T
    mat1X = sparse.coo_array((data1[:,0],(I,J1)),shape=(num_verts,num_verts)).tocsr()
    mat1Y = sparse.coo_array((data1[:,1],(I,J1)),shape=(num_verts,num_verts)).tocsr()
    mat1Z = sparse.coo_array((data1[:,2],(I,J1)),shape=(num_verts,num_verts)).tocsr()
    
    J2 = np.hstack((T2, T3, T1))
    data2 = np.hstack(( ((P3 - P1).T * cot2), ((P1 - P2).T * cot3), ((P2 - P3).T * cot1) )).T
    mat2X = sparse.coo_array((data2[:,0],(I,J2)),shape=(num_verts,num_verts)).tocsr()
    mat2Y = sparse.coo_array((data2[:,1],(I,J2)),shape=(num_verts,num_verts)).tocsr()
    mat2Z = sparse.coo_array((data2[:,2],(I,J2)),shape=(num_verts,num_verts)).tocsr()
    
    onevec = np.ones(num_verts)
    normal_X = (mat1X + mat2X) @ onevec
    normal_Y = (mat1Y + mat2Y) @ onevec
    normal_Z = (mat1Z + mat2Z) @ onevec
    
    normal_raw = np.column_stack(( normal_X, normal_Y, normal_Z ))
    
    # For all vertices, find sum of areas of bordering triangles
    TA = tri_areas(P, T)
    adj = tri_vert_adj(P, T)
    VA = adj @ TA
    
    HVn = normal_raw / (4 * VA)[:,np.newaxis]
    # By the formula from the Zhao and Xu paper, mean curvature times unit
    # normal vector on a sphere of radius r has a norm of 1/(3r) and is
    # pointed into the sphere. We want 1/r outward.
    return HVn * -3

def gaussian_curvature(P, T, adjang):
    TA = tri_areas(P, T)
    adj = tri_vert_adj(P, T)
    VA = adj @ TA
    V_ang = adjang @ np.ones(T.shape[0])
    
    return np.divide(3 * (2 * np.pi - V_ang), VA)

def derivative_magnitude(H, K):
    num_verts = H.shape[0]
    result = 1 / (1 + np.power(np.abs(K), beta))
    for i in range(0,num_verts):
        if -epsilon < H[i] < epsilon:
            result[i] *= H[i]
            colors[i,0] = 1  # Red
        elif K[i] > 0:
            result[i] *= np.sign(H[i]) * K[i]
            colors[i,1] = 1  # Green
            if H[i] > 0:
                colors[i,0] = 1  # Yellow
        else:
            result[i] *= alpha * K[i]
            colors[i,2] = 1  # Blue
    return result

#P = np.genfromtxt('meshes/p_sphere_7446.txt', delimiter=',')
#T = np.genfromtxt('meshes/t_sphere_7446.txt', delimiter=',', dtype=int)
#T = T[:,::-1] - 1 # matlab indexes from 1 and orients in reverse
P,T = unit_cube(10)
P = P * 2

colors = np.zeros((P.shape[0],4))
colors[:,3] = 1

fig = plt.figure(figsize=plt.figaspect(1))
ax = fig.add_subplot(projection='3d',autoscale_on=False)
surf = None
arrows = None
dots = None

# ax.set_xbound(-1.25, 1.25)
# ax.set_ybound(-1.25, 1.25)
# ax.set_zbound(-1.25, 1.25)
ax.set_xbound(-0.25, 2.25)
ax.set_ybound(-0.25, 2.25)
ax.set_zbound(-0.25, 2.25)
#surf.set_fc(colors)

def update(frame):
    global P, surf, ax, colors, arrows, dots
    colors = np.zeros((P.shape[0],4))
    colors[:,3] = 1
    
    adjang = tri_vert_adj_angles(P, T)
    Fn = face_normals(P, T)
    Vn = vertex_normals(adjang, Fn)
    HVn = mean_curvature_times_vertex_normals(P, T)
    H = np.sum(HVn * Vn,axis=1)
    K = gaussian_curvature(P, T, adjang)
    print("min H =", min(H), "; max H =", max(H))
    print("min K =", min(K), "; max K =", max(K))
    der_mag = derivative_magnitude(H,K)
    der = der_mag[:, np.newaxis] * Vn
    
    if surf: surf.remove()
    if arrows: arrows.remove()
    if dots: dots.remove()
    print("t =", frame)
    surf = ax.plot_trisurf(P[:,0], P[:,1], T, P[:,2], edgecolor='k',
                color='RoyalBlue', linewidth=1.0, shade=False)
    #arrows = ax.quiver(P[:,0], P[:,1], P[:,2], der[:,0], der[:,1], der[:,2], normalize=False, length=0.5, color=colors)
    #dots = ax.scatter(P[:,0], P[:,1], P[:,2], c=colors)
    
    P = P - timestep * der

ani = FuncAnimation(fig, update, frames=np.arange(0, max_time, timestep), repeat=False,
                    init_func=lambda : None)
#ani.save("cube_shrink.gif", writer=PillowWriter(fps=20))
plt.show()