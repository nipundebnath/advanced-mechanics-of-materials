import numpy as np
from itertools import combinations, permutations

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

def get_cuboid_faces( center=(0,0,0), size=(1,1,1) ):
    '''
    Returns six faces of a cuboid
    
    Input:
        center: (x, y, z) center of the cuboid
        size:   width, height, depth
                (length along x-, y-, z-axis, respectively)
                
    Output:
        faces:  a list of 6 np.ndarray
                each representing a face of the cuboid
                each face contains 4 vertices
    '''

    center = np.array( center )
    dx     = np.array( size )
    
    ### Vertices
    points      = np.zeros( (8,3) ) + center
    points[0]   = points[0]   - dx / 2
    points[1]   = points[0]   + dx * [1, 0, 0]
    points[2:4] = points[0:2] + dx * [0, 1, 0]
    points[4: ] = points[ :4] + dx * [0, 0, 1]
    
    ### Order of the points for a valid face
    pts_order_0 = np.array( [0, 4, 6, 2] ) # left
    pts_order_1 = np.array( [0, 1, 5, 4] ) # bottom
    pts_order_2 = np.array( [0, 1, 3, 2] ) # front
    
    ### Faces
    faces = [ 
        points[ pts_order_0     ], # left
        points[ pts_order_0 + 1 ], # right
        points[ pts_order_1     ], # bottom
        points[ pts_order_1 + 2 ], # top
        points[ pts_order_2     ], # front
        points[ pts_order_2 + 4 ], # back
    ]
    
    return faces

def get_arrow_head_faces( base_center=(0,0,0), scale=0.1 ):
    '''
    Returns pentahedron faces
    '''
    dx = np.ones(3) * scale
    
    ### Pentahedron vertices
    points      = np.zeros((5,3))  + base_center
    points[0]   = points[0]   - dx * [0, 1, 1] / 2
    points[1]   = points[0]   + dx * [0, 1, 0]
    points[2:4] = points[0:2] + dx * [0, 0, 1]
    points[4]   = points[4]   + dx * [1, 0, 0]
    
    ### Pentahedron faces
    faces = [ 
        points[ [0, 2, 3, 1] ],
        points[ [0, 1, 4] ],
        points[ [0, 2, 4] ],
        points[ [2, 3, 4] ],
        points[ [3, 1, 4] ]
    ]
    
    return faces

def get_arrow_faces( start_point=(0,0,0), length=1, body_scale=0.05, head_scale=0.1 ):
    '''
    Returns polygonal faces of an arrow
    '''
    start_point = np.array(start_point)
    shape       = np.array( [ length, body_scale, body_scale ] )
    
    ### arrow body
    center     = start_point + shape * [1, 0, 0] / 2
    body_faces = get_cuboid_faces( center, shape )
    
    ### arrow_head
    base_center = start_point + shape * [1, 0, 0]
    head_faces  = get_arrow_head_faces( base_center, head_scale )
    
    return body_faces + head_faces # concatenated list

def transform( faces, transformation=None, translation=None ):
    '''
    Retruns faces in the transformed coordinate position
    '''
    ### Prepare
    if transformation is None:
        transformation = np.eye(3)
    else:
        transformation = np.array(transformation)
    
    if translation is None:
        translation = np.zeros(3)
    else:
        translation = np.array(translation)
        
    ### Transformation + Translation
    for i, fc in enumerate(faces):
        faces[i] = np.matmul( fc , transformation ) + translation
    return faces

def get_figure_ready( ax ):
    '''
    Prepares the figure for a clean view
    '''    
    ax.set_xlim( [-2, 2] )
    ax.set_ylim( [-2, 2] )
    ax.set_zlim( [-2, 2] )
    ax.set_axis_off()
    
    return ax

def show_axes_direction( ax, origin_point=[0,0,0], arrow_length=1 ):
    '''
    Shows global axes direction
    '''
    ### Arrow
    for trans_mat in [None, rotate_x1_to_x2, rotate_x1_to_x3]:
        # arrow faces
        arr_faces = get_arrow_faces( start_point=(0,0,0), length=arrow_length, body_scale=0.025, head_scale=0.15 )
        arr_faces = transform( arr_faces, trans_mat, origin_point )
        # draw
        poly_faces = Poly3DCollection( arr_faces, facecolor=(0,0,1,1), linewidths=0 )
        ax.add_collection3d(poly_faces)
    
    ### Text
    text_pos = np.eye(3) * (arrow_length + 0.4) + origin_point
    for i in range(3):
        ax.text( *text_pos[i], '$x_%s$' % (i+1), color='b', ha='center')
        
    return ax

'''Common transformation matrices'''
rotate_x1_to_x2 = np.array([
    [0, 1, 0],
    [-1,  0, 0],
    [0,  0, 1]
])
rotate_x1_to_x3 = np.array([
    [0,  0, 1],
    [0,  1, 0],
    [-1, 0, 0]
])

def plot_element_body( ax, tensor_len=2, tensor_ax_len=1, transformation=None ):
    '''
    Plots cuboid shape of the element body
    '''
    ### Get cuboid faces
    faces = get_cuboid_faces( size = np.ones(3)*tensor_len )
    # transformation
    if transformation is not None:
        faces = transform( faces, transformation )
    # draw
    poly_faces = Poly3DCollection( faces, facecolor=(1,1,1,0), linewidths=1, edgecolors='grey' )
    ax.add_collection3d(poly_faces)

    ### Tensor axis
    for trans_mat in [None, rotate_x1_to_x2, rotate_x1_to_x3]:
        # arrow faces
        arr_faces = get_arrow_faces( start_point=( tensor_len/2, 0, 0 ), length=tensor_ax_len, body_scale=0.025, head_scale=0.15 )
        arr_faces = transform( arr_faces, trans_mat )
        # transformation
        if transformation is not None:
            arr_faces = transform( arr_faces, transformation )
        # draw
        poly_faces = Poly3DCollection( arr_faces, facecolor=(1, 0, 0, 1), linewidths=0, zorder=-1 )
        ax.add_collection3d(poly_faces)
    
    ### Axes labels
    text_pos = np.eye(3) * (tensor_len/2 + tensor_ax_len + 0.5)
    # transformation
    if transformation is not None:
        text_pos = transform( text_pos, transformation )
    # draw
    for i in range(3):
        ax.text( *text_pos[i], '$\hat{e}_%s$' %(i+1), color='r', ha='center')
        
    return ax

def plot_tensor_component( ax, pos=(1,1), val=1, label_val=1, tensor_len=1, transformation=None, label='value' ):
    '''
    Shows tensor component on the face of the element body as arrows
    
    label: {'symbol', 'value', 'full'}
    '''
    if val == 0:
        return ax
    
    ### Get the faces
    arr_faces = get_arrow_faces( length=abs(val), body_scale=0.05, head_scale=0.2 )
    
    ### For negative value, reverse the arrow
    sign = 1
    if val < 0:
        arr_faces = transform( arr_faces, -1*np.eye(3) )
        sign = -1
    
    ### Start-end point of the arrow
    start_point = np.zeros(3)
    start_point[ pos[0]-1 ] = tensor_len / 2
    end_point = (val + 0.4*sign) * np.array([1, 0, 0])
    
    ### Put to proper position
    # rotation + translation
    rotation_matrix = [None, rotate_x1_to_x2, rotate_x1_to_x3] [ pos[1]-1 ]
    arr_faces = transform( arr_faces, rotation_matrix, start_point )
    end_point = transform( [end_point], rotation_matrix, start_point )[0]
    
    # transformation
    if transformation is not None:
        arr_faces = transform( arr_faces, transformation )
        end_point = transform( [end_point], transformation )[0]
    
    ### Draw
    if val:
        poly_faces = Poly3DCollection( arr_faces, facecolor=(0,0,0,1), linewidths=0 )
        ax.add_collection3d(poly_faces)
    
    ### Text
    if label == 'symbol':
        ax.text( *end_point, '$\sigma_{%s%s}$' % (pos[0], pos[1]), color='k', ha='center')
    elif label == 'value':
        ax.text( *end_point, '%.0f' % label_val, color='k', ha='center')
    else:
        ax.text( *end_point, '$\sigma_{%s%s} = %.0f$' % (pos[0], pos[1], label_val), color='k', ha='center')
    
    return ax

def plot_tensor_element (ax, tensor, axes=None, max_value=1):
    
    ### Geometric size
    TENSOR_LEN = 3
    TENSOR_AX_LEN = 1.5
    
    ### If no axes is given, use default global axes
    if axes is None:
        axes = np.eye(3)
        
    ### Scale the tensor values
    scaled_tensor = tensor / max_value
    
    ### Prepare the plot
    ax = get_figure_ready    ( ax )
    ax = show_axes_direction ( ax, origin_point=[2,-2,-2], arrow_length=1 )
    ax = plot_element_body   ( ax, tensor_len=TENSOR_LEN, tensor_ax_len=TENSOR_AX_LEN, transformation=axes )
    
    ### Show tensor components
    for i in range(3):
        for j in range(3):
            ax = plot_tensor_component( ax, pos=(i+1,j+1), val=scaled_tensor[i,j], label_val=tensor[i,j],
                                       tensor_len=TENSOR_LEN, transformation=axes )
    
    # set default view
    ax.view_init(elev=25, azim=45)
    
    return ax