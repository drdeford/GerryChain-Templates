"""
Created on Wed May 15 11:54:57 2019
@author: daryl

#######################################################################
# BASED On An Original Sage Version by:
# AUTHOR: Dr. Christian Schridde
# E-MAIL: christianschridde [at] googlemail [dot] com
#
# DESCRIPTION: Implementation of the FKT-algorithm
#
# INPUT:  Adjacency matrix A of a undirected loop-free planar graph G
# OUTPUT: The number of perfect matchings of G
########################################################################
"""

import networkx as nx  # Requires at least networkx 2.3+
import math
import numpy as np


# Helper Functions
def doNothing():
    return 0


def find_faces(embd):

    # Returns a list of faces of the planar embedding by
    # the edges that bound the face
    ed_list = list(embd.edges())
    faces = []

    for ed in embd.edges():
        if ed in ed_list:
            faces.append(embd.traverse_face(ed[0], ed[1]))

            for i in range(len(faces[-1])):
                ed_list.remove((faces[-1][i], faces[-1][(i + 1) % len(faces[-1])]))

    face_edges = []
    for face in faces:
        face_edges.append([])
        for i in range(len(face)):
            face_edges[-1].append((face[i], face[(i + 1) % len(face)]))

    return face_edges


def toSkewSymmetricMatrix(A):
    # Skew--symmetrize a matrix

    A[(A == 1).T] = -1

    return A


def numberOfClockwiseEdges(face, edgesT1):

    # Iterate over edges of a face to determine
    # the number of positive orientations

    clockwise = 0
    for edge in face:
        try:
            edgesT1.index(edge)
        except ValueError:
            doNothing()
        else:
            clockwise += 1
    return clockwise


def isClockwise(e, face):
    # Checks orientation of an edge on a face
    try:
        face.index(e)
    except ValueError:
        return False
    else:
        return True


# Main Function
def FKT(A):
    n = len(A)

    G = nx.Graph(A)

    tf, embd = nx.check_planarity(G)

    if embd is None:
        return 0

    faces = find_faces(embd)

    T1 = nx.minimum_spanning_tree(G)
    T1 = nx.Graph(T1)

    mask = np.random.randint(2, size=(n, n))
    mask = (mask + mask.T) == 1

    B_digraph = A * mask

    G = nx.DiGraph(B_digraph)

    edgesT1 = T1.edges()
    adj_T1 = (nx.adjacency_matrix(T1)).todense()

    for edge in edgesT1:
        if B_digraph[edge[0], edge[1]] == 0:
            adj_T1[edge[0], edge[1]] = 0
        else:
            adj_T1[edge[1], edge[0]] = 0

    T1 = nx.DiGraph(adj_T1)
    edgesT1 = list(T1.edges())
    if embd is not None:
        faces.sort(key=len)
        faces.reverse()
        faces.pop(0)

    if embd is not None:
        while len(faces) > 0:
            index = -1
            for face in faces:
                countMissingEdges = 0
                missingEdge = 0
                index += 1
                for edge in face:
                    try:
                        edgesT1.index(edge)
                    except ValueError:
                        try:
                            edgesT1.index((edge[1], edge[0]))
                        except ValueError:
                            countMissingEdges += 1
                            missingEdge = edge
                        else:
                            doNothing()
                    else:
                        doNothing()

                if countMissingEdges == 1:
                    # in this face, only one edge is missing.
                    # Place the missing edge such that the total number
                    # of clockwise edges of this face is odd
                    # add this edge to the spanning tree
                    if (numberOfClockwiseEdges(face, edgesT1)) % 2 == 1:
                        # insert counterclockwise in adj_T1;
                        if not isClockwise(missingEdge, face):
                            adj_T1[missingEdge[0], missingEdge[1]] = 1
                        else:
                            adj_T1[missingEdge[1], missingEdge[0]] = 1
                    else:
                        # insert clockwise in adj_T1
                        if isClockwise(missingEdge, face):
                            adj_T1[missingEdge[0], missingEdge[1]] = 1
                        else:
                            adj_T1[missingEdge[1], missingEdge[0]] = 1

                    # rebuild the graph
                    T1 = nx.DiGraph(adj_T1)
                    edgesT1 = list(T1.edges())

                    # remove the face that was found
                    faces.pop(index)
                    break
        try:
            return math.sqrt(np.linalg.det(toSkewSymmetricMatrix(adj_T1)))
        except ValueError:
            pass
