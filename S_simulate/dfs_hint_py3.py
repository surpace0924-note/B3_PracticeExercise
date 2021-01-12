#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dfs_io import *
import requests
import turtle as tt
import time
import random

__author__ = 'takei'


###########################
# DFS edge cover program (hint file)
###########################

#
# Building Blocks (クラスの設定)
#

class Edge:
    def __init__(self, id):
        self.id = id  # identifier >= 0
        self.source_vidbid = [-1, -1]  # vertex/branch id of the first end
        self.destin_vidbid = [-1, -1]  # and the second end
        self.trajectory = []  # the list of coordinates

    def SetSource(self, vidbid):
        self.source_vidbid = vidbid

    def SetDestin(self, vidbid):
        self.destin_vidbid = vidbid

    def PutPos(self, pos):
        self.trajectory.append(pos)


class EdgeDB:
    def __init__(self):
        self.lastid = -1  # the valid smallest id is 0
        self.edges = []  # list of known edges

    def NewEdge(self):
        self.lastid = self.lastid + 1
        self.edges.append(Edge(self.lastid))
        return self.edges[self.lastid]

    def SearchEdgeById(self, id):
        for e in self.edges:
            if e.id == id:
                return e
        return []

    def Dump(self, fp):
        fp.write('###\n')
        fp.write('EdgeDB\n')
        for e in self.edges:
            fp.write(str([e.id, e.source_vidbid, e.destin_vidbid, e.trajectory]) + '\n')


class Branch:
    def __init__(self, id, vec, v, e):
        self.id = id  # its own id >= 0
        self.vec = vec  # departing vector
        self.vertex = v  # the vertex to which self belongs
        self.edge = e  # the edge to which self is connected
        self.otherend_vidbid = [-1, -1]  # vid/bid of the other side via edge
        # it is unknown when self is created

    def SetVec(self, vec):
        self.vec = vec

    def GetVec(self):
        return self.vec

    def SetEdge(self, e):
        self.edge = e

    def SetOtherEnd(self, vidbid):
        self.otherend_vidbid = vidbid

    def GetOtherEnd(self):
        return self.otherend_vidbid


class Vertex:
    def __init__(self, id, pos, canvec):
        self.id = id  # vertex id >= 0
        self.pos = pos  # position (in a complex number)
        self.branch = []  # branches
        i = 0
        for vec in canvec:  # for each of candidate vectors
            self.branch.append(Branch(i, vec, self, -1))  # a branch is assigned
            i = i + 1

    def SearchBranchByVec(self, vec):
        print("K", vec)
        for b in self.branch:
            print("V", b.vec)
            print(abs(b.vec - vec))
            if abs(b.vec - vec) < 0.02:
                return b
        return None

    def SearchBranchByOtherEnd(self, vidbid):
        print("vidbid", vidbid)
        for b in self.branch:
            print("b.vidbid", b.otherend_vidbid)
            if b.otherend_vidbid == vidbid:
                return b
        return None


class VertexDB:
    def __init__(self):  # DB of known vertices
        self.lastid = -1  # the smallest valid vid is 0
        self.vertices = []  # the list containing vertices

    def NewVertex(self, pos, canvec):
        self.lastid = self.lastid + 1
        self.vertices.append(Vertex(self.lastid, pos, canvec))
        return self.vertices[self.lastid]

    def SearchVertexByPos(self, pos):
        for v in self.vertices:
            if abs(v.pos - pos) < 1:
                return v
        # return []
        return None

    def Dump(self, fp):
        fp.write('VertexDB\n')
        for v in self.vertices:
            fp.write(str([v.id, v.pos]) + '\n')
            for b in v.branch:
                fp.write(str(['b', b.id, b.vec, b.otherend_vidbid, 'e', b.edge.id]) + '\n')


def searchUnknownBranches(v):
    for x in v.branch:
        if x.otherend_vidbid[0] == -1 and x.otherend_vidbid[1] == -1:
            return x
    return None


#####################
# the main procedure　メイン
#####################

if __name__ == '__main__':

    # HTTP設定 ===================
    ip, port = '127.0.0.1', '30002'
    contact = Comunicator(ip, port)

    # (0,0) will reset server's position back to default (1,1)
    # Because DFS uses complex number, we shall follow
    # (複素数)
    pos, canvec = contact.get_data(complex(0, 0))

    # hire a mapper to draw our map　(マップ生成)
    pi = Plotter()
    pi.start_at(-200, 100)

    # -------below this line is DFS　（以下が深さ優先探索）

    # Initialize global objects
    VDB = VertexDB()  # DB of vertices
    EDB = EdgeDB()  # DB of edges
    vec = complex(0, 0)

    # Set initial position
    pos, canvec = contact.get_data(vec)

    print("pos", pos)
    print("canvec", canvec)

    # Generate the first vertex
    v = VDB.NewVertex(pos, canvec)
    vec = canvec[0]  # assuming that it is terminal
    b = v.SearchBranchByVec(vec)  # keep the departing branch

    vidbid = [v.id, b.id]
    direction = 1  # forward
    stack = [vidbid]  # Initialize stack by current vid/bid pair

    # Obtain the first edge and maintain it
    e = EDB.NewEdge()
    e.PutPos(pos)
    e.SetSource(vidbid)

    ###########################################################
    # The DFS loop (until the first vidbid is popped from
    #  the stack)　(以下を完成させる)
    ###########################################################
    while 1:
        # draw the map
        pi.goto(vec, direction)
        # Do a move, then obtain the new position and
        # candidate vectors from the robot(simulated)
        pos, canvec = contact.get_data(vec)

        print('Position:', pos)
        print('Candidate vectors:', canvec)

        #
        #       |<-- Regular or Intersection ??
        #

        if len(canvec) == 2:  # The number of branch is 2;
            # thus regular point
            #
            #           |<-- Forward or Backward ??
            if direction == 1:  # we are  moving forward,
                e.PutPos(pos)  # so update the trajectory
            # register current position to the edge
            # Updating the velocity.
            # the velocity vector should be one that
            # is not opposite of the previous velocity vector
            if canvec[0] == -vec:
                vec = canvec[1]
            else:
                vec = canvec[0]
            print("Vector:", vec)

        #       |<-- Regular or Intersection ??
        else:  # We are at an intersection (or the terminal)
            # There are two possibilities (new/known) here.
            ##############################################
            print(stack)
            # Essential Part! You must program this
            # long block by your help.
            # Reference; 20151215 handout

            # search VertexDB of pos
            # new or known ??
            # x = VDB.SearchVertexByPos(pos)
            # if len(x) == 0:
            if VDB.SearchVertexByPos(pos) == None:
                # new vertex
                v = VDB.NewVertex(pos, canvec)
                # identify branches
                '''b2 = v.SearchBranchByVec(vec)
                vidbid_other = [ v.id,b2.id ]
                b.otherend_vidbid = vidbid_other
                b2.otherend_vidbid = vidbid
                e.destin_vidbid = vidbid_other'''
            else:
                # known vertex
                v = VDB.SearchVertexByPos(pos)

            # select vec in canvec?
            vec *= -1
            b2 = v.SearchBranchByVec(vec)
            vec *= -1
            print(v.id)
            print(v.pos)
            print(v.branch)
            print(b2)
            # v has id,pos and branch[], but b2 is "None"

            # forward or backward ??
            if direction == 1:
                # forward

                # arrival process
                '''v = VDB.NewVertex(pos,canvec)'''
                vidbid_other = [v.id, b2.id]
                b = VDB.vertices[stack[0][0]].branch[stack[0][1]]
                print("b = VDB.vertices[stack[0][0]].branch[stack[0][1]],b")
                b.SetOtherEnd(vidbid_other)
                b2.SetOtherEnd(vidbid)
                e.SetDestin(vidbid_other)
                b.SetEdge(e)
                b2.SetEdge(e)
                # stack.insert(0,vidbid_other)

                # departure process
                if len(canvec) > 2 and VDB.vertices[len(VDB.vertices) - 1] == v:
                    # forward -> forward
                    print("FWD,FWD")
                    """for x in v.branch:
                        if x.otherend_vidbid[0] == -1 and  x.otherend_vidbid[1] == -1:
                            b = x
                            break"""
                    b = v.SearchBranchByOtherEnd([-1, -1])
                    vidbid = [v.id, b.id]
                    stack.insert(0, vidbid)
                    e = EDB.NewEdge()
                    e.SetSource(vidbid)
                    vec = b.GetVec()
                    print("vec", vec)
                else:
                    # forward -> backward
                    print("FWD,BACK")
                    direction = -1
                    b = b2
                    vec *= -1
                    print("vec", vec)

            else:
                # backward
                # search unknown branches
                # b = searchUnknownBranches(v)
                b = v.SearchBranchByOtherEnd([-1, -1])
                # departure process
                if b != None:
                    # backward -> forward
                    print("BACK,FWD")
                    direction = 1
                    # vec = b.GetVec()
                    del stack[0]
                    '''stack.pop(0)'''
                    vidbid = [v.id, b.id]
                    stack.insert(0, vidbid)
                    e = EDB.NewEdge()
                    e.SetSource(vidbid)
                    vec = b.GetVec()
                    print("vec", vec)
                else:
                    # backward -> backward
                    print("BACK,BACK")
                    """if len(stack) == 0:
                        break"""

                    del stack[0]
                    if len(stack) == 0:
                        break
                    else:
                        b = v.SearchBranchByOtherEnd(stack[0])
                        vec = b.GetVec()

                    print("vec", vec)

            ##############################################

    print('Done')

    #############################################
    # You must program "save to file routines" of
    # vertices (and branches) and edges here.
    # (頂点とエッジを保存するプログラムを追加する)

    fp = open("VDBEDB", "w")

    VDB.Dump(fp)
    EDB.Dump(fp)

    fp.close()
    #############################################
    tt.mainloop()  # wait for user to close the window
