# Python implementation of the Eshelby ellipsoidal inclusion code presented by Meng et al. (2012)
# "Evaluation of the Eshelby solution for the ellipsoidal inclusion and heterogeneity", Computers & Geosciences 40, 40-48.

import numpy as np
from copy import copy
from numpy.linalg import inv
from matplotlib import pyplot as plt
from multiprocessing import Pool
from time import time
from scipy.special import ellipkinc, ellipeinc

# classes
class Inclusion(object):
    def __init__(self):
        # stress/strain ordering is: _xx, _xy, _xz, _yy, _yz, _zz
        self.vm = 0.25
        self.Em = 2.2e10
        self.vh = 0
        self.Eh = 0
        self.dim = [1, .2, 20]
        self.ang = np.array([0, 0, 0])
        self.stressvec = [-1.e6, 0, 0, 0, 1.e6, 0]
        self.eigp = [0, 0, 0, 0, 0, 0]
        self.out = InclusionOutput()
        self.x = np.array([0.])
        self.y = np.array([0.])
        self.z = np.array([0.])
    def rotate_stress(self):
        ''' rotate the stress vector into ellipsoid coordinates '''
        self.stressvec = rotate(self.stressvec, self.R, self.R_i) 
    def rotate_eigp(self):
        ''' rotate the eigenstrain vector into ellipsoid coordinates '''
        self.eigp = rotate(self.eigp, self.R, self.R_i) 
    def compute_applied_strain(self):
        ''' compute the applied strain '''
        self.epsvec=np.dot(inv(self.Cm),self.stressvec)
    def compute_internal_eshelby(self):
        ''' compute the internal eshelby tensor
        '''
        #******************************************************************#
        #Calculation of I's
        #******************************************************************#
        a = self.dim
        
        if any(np.array(a)<0):
            raise ValueError('Ellipsoid dimensions (a) must be positive')
        
        if (abs(a[0]-a[1])<(1e-6*a[0])) and (abs(a[1]-a[2])<(1e-6*a[0])):  # checks that geometric mean of ellipsoid dimensions is not more than 1e-6 different from first dimension
            # Spherical Case
            Ifir=np.ones(3)
            Isec = np.ones((3,3))
            Ifir=(4/3)*np.pi*Ifir
            Isec=(4/5)*np.pi*a[0]**2*Isec
            
        elif ((a[0]-a[1])>(1e-6*a[0])) and (abs(a[2]-a[1])<(1e-6*a[0])):
            # Prolate Spheriod Case
            rat=a[0]/a[2]	
            
            Ifir=np.zeros(3)
            Ifir[1]=(2*np.pi*a[0]*a[2]**2/((a[0]**2-a[2]**2)**(3/2)))*(rat*np.sqrt(rat**2-1)-np.arccosh(rat))
            Ifir[2]=Ifir[1]
            Ifir[0]=4*np.pi-2*Ifir[1]
            
            Isec = np.zeros((3,3))
            Isec[0,1]=(Ifir[1]-Ifir[0])/(a[0]**2-a[1]**2)
            Isec[0,2]=Isec[0,1]
            Isec[1,0]=Isec[0,1]
            Isec[2,0]=Isec[0,2]
            Isec[0,0]=(4*np.pi/a[0]**2-2*Isec[0,1])/3
            Isec[1,2]=np.pi/(a[1]**2)-(Ifir[1]-Ifir[0])/(4*(a[0]**2-a[1]**2))
            Isec[2,1]=Isec[1,2]
            Isec[1,1]=Isec[1,2]
            Isec[2,2]=Isec[1,2]

        elif abs(a[0]-a[1])<(1e-6*a[0]) and (a[1]-a[2])>(1e-6*a[1]):
            # Oblate Spheriod Case
            rat=a[2]/a[0]	
            
            Ifir=np.zeros(3)
            Ifir[0]=(2*np.pi*a[0]**2*a[2]/((a[0]**2-a[2]**2)**(3/2)))*(np.arccos(rat)-rat*np.sqrt(1-rat**2))
            Ifir[1]=Ifir[0]
            Ifir[2]=4*np.pi-2*Ifir[0]
            
            Isec = np.ones((3,3))
            Isec[0,2]=(Ifir[0]-Ifir[2])/(a[2]**2-a[0]**2)
            Isec[2,0]=Isec[0,2]
            Isec[1,2]=Isec[0,2]
            Isec[2,1]=Isec[1,2]
            Isec[0,1]=np.pi/a[0]**2-Isec[0,2]/4
            Isec[1,0]=Isec[0,1]
            Isec[0,0]=Isec[0,1]
            Isec[1,1]=Isec[0,1]
            Isec[2,2]=(4*np.pi/a[2]**2-2*Isec[0,2])/3
            
        else:
            # Triaxial Ellipsoid Case    
            theta=np.arcsin(np.sqrt(1-(a[2]/a[0])**2)) # amplitude
            m = np.sqrt((a[0]**2-a[1]**2)/(a[0]**2-a[2]**2)) # m=k**2 is the parameter
            F = ellipkinc(theta, m)
            E = ellipeinc(theta, m)
            # Mura 11.17
            Ifir=np.zeros(3)
            Ifir[0]=(4*np.pi*np.prod(a)/((a[0]**2-a[1]**2)*np.sqrt(a[0]**2-a[2]**2)))*(F-E)
            Ifir[2]=(4*np.pi*np.prod(a)/((a[1]**2-a[2]**2)*np.sqrt((a[0]**2-a[2]**2))))*(a[1]*np.sqrt((a[0]**2-a[2]**2))/(a[0]*a[2])-E)
            Ifir[1]=4*np.pi-Ifir[0]-Ifir[2]
            
            Isec = np.ones((3,3))
            Isec[0,1]=(Ifir[1]-Ifir[0])/(a[0]**2-a[1]**2)
            Isec[1,2]=(Ifir[2]-Ifir[1])/(a[1]**2-a[2]**2)
            Isec[2,0]=(Ifir[0]-Ifir[2])/(a[2]**2-a[0]**2)
            Isec[1,0]=Isec[0,1]
            Isec[2,1]=Isec[1,2]
            Isec[0,2]=Isec[2,0]
            Isec[0,0]=(4*np.pi/a[0]**2-Isec[0,1]-Isec[0,2])/3
            Isec[1,1]=(4*np.pi/a[1]**2-Isec[1,2]-Isec[1,0])/3
            Isec[2,2]=(4*np.pi/a[2]**2-Isec[2,0]-Isec[2,1])/3    
        
        denom=8*np.pi*(1-self.vm)

        S1111=(3*a[0]**2*Isec[0,0]+(1-2*self.vm)*Ifir[0])/denom
        S2222=(3*a[1]**2*Isec[1,1]+(1-2*self.vm)*Ifir[1])/denom
        S3333=(3*a[2]**2*Isec[2,2]+(1-2*self.vm)*Ifir[2])/denom

        S1122=(a[1]**2*Isec[0,1]-(1-2*self.vm)*Ifir[0])/denom
        S2233=(a[2]**2*Isec[1,2]-(1-2*self.vm)*Ifir[1])/denom
        S3311=(a[0]**2*Isec[2,0]-(1-2*self.vm)*Ifir[2])/denom

        S1133=(a[2]**2*Isec[0,2]-(1-2*self.vm)*Ifir[0])/denom
        S2211=(a[0]**2*Isec[1,0]-(1-2*self.vm)*Ifir[1])/denom
        S3322=(a[1]**2*Isec[2,1]-(1-2*self.vm)*Ifir[2])/denom

        S1212=((a[0]**2+a[1]**2)*Isec[0,1]+(1-2*self.vm)*(Ifir[0]+Ifir[1]))/(2*denom)
        S2323=((a[1]**2+a[2]**2)*Isec[1,2]+(1-2*self.vm)*(Ifir[1]+Ifir[2]))/(2*denom)
        S3131=((a[2]**2+a[0]**2)*Isec[2,0]+(1-2*self.vm)*(Ifir[2]+Ifir[0]))/(2*denom)
        S1313=S3131

        self.S4=np.array([
            [S1111,  0,      0,      S1122,  0,      S1133],
            [0,      2*S1212,  0,      0,      0,      0],
            [0,      0,      2*S1313,  0,      0,      0],
            [S2211,  0,      0,      S2222,  0,      S2233],
            [0,      0,      0,      0,      2*S2323,  0],
            [S3311,  0,      0,      S3322,  0,      S3333]]
        )
    def setup_grid(self):
        
        for att in ['x', 'y', 'z']:
            v = self.__getattribute__(att)
            if not isiterable(v):
                self.__setattr__(att, np.array([v,]))

        self.out.X, self.out.Y, self.out.Z = np.meshgrid(self.x,self.y,self.z, indexing='ij')
        self.Nx = len(self.x)
        self.Ny = len(self.y)
        self.Nz = len(self.z)
    def solve(self, computeDisp=True, computeStress=True, computeStrain=True, ncpus = 1):
        '''
        '''
        # compute stiffness tensors
        self.Cm = Ctensord(self.Em, self.vm)
        self.Ch = Ctensord(self.Eh, self.vh)

        # arrange ellipsoid axes largest to smallest
        exh = np.zeros((3,3))
        for i in range(2):
            for j in range(1,3):
                if self.dim[i]<self.dim[j]:
                    exh[i,j] = 1
                    tmp = self.dim[i]
                    self.dim[i] = self.dim[j]
                    self.dim[j] = tmp
                    
        # pre-rotation in order of [z,y,x]
        self.ang_i = np.pi/2*np.array([exh[1,2], exh[0,2], exh[0,1]])
        #self.ang_i = np.pi/2*np.array([exh[0,1], exh[0,2], exh[1,2]])
        #self.ang_i = np.array([0,0,0])
        self.R_i = Rmats(self.ang_i)

        # rotation matrices w.r.t the ellipsoid
        self.R = Rmats(self.ang)
        # rotate stress w.r.t ellipsoid
        self.rotate_stress()
        # compute the applied strain due to remote stress (rotated)
        self.compute_applied_strain()
        # rotate the eigen strain w.r.t ellipsoid
        self.rotate_eigp()
        # compute the internal eshelby tensor.
        self.compute_internal_eshelby()
        # compute eigenstrain
        self.eigen=np.dot(inv(np.dot(self.Cm-self.Ch,self.S4)-self.Cm),(-np.dot(self.Cm-self.Ch,self.epsvec)-np.dot(self.Ch,self.eigp)))
        # compute inclusion internal strain and stress
        #self.incstrain = rotate(self.epsvec+istrain, self.R, self.R_i)
        #self.incstress = rotate(self.stressvec+np.dot(self.Cm,(np.dot(self.S4,self.eigen)-self.eigen)), self.R, self.R_i)
        self.incstrain = self.epsvec+np.dot(self.S4,self.eigen)
        self.incstress = self.stressvec+np.dot(self.Cm,(np.dot(self.S4,self.eigen)-self.eigen))

        # setup the simulation grid
        self.setup_grid()
        
        computeD4 = (computeStress or computeStrain)

        if computeD4:
            if computeStrain:
                e = np.zeros((self.Nx,self.Ny,self.Nz,6))
            if computeStress:
                s = np.zeros((self.Nx,self.Ny,self.Nz,6))
        if computeDisp:
            u = np.zeros((self.Nx,self.Ny,self.Nz,3))
        pars = []
        for i in range(self.Nx):
            for j in range(self.Ny):
                for k in range(self.Nz):
                    x = np.array([self.out.X[i,j,k], self.out.Y[i,j,k], self.out.Z[i,j,k]])
                    pars.append([i,j,k,x])

        args = (self.R, self.R_i, self.vm, self.dim, computeDisp, computeD4, computeStress, computeStrain, self.eigen, self.incstrain, self.incstress,self.epsvec,self.stressvec,self.Cm)
        outs = run_all(pars, args, ncpus)
        for par, out in zip(pars,outs):
            i,j,k,x = par
            d,sn,sr = out
            u[i,j,k,:] = d
            s[i,j,k,:] = sr
            e[i,j,k,:] = sn

        output = InclusionOutput()
        output.x = self.out.X
        output.y = self.out.Y
        output.z = self.out.Z
        if computeDisp:
            output.u = dict([(cpt, u[:,:,:,i]) for i,cpt in enumerate(['x','y','z'])])
        if computeStress:
            output.s = dict([(cpt, s[:,:,:,i]) for i,cpt in enumerate(['xx','xy','xz','yy','yz','zz'])])
        if computeStrain:
            output.e = dict([(cpt, e[:,:,:,i]) for i,cpt in enumerate(['xx','xy','xz','yy','yz','zz'])])

        self.sol = output

class InclusionOutput(object):
    def __i__(self):
        self.x = None
        self.y = None
        self.z = None
        self.disp = None
        self.stress = None
        self.strain = None

# helper functions
def Rmat(ang):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(ang[0]), -np.sin(ang[0])],
        [0, np.sin(ang[0]), np.cos(ang[0])]
        ])
    Ry = np.array([
        [np.cos(ang[1]), 0, np.sin(ang[1])],
        [0, 1, 0,],
        [-np.sin(ang[1]), 0, np.cos(ang[1])]
        ])
    Rz = np.array([
        [np.cos(ang[2]), -np.sin(ang[2]), 0],
        [np.sin(ang[2]), np.cos(ang[2]), 0],
        [0, 0, 1]])
    return Rx, Ry, Rz
def Rmats(ang):
    Rx,Ry,Rz = Rmat(ang)
    R = np.dot(np.dot(Rx,Ry),Rz)
    return R#,Rb
def vec2mat(v):
    ''' convert 6-element vector to 3x3 matrix'''
    v = np.array(v)
    return np.array([v[:3], v[[1, 3, 4]], v[[2, 4, 5]]])
def mat2vec(m):
    ''' convert 3x3 matrix to 6-element vector'''
    return np.array([m[0,0], m[0,1], m[0,2], m[1,-2], m[1,-1], m[-1,-1]])
def rotate(vec, R1, R2):
    ''' rotate a vector according to R and Rbi'''
    mat = vec2mat(vec)
    #rmat = np.dot(np.dot(np.dot(R,Rbi),mat),np.dot(R,Rbi))
    rmat = np.dot(np.dot(np.dot(R2,R1),mat),np.dot(R1.T,R2.T))
    #rmat = np.dot(np.dot(np.dot(R,Rbi),mat),np.dot(Rbi.T,R.T))
    return mat2vec(rmat)
def iterable(a):
    try:
        [_ for _ in a]
    except TypeError:
        a = np.array([[a,],])
    return a
def buildtensors(a):
    """
    builds tensors of up to rank 2 for elementwise multiplication to avoid
    nested for loop evaluations

    Output naming convention is inputvector_## where the first # is the
    tensor order and the second # is the coordinate direction in which the
    elements of the input vector are advanced (i.e. in which the elements are
    unique)
    """
    A_11 = np.array([a,])
    A_21 = np.concatenate((A_11.T,A_11.T,A_11.T),1)


    return A_21, A_21.T
def Ctensord(Em,vm):

    Gm = Em/(2+2*vm)
    lamem = 2*Gm*vm/(1-2*vm)
    q = np.zeros((6,6))

    q = np.array([
        [lamem+2*Gm, 0, 0, lamem, 0, lamem],
        [0,2*Gm,0,0,0,0],
        [0,0,2*Gm,0,0,0],
        [lamem,0,0,lamem+2*Gm,0,lamem],
        [0,0,0,0,2*Gm,0],
        [lamem,0,0,lamem,0,lamem+2*Gm]
        ])

    return q
def Cmatrix(Cm):
    """this function converts the 4th order isotropic stiffness tensor into 6x6 matrix"""
    matr = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            m,n = index6(i)
            p,q = index6(j)

            if j in [1,2,4]:
                matr[i,j]=Cm[m,n,p,q]+Cm[m,n,q,p]
            else:
                matr[i,j]=Cm[m,n,p,q]
    
    return matr
def kdelta(i,j):
    """ returns the Kroneker Delta of two variables """
    if i==j:
        q = 1
    else:
        q = 0
    return q
def index6(i):
    """ converts from a vector index to a tensor index"""
    return [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)][i]
def run_all(pars, args, ncpus):
    if ncpus != 1:
        p = Pool(ncpus)
        outs = p.map(run_one, zip([par[-1] for par in pars], [args for i in range(len(pars))]))
    else:
        outs = [run_one([par[-1], args]) for par in pars]
    
    return outs
def run_one(inps):
    x,args = inps
    R, R_i, vm, dim, computeDisp, computeD4, computeStress, computeStrain, eigen, incstrain, incstress,epsvec,stressvec,Cm = args

    pos = np.dot(np.dot(R_i,R),x)
    out = Esh(vm, dim, pos, eigen, computeDisp, computeD4)
    d = np.zeros(3)
    sn = np.zeros(6)
    sr = np.zeros(6)
    
    if computeDisp and computeD4:
        rd4=Cmatrix(out[0])
        d = np.dot(np.dot(R.T,R_i.T),out[1])
    elif computeDisp:
        rd4=Cmatrix(out)
    elif computeD4:
        d = np.dot(np.dot(R.T,R_i.T),out)

    if computeD4:
        if pos[0]**2/dim[0]**2+pos[1]**2/dim[1]**2+pos[2]**2/dim[2]**2 <= 1: # for interior points
            if computeStrain:
                sn = rotate(incstrain, R_i.T, R.T)
            if computeStress:
                sr = rotate(incstress, R_i.T, R.T)
        else:
            if computeStrain:
                strainr = epsvec+np.dot(np.squeeze(rd4),eigen)
                sn = rotate(strainr, R_i.T, R.T)
            if computeStress:
                stressr = stressvec+np.dot(np.dot(Cm,np.squeeze(rd4)),eigen)
                sr = rotate(stressr, R_i.T, R.T)

    return d, sn, sr
def Esh(vm, a, x, eigen, computeDisp=False, computeD4=False):
    """
    todo search for todos in function
    are the case statements supposed to be exact or with a tolerance like in eshint
    get rid of all vars not used

    ******************************************************************#
    Calculation of F and E integrals
    ******************************************************************#

    this subroutines finds the largest positive root of
    x[0]**2/(a[0]+lmb) + x[1]**2/(a[1]+lmb) + x[2]**2/(a[2]+lmb) = 1
    (Mura 11.37) for the exterior point x and elliopsoid dimensions a.  When 
    expanded and like terms in lmb are collected, the coefficients of 
    lmb**3, **2, etc. are as below
    """
    # precondition
    assert (computeDisp or computeD4)
    a = np.array(a)
    
    coef3 = 1 # coefficient of lambds**3 term
    coef2 = a[0]**2+a[1]**2+a[2]**2-(x[0]**2+x[1]**2+x[2]**2) # coefficient of lambds**2 term
    coef1 = a[0]**2*a[1]**2+a[0]**2*a[2]**2+a[1]**2*a[2]**2-((a[1]**2+a[2]**2)*x[0]**2+(a[0]**2+a[2]**2)*x[1]**2+(a[0]**2+a[1]**2)*x[2]**2) # coefficient of lambds term
    coef0 = a[0]**2*a[1]**2*a[2]**2-(a[1]**2*a[2]**2*x[0]**2+a[0]**2*a[2]**2*x[1]**2+a[0]**2*a[1]**2*x[2]**2) # coefficient of constant term
    poly = [coef3, coef2, coef1, coef0] # matlab polynomial format
    lmb = 0 # initialize lmb to zero

    if (x[0]**2/a[0]**2+x[1]**2/a[1]**2+x[2]**2/a[2]**2) > 1: # if x is exterior point set
        # lmb to the largest positive real root, otherwise lmb=0
        lmbroots = np.roots(poly) # store the roots of the cubic equation
        for i in range(3): # find the largest positive real root
            if np.isreal(lmbroots[i]) and lmbroots[i]>lmb:
                lmb = lmbroots[i]

    theta = np.arcsin(np.sqrt((a[0]**2-a[2]**2)/(a[0]**2+lmb))) # the amplitude
    
    # todo this argument was taken from the previous code (with the lmb) and
    # modified with the arcsin.  need to see if can get here via Gradshteyn and
    # Ryzhik from Mura 11.36
    m = np.sqrt((a[0]**2-a[1]**2)/(a[0]**2-a[2]**2)) 
    F = ellipkinc(theta, m)
    E = ellipeinc(theta, m)
    
    #******************************************************************#
    #Calculation of I's
    #******************************************************************#

    Ifir = np.zeros(3)
    Isec = np.zeros((3,3))
    if a[0]==a[1] and a[0]==a[2]:
        # Spherical Case
        #print('Spherical case..')
        delta = np.sqrt(np.prod(a**2+lmb))
        # can simplify to del3=sqrt((a[0]**2+lmb)**3) for sphere
        Ifir=(4/3)*np.pi*a[0]**3/(np.sqrt(a[0]**2+lmb))**3*np.ones(3)
        Isec=(4/5)*np.pi*a[0]**3/np.sqrt(a[0]**2+lmb)*np.ones((3,3))  # todo: i changed the 5/2 to 1/2 to make units right--not sure if correct

    elif a[0]>a[1] and a[2]==a[1]:
        #print('Prolate case..')
        
        delta=np.sqrt((a[0]**2+lmb)*(a[1]**2+lmb)*(a[2]**2+lmb))
        bbar=np.sqrt(a[0]**2+lmb)/np.sqrt(a[2]**2+lmb)
        dbar=np.sqrt(a[0]**2-a[2]**2)/np.sqrt(a[2]**2+lmb)
        I=(np.arccosh(bbar))*4*np.pi*a[0]*a[1]**2/np.sqrt(a[0]**2-a[1]**2)
        Ifir[0]=4*np.pi*a[0]*a[1]**2*(np.arccosh(bbar)-dbar/bbar)/(np.sqrt(a[0]**2-a[1]**2))**3
        Ifir[1]=2*np.pi*a[0]*a[1]**2*(-np.arccosh(bbar)+dbar*bbar)/(np.sqrt(a[0]**2-a[1]**2))**3
        Ifir[2]=Ifir[1]

        Isec[0,1]=(Ifir[1]-Ifir[0])/(a[0]**2-a[1]**2)
        Isec[0,2]=Isec[0,1]
        Isec[1,0]=Isec[0,1]
        Isec[2,0]=Isec[0,2]
        Isec[1,2]=np.pi*np.prod(a)/((a[2]**2+lmb)*delta)-Isec[0,2]/4
        Isec[2,1]=Isec[1,2]
        Isec[0,0]=((4*np.pi*np.prod(a))/((a[0]**2+lmb)*delta)-Isec[0,1]-Isec[0,2])/3
        Isec[1,1]=Isec[1,2]
        Isec[2,2]=Isec[1,2]
        
    elif a[0]==a[1]and a[1]>a[2]:
        #print('Oblate case...')
        delta=np.sqrt((a[0]**2+lmb)*(a[1]**2+lmb)*(a[2]**2+lmb))
        bnonbar=np.sqrt(a[2]**2+lmb)/np.sqrt(a[0]**2+lmb)
        dnonbar=np.sqrt(a[0]**2-a[2]**2)/np.sqrt(a[0]**2+lmb)
        I=(np.arccos(bnonbar))*4*np.pi*a[0]**2*a[2]/np.sqrt(a[0]**2-a[2]**2)
        Ifir[0]=2*np.pi*a[0]**2*a[2]*(np.arccos(bnonbar)-dnonbar*bnonbar)/(a[0]**2-a[2]**2)**1.5
        Ifir[1]=Ifir[0]
        Ifir[2]=4*np.pi*np.prod(a)/delta-2*Ifir[0]
        
        Isec[0,2]=(Ifir[2]-Ifir[0])/(a[0]**2-a[2]**2)
        Isec[2,0]=Isec[0,2]
        Isec[1,2]=Isec[0,2]
        Isec[2,1]=Isec[1,2]
        
        Isec[0,0]=np.pi*np.prod(a)/((a[0]**2+lmb)*delta)-Isec[0,2]/4
        Isec[0,1]=Isec[0,0]
        Isec[1,0]=Isec[0,1]
        
        Isec[1,1]=Isec[0,0]
        Isec[2,2]=((4*np.pi*np.prod(a))/((a[2]**2+lmb)*delta)-Isec[0,2]-Isec[1,2])/3
    else:
        #print('triaxial ellipsoid case ..')
        delta=np.sqrt((a[0]**2+lmb)*(a[1]**2+lmb)*(a[2]**2+lmb))
        I=4*np.pi*np.prod(a)*F/np.sqrt(a[0]**2-a[2]**2)
        Ifir[0]=I*(1-E/F)/(a[0]**2-a[1]**2)
        
        Ifir[1]=4*np.pi*np.prod(a)*(E*np.sqrt(a[0]**2-a[2]**2)/((a[0]**2-a[1]**2)*(a[1]**2-a[2]**2))-F/((a[0]**2-a[1]**2)*np.sqrt(a[0]**2-a[2]**2))-(1/(a[1]**2-a[2]**2))*np.sqrt((a[2]**2+lmb)/((a[0]**2+lmb)*(a[1]**2+lmb))))
        
        Ifir[2]=4*np.pi*np.prod(a)/delta-Ifir[0]-Ifir[1]
        
        Isec[0,1]=(Ifir[1]-Ifir[0])/(a[0]**2-a[1]**2)
        Isec[1,0]=Isec[0,1]
        Isec[0,2]=(Ifir[2]-Ifir[0])/(a[0]**2-a[2]**2)
        Isec[2,0]=Isec[0,2]
        Isec[1,2]=(Ifir[2]-Ifir[1])/(a[1]**2-a[2]**2)
        Isec[2,1]=Isec[1,2]
        Isec[0,0]=((4*np.pi*np.prod(a))/((a[0]**2+lmb)*delta)-Isec[0,1]-Isec[0,2])/3
        Isec[1,1]=((4*np.pi*np.prod(a))/((a[1]**2+lmb)*delta)-Isec[0,1]-Isec[1,2])/3
        Isec[2,2]=((4*np.pi*np.prod(a))/((a[2]**2+lmb)*delta)-Isec[0,2]-Isec[1,2])/3

    #*************************************************************************************************
    #I derivatives
    #*************************************************************************************************

    a_21, a_22 = buildtensors(a)
    ultadelfir = -2*np.pi*np.prod(a)/((a**2+lmb)*delta)
    ultadelfir_21, ultadelfir_22 = buildtensors(ultadelfir)
    ultadelsec = -2*np.pi*np.prod(a)/((a_21**2+lmb)*(a_22**2+lmb)*delta)

    # derivatives of lmb
    c1 = np.sum((x**2)/((a**2+lmb)**2))
    c2 = np.sum((x**2)/((a**2+lmb)**3))
    c3 = np.sum((x**2)/((a**2+lmb)**4))
        
    F = 2*x/(a**2+lmb) 
    if computeD4: 
        F_21, F_22 = buildtensors(F)

    if lmb == 0:
        fderlmb = np.zeros(3)
    else:
        fderlmb = F/c1
    
    fderlmb_21, fderlmb_22 = buildtensors(fderlmb)

    if computeD4:
        diagvals = np.eye(3)
        nondiagvals = np.ones((3,3))-np.eye(3)
        fderF = nondiagvals*(1/(a_21**2+lmb))*(-F_21*fderlmb_22)+diagvals*(1/(a_21**2+lmb))*(2-F_21*fderlmb_22)
        fderc1 = F/(a**2+lmb)-2*c2*fderlmb
        fderc1_21, fderc1_22 = buildtensors(fderc1)
        fderc2 = F/(a**2+lmb)**2-3*c3*fderlmb
        fderc2_21, fderc2_22 = buildtensors(fderc2)

        if lmb == 0:
            sderlmb = np.zeros((3,3))
        else:
            sderlmb = (fderF-fderlmb_21*fderc1_22)/c1
        
        sderc1 = (1/(a_21**2+lmb))*(fderF-fderlmb_22*F_21/(a_21**2+lmb))-2*(fderc2_22*fderlmb_21+c2*sderlmb)
    fderIfir = ultadelfir_21*fderlmb_22
    
    if computeD4:
        sderF  = np.zeros((3,3,3))
        for q in range(3):
            for p in range(3):
                for r in range(3):
                    sderF[q,p,r] = -(fderF[q,p]*fderlmb[r]+fderF[q,r]*fderlmb[p]+F[q]*sderlmb[p,r])/(a[q]**2+lmb)
    
        zeefir = 1/(a**2+lmb)+0.5*np.sum(1/(a**2+lmb))
        zeesec = 1/(a_21**2+lmb)+1/(a_22**2+lmb)+0.5*np.sum(1/(a**2+lmb))
        
        sderIfir  = np.zeros((3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    sderIfir[i,j,k] = ultadelfir[i]*(sderlmb[j,k]-fderlmb[j]*fderlmb[k]*zeefir[i])

    fderIsec  = np.zeros((3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                fderIsec[i,j,k] = ultadelsec[i,j]*fderlmb[k]

    if computeD4:
        sderIsec  = np.zeros((3,3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        sderIsec[i,j,k,l] = ultadelsec[i,j]*(sderlmb[k,l]-fderlmb[k]*fderlmb[l]*zeesec[i,j])

        tderlmb  = np.zeros((3,3,3))
        for q in range(3):
            for p in range(3):
                for r in range(3):
                    if lmb == 0:
                        tderlmb[q,p,r] = 0
                    else:
                        tderlmb[q,p,r] = (-1/c1)*(sderlmb[q,p]*fderc1[r]-sderF[q,p,r]+sderlmb[q,r]*fderc1[p]+fderlmb[q]*sderc1[p,r])
    
        #************************************************
        #Calculation of V-potentials
        #***********************************************

        sderVfir  = np.zeros((3,3,3))
        for i in range(3):
            for p in range(3):
                for q in range(3): 
                    sderVfir[i,p,q] = -(kdelta(p,q)*Isec[p,i]+x[p]*fderIsec[p,i,q])
        
        tderVfir  = np.zeros((3,3,3,3))
        for i in range(3):
            for p in range(3):
                for q in range(3): 
                    for r in range(3):
                        tderVfir[i,p,q,r] = -(kdelta(p,q)*fderIsec[p,i,r]+kdelta(p,r)*fderIsec[p,i,q]+x[p]*sderIsec[p,i,q,r])

    #*********************************************
    #calculation of phi and psi potentials
    #*********************************************

    #calculation of phi derivatives
    if computeDisp:
        fderphi=-x*Ifir

    if computeD4:
        #calculation of phi derivatives
        sderphi  = np.zeros((3,3))
        for p in range(3):
            for q in range(3): 
                sderphi[p,q] = -(kdelta(p,q)*Ifir[p]+x[p]*fderIfir[p,q])

        tderphi  = np.zeros((3,3,3))
        for p in range(3):
            for q in range(3): 
                for r in range(3):
                    tderphi[p,q,r] = -(kdelta(p,q)*fderIfir[p,r]+kdelta(p,r)*fderIfir[p,q]+x[p]*sderIfir[p,q,r])

    #*******************
    #psi's
    #***************

    if computeDisp:
        tderpsi  = np.zeros((3,3,3))
        for i in range(3):
            for j in range(3):
                for l in range(3):            
                    tderpsi[i,j,l]=-kdelta(i,j)*x[l]*(Ifir[l]-a[i]**2*Isec[i,l])-x[i]*x[j]*(fderIfir[j,l]-a[i]**2*fderIsec[i,j,l])-(kdelta(i,l)*x[j]+kdelta(j,l)*x[i])*(Ifir[j]-a[i]**2*Isec[i,j])
      
    if computeD4:
        foderpsi  = np.zeros((3,3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        foderpsi[i,j,k,l]=kdelta(i,j)*(sderphi[k,l]-a[i]**2*sderVfir[i,k,l])+kdelta(i,k)*(sderphi[j,l]-a[i]**2*sderVfir[i,j,l])+kdelta(i,l)*(sderphi[j,k]-a[i]**2*sderVfir[i,j,k])+x[i]*(tderphi[j,k,l]-a[i]**2*tderVfir[i,j,k,l])

    #*******************************************
    #calculation of D4 
    #******************************************
    premult1=1/(8*np.pi*(1-vm))

    #calculation of D4 
    if computeD4:
        D4  = np.zeros((3,3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        D4[i,j,k,l]=premult1*(foderpsi[k,l,i,j]-2*vm*kdelta(k,l)*sderphi[i,j]-(1-vm)*(sderphi[k,j]*kdelta(i,l)+sderphi[k,i]*kdelta(j,l)+sderphi[l,j]*kdelta(i,k)+sderphi[l,i]*kdelta(j,k)))

    #calculate disp
    if computeDisp:
        eigenM = vec2mat(eigen)
        diag = eigenM[0,0]+eigenM[1,1]+eigenM[2,2] 

        u = premult1*(np.tensordot(tderpsi, eigenM, axes=[[1,2],[0,1]])-2*vm*diag*fderphi.T-4*(1-vm)*(np.dot(eigenM,fderphi.T))).T
    
    if computeD4 and computeDisp:
        return D4, u
    elif computeD4:
        return D4
    elif computeDisp:
        return u
def isiterable(inp):
    try:
        [_ for _ in inp]
        return True
    except TypeError:
        return False

# validate against figures in Meng
def meng_fig1():


    ayaxs = [100, 50, 5, 1, 0.5, 0.25, 0.125, 0.01]
    #ayaxs = [1.-1.e-6, 1+1.e-6]

    f,ax = plt.subplots(1,1,figsize=(8,8))
    for ayax in ayaxs:
        inc = Inclusion()
        inc.dim = [2., 2*ayax, 4.]
        #inc.x = np.linspace(2,10,51)
        inc.x = 2.+(np.logspace(-4, 0, 21)-0*2.e-4)*8.
        inc.y = 0.0
        inc.z = 0

        inc.vm = 0.15
        inc.Em = 2.2e10
        inc.vh = 0.15
        inc.Eh = 2.2e10
        inc.stressvec = [0, 0, 0, 0, 0, 0]
        inc.eigp = [1.e-3, 0, 0, 1.e-3, 0, 1.e-3]
            
        inc.solve(ncpus=1)

        x = inc.sol.x.squeeze()
        eyy = inc.sol.e['yy'].squeeze()
        ax.plot(x, np.log10(eyy), 'k-')
        
    ax.set_ylabel('log$_{10}\epsilon_{yy}$')
    ax.set_xlabel('x')
    ax.set_xlim([2,10])
    ax.set_ylim([-7.5, -2.5])
    plt.show()
def meng_fig2():
    

    azays = [0.25, 0.5, 0.6, 1, 2, 4]
    #ayaxs = [1.-1.e-6, 1+1.e-6]
    #azays = [1,1.001]

    f,ax = plt.subplots(1,1,figsize=(8,8))
    cs = ['k','r']
    cs = ['k']*len(azays)
    for azay,c in zip(azays,cs):
        inc = Inclusion()
        inc.dim = [0.1, 1, azay]
        #inc.x = np.linspace(2,10,51)
        inc.x = 0.+(np.logspace(-2, 0, 41)-0*2.e-4)*2.5
        inc.y = 0.4
        inc.z = 0.4

        inc.vm = 0.15
        inc.Em = 2.2e10
        inc.vh = 0.
        inc.Eh = 0.
        inc.stressvec = [1.e6, 0, 0, 0, 0, 0]
        inc.eigp = [0., 0, 0, 0., 0, 0.]
            
        inc.solve(ncpus=1)

        x = inc.sol.x.squeeze()
        sxx = inc.sol.s['xx'].squeeze()
        ax.plot(x, sxx/1.e5, c+'-')
        
    ax.set_ylabel('$\sigma_{xx}$')
    ax.set_xlabel('x')
    ax.set_xlim([0,2.5])
    ax.set_ylim([-2, 14])
    plt.show()
def meng_fig3():
    azays = [10, 15, 20, 30, 60, 1000]
    #ayaxs = [1.-1.e-6, 1+1.e-6]
    #azays = [1,1.001]

    f,ax = plt.subplots(1,1,figsize=(8,8))
    ax2 = ax.twinx()
    cs = ['k','r']
    cs = ['k']*len(azays)
    for azay,c in zip(azays,cs):
        inc = Inclusion()
        inc.dim = [0.2, 1, azay]
        #inc.x = np.linspace(2,10,51)
        inc.x = 0.+np.logspace(-2, 0, 41)*0.5
        inc.x = np.concatenate([inc.x, np.array([100.,])])
        inc.y = 1.1
        inc.z = 5.

        inc.vm = 0.25
        inc.Em = 2.2e10
        inc.vh = 0.
        inc.Eh = 0.
        inc.stressvec = [1.e6, 0, 0, -1.e6, 0, 0]
        inc.eigp = [0., 0, 0, 0., 0, 0.]
            
        inc.solve(ncpus=1)

        x = inc.sol.x.squeeze()
        sxx = inc.sol.s['xx'].squeeze()
        ax.plot(x, np.log10(sxx), c+'-')
        uy = inc.sol.u['y'].squeeze()
        ax2.plot(x, uy*1.e6/np.pi, 'r-')
        #print(uy[-1])  
    ax.set_ylabel('$\sigma_{xx}$')
    ax.set_xlabel('x')
    ax.set_xlim([0,0.5])
    ax.set_ylim([6.15, 6.45])
    ax2.set_ylabel('$u_y-u_y^{\inf}$')
    ax2.set_ylim([-14, -2])
    plt.show()
def meng_fig5():
    

    axays = [0.2,1,2,3,4,5]
    
    f,ax = plt.subplots(1,1,figsize=(8,8))
    cs = ['k','r']
    cs = ['k']*len(axays)
    for axay,c in zip(axays,cs):
        inc = Inclusion()
        inc.dim = [axay, 1, 1.e3]
        #inc.x = np.linspace(2,10,51)
        inc.x = 0.+(np.linspace(0,1, 101)-0*2.e-4)*5
        inc.y = 1
        inc.z = 0.

        inc.vm = 0.25
        inc.Em = 2.2e10
        inc.vh = 0.
        inc.Eh = 0.
        inc.stressvec = [1.e6, 0, 0, -1.e6, 0, 0]
        inc.eigp = [0., 0, 0, 0., 0, 0.]
            
        inc.solve(ncpus=1)

        x = inc.sol.x.squeeze()
        sxx = inc.sol.s['xx'].squeeze()
        ax.plot(x, np.log10(sxx), c+'-')
        
    ax.set_ylabel('log$_{10}\sigma_{xx}$')
    ax.set_xlabel('x')
    ax.set_xlim([0,5])
    ax.set_ylim([5.8, 6.65])
    plt.show()
def meng_fig6():
    

    ayaxs = [0.01, 0.04, 0.08, 0.1]
    
    f,ax = plt.subplots(1,1,figsize=(8,8))
    cs = ['k','r']
    cs = ['k']*len(ayaxs)
    for ayax,c in zip(ayaxs,cs):
        inc = Inclusion()
        inc.dim = [1, ayax, 1.e3]
        #inc.x = np.linspace(2,10,51)
        inc.x = np.linspace(0.99,1.01, 41)
        inc.y = 0.01
        inc.z = 0.

        inc.vm = 0.25
        inc.Em = 2.2e10
        inc.vh = 0.
        inc.Eh = 0.
        inc.stressvec = [0, 0, 0, 1.e6, 0, 0]
        inc.eigp = [0., 0, 0, 0., 0, 0.]
            
        inc.solve(ncpus=1)

        x = inc.sol.x.squeeze()
        syy = inc.sol.s['yy'].squeeze()
        ax.plot(x, np.log10(syy), c+'-')
        
    ax.set_ylabel('log$_{10}\sigma_{yy}$')
    ax.set_xlabel('x')
    ax.set_xlim([0.99,1.01])
    ax.set_ylim([6, 7.1])
    plt.show()
def meng_fig7():
    

    ayaxs = [0.01, 0.04, 0.08, 0.1]
    
    f,ax = plt.subplots(1,1,figsize=(8,8))
    cs = ['k','r']
    cs = ['k']*len(ayaxs)
    for ayax,c in zip(ayaxs,cs):
        inc = Inclusion()
        inc.dim = [1, ayax, 1.e3]
        #inc.x = np.linspace(2,10,51)
        inc.x = np.linspace(0.99,1.01, 41)
        inc.y = 0.01
        inc.z = 0.

        inc.vm = 0.25
        inc.Em = 2.2e10
        inc.vh = 0.
        inc.Eh = 0.
        inc.stressvec = [0, 1.e6, 0, 0., 0, 0]
        inc.eigp = [0., 0, 0, 0., 0, 0.]
            
        inc.solve(ncpus=1)

        x = inc.sol.x.squeeze()
        sxy = inc.sol.s['xy'].squeeze()
        ax.plot(x, np.log10(sxy), c+'-')
        
    ax.set_ylabel('log$_{10}\sigma_{xy}$')
    ax.set_xlabel('x')
    ax.set_xlim([0.99,1.01])
    ax.set_ylim([6, 7.1])
    plt.show()
def meng_fig8():
    ayaxs = [0.01, 0.04, 0.08, 0.1]
    
    f,ax = plt.subplots(1,1,figsize=(8,8))
    cs = ['k','r']
    cs = ['k']*len(ayaxs)
    for ayax,c in zip(ayaxs,cs):
        inc = Inclusion()
        inc.dim = [1, ayax, 1.e3]
        #inc.x = np.linspace(2,10,51)
        inc.x = np.linspace(0.99,1.01, 41)
        inc.y = 0.01
        inc.z = 0.

        inc.vm = 0.25
        inc.Em = 2.2e10
        inc.vh = 0.
        inc.Eh = 0.
        inc.stressvec = [0, 0., 0, 0., 1.e6, 0]
        inc.eigp = [0., 0, 0, 0., 0, 0.]
            
        inc.solve(ncpus=1)

        x = inc.sol.x.squeeze()
        syz = inc.sol.s['yz'].squeeze()
        ax.plot(x, np.log10(syz), c+'-')
        
    ax.set_ylabel('log$_{10}\sigma_{yz}$')
    ax.set_xlabel('x')
    ax.set_xlim([0.99,1.01])
    ax.set_ylim([6, 7.1])
    plt.show()
if __name__ == "__main__":
    #meng_fig1()
    #meng_fig2()
    #meng_fig3()
    #meng_fig5()
    #meng_fig6()
    #meng_fig7()
    meng_fig8()
    
    '''
    f,ax = plt.subplots(1,1,figsize=(8,8))
    x = inc.sol.x.squeeze()
    y = inc.sol.y.squeeze()
    z = inc.sol.z.squeeze()
    u = inc.sol.disp.squeeze()[:,:,0]
    umin = u.min()
    umax = u.max()
    cax = ax.contourf(y,z,u, levels = np.linspace(umin, umax, 11))
    plt.colorbar(cax)
    plt.show()
    '''