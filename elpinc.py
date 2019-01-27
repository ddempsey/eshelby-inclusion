# Python implementation of the Eshelby ellipsoidal inclusion code presented by Meng et al. (2012)
# "Evaluation of the Eshelby solution for the ellipsoidal inclusion and heterogeneity", Computers & Geosciences 40, 40-48.

import numpy as np
from copy import copy
from numpy.linalg import inv
from matplotlib import pyplot as plt
from multiprocessing import Pool
from time import time

# classes
class Inclusion(object):
    def __init__(self):
        self.vm = 0.25
        self.Em = 2.2e10
        self.vh = 0
        self.Eh = 0
        self.dim = [1, .2, 20]
        self.ang = [0, 0, 0]
        self.stressvec = [-1.e6, 0, 0, 0, 1.e6, 0]
        self.eigp = [0, 0, 0, 0, 0, 0]
        self.out = InclusionOutput()
    def rotate_stress(self):
        ''' rotate the stress vector into ellipsoid coordinates '''
        self.stressvec = rotate(self.stressvec, self.R_i, self.Rb) 
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
        
        Ifir=np.ones(3)
        Isec = np.ones((3,3))
        if (abs(a[0]-a[1])<(1e-6*a[0])) and (abs(a[1]-a[2])<(1e-6*a[0])):  # checks that geometric mean of ellipsoid dimensions is not more than 1e-6 different from first dimension
            # Spherical Case
            Ifir=(4/3)*np.pi*Ifir
            Isec=(4/5)*np.pi*a[0]**2*Isec
            
        elif ((a[0]-a[1])>(1e-6*a[0])) and (abs(a[2]-a[1])<(1e-6*a[0])):
            # Prolate Spheriod Case
            rat=a[0]/a[2]	
            
            Ifir[1]=(2*np.pi*a[0]*a[2]**2/((a[0]**2-a[2]**2)**(3/2)))*(rat*np.sqrt(rat**2-1)-np.arccosh(rat))
            Ifir[2]=Ifir[1]
            Ifir[0]=4*np.pi-2*Ifir[1]
            
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
            
            Ifir[0]=(2*np.pi*a[0]**2*a[2]/((a[0]**2-a[2]**2)**(3/2)))*(np.arccos(rat)-rat*np.sqrt(1-rat**2))
            Ifir[1]=Ifir[0]
            Ifir[2]=4*np.pi-2*Ifir[0]
            
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
            # k=((a[0]**2-a[1]**2)/(a[0]**2-a[2]**2))**(1/2) # the elliptic modulus
            m=(a[0]**2-a[1]**2)/(a[0]**2-a[2]**2) # m=k**2 is the parameter
            F,E,Z = elliptic12(theta, m) #this sets the tolerance to eps, add a third argument to set to a larger tol
            # Mura 11.17
            Ifir[0]=(4*np.pi*np.prod(a)/((a[0]**2-a[1]**2)*np.sqrt(a[0]**2-a[2]**2)))*(F-E)
            Ifir[2]=(4*np.pi*np.prod(a)/((a[1]**2-a[2]**2)*np.sqrt((a[0]**2-a[2]**2))))*(a[1]*np.sqrt((a[0]**2-a[2]**2))/(a[0]*a[2])-E)
            Ifir[1]=4*np.pi-Ifir[0]-Ifir[2]
            
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
        self.out.X, self.out.Y, self.out.Z = np.meshgrid(self.x,self.y,self.z, indexing='ij')
        self.Nx = len(self.x)
        self.Ny = len(self.y)
        self.Nz = len(self.z)
        
    def solve(self, computeDisp=False, computeStress=False, computeStrain=False, ncpus = 1):
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
        self.R_i,self.Rb_i = Rmats(self.ang_i)

        # rotation matrices w.r.t the ellipsoid
        self.R,self.Rb = Rmats(self.ang)
        # rotate stress w.r.t ellipsoid
        self.rotate_stress()
        # compute the applied strain
        self.compute_applied_strain()
        # compute the internal eshelby tensor.
        self.compute_internal_eshelby()
        # compute eigenstrain
        self.eigen=np.dot(inv(np.dot(self.Cm-self.Ch,self.S4)-self.Cm),(-np.dot(self.Cm-self.Ch,self.epsvec)-np.dot(self.Ch,self.eigp)))
        # compute inclusion internal strain and stress
        self.incstrain = rotate(self.epsvec+np.dot(self.S4,self.eigen), self.R, self.Rb_i)
        self.incstress = rotate(self.stressvec+np.dot(self.Cm,(np.dot(self.S4,self.eigen)-self.eigen)), self.R, self.Rb_i)

        # setup the simulation grid
        self.setup_grid()
        
        computeD4 = (computeStress or computeStrain)

        if computeD4:
            if computeStrain:
                strain = np.zeros((Nx,Ny,Nz,6))
            if computeStress:
                stress = np.zeros((Nx,Ny,Nz,6))
        if computeDisp:
            disp = np.zeros((Nx,Ny,Nz,3))
        pars = []
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    x = np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]])
                    pars.append([i,j,k,x])

        args = (R, Rb, R_i, Rb_i, vm, dim, eigen, computeDisp, computeD4, computeStress, computeStrain, eigen, incstrain, incstress,epsvec,stressvec,Cm)
        outs = run_all(pars, args, ncpus)
        for par, out in zip(pars,outs):
            i,j,k,x = par
            d,sn,sr = out
            disp[i,j,k,:] = d
            stress[i,j,k,:] = sr
            strain[i,j,k,:] = sn

        output = InclusionOutput()
        output.x = X
        output.y = Y
        output.z = Z
        if computeDisp:
            output.disp = disp
        if computeStress:
            output.stress = stress
        if computeStrain:
            output.strain = strain   

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
    Rx,Ry,Rz = Rmat(-ang)
    Rb = np.dot(np.dot(Rz,Ry),Rx)
    return R,Rb
def vec2mat(v):
    ''' convert 6-element vector to 3x3 matrix'''
    v = np.array(v)
    return np.array([v[:3], v[[1, 3, 4]], v[[2, 4, 5]]])
def mat2vec(m):
    ''' convert 3x3 matrix to 6-element vector'''
    return np.array([m[0,0], m[0,1], m[0,2], m[1,-2], m[1,-1], m[-1,-1]])
def rotate(vec, R, Rbi):
    ''' rotate a vector according to R and Rbi'''
    mat = vec2mat(vec)
    rmat = np.dot(np.dot(np.dot(R,Rbi),mat),np.dot(R,Rbi))
    return mat2vec(rmat)
def elliptic12(u, m, tol = None):
    """
    ELLIPTIC12 evaluates the value of the Incomplete Elliptic Integrals 
    of the First, Second Kind and Jacobi's Zeta Function.
    
      [F,E,Z] = ELLIPTIC12(U,M,TOL) where U is a phase in radians, 0<M<1 is 
      the module and TOL is the tolerance (optional). Default value for 
      the tolerance is eps = 2.220e-16.
    
      ELLIPTIC12 uses the method of the Arithmetic-Geometric Mean 
      and Descending Landen Transformation described in [1] Ch. 17.6,
      to determine the value of the Incomplete Elliptic Integrals 
      of the First, Second Kind and Jacobi's Zeta Function [1], [2].
    
          F(phi,m) = int(1/sqrt(1-m*sin(t)**2), t=0..phi)
          E(phi,m) = int(sqrt(1-m*sin(t)**2), t=0..phi)
          Z(phi,m) = E(u,m) - E(m)/K(m)*F(phi,m).
    
      Tables generating code ([1], pp. 613-621):
          [phi,alpha] = meshgrid(0:5:90, 0:2:90)                  modulus and phase in degrees
          [F,E,Z] = elliptic12(pi/180*phi, sin(pi/180*alpha)**2)  values of integrals
    
      See also ELLIPKE, ELLIPJ, ELLIPTIC3, THETA, AGM.
    
      References:
      [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions", 
          Dover Publications", 1965, Ch. 17.1 - 17.6 (by L.M. Milne-Thomson).
      [2] D. F. Lawden, "Elliptic Functions and Applications"
          Springer-Verlag, vol. 80, 1989

      For support, please reply to 
          moiseev[at]sissa.it, moiseev.igor[at]gmail.com
          Moiseev Igor, 
          34106, SISSA, via Beirut n. 2-4,  Trieste, Italy
    
      The code is optimized for ordered inputs produced by the functions 
      meshgrid, ndgrid. To obtain maximum performace (up to 30) for singleton, 
      1-dimensional and random arrays remark call of the function unique(.) 
      and edit further code. 
    """

    if tol is None:
        tol = np.finfo(float).eps
    
    if not np.isreal(u) or not np.isreal(m):
       raise TypeError('Input arguments must be real.')
    
    u = iterable(u)
    m = iterable(m)

    if len(m)==1:
        m = m*np.ones(np.shape(u))
        
    if len(u)==1:
        u = u*np.ones(np.shape(m))
        
    if not (m.shape[0] == u.shape[0]) or not (m.shape[1] == u.shape[1]):
        raise TypeError('U and M must be the same size.')

    F = np.zeros(np.shape(u)) 
    E = copy(F)              
    Z = copy(E)
    m = m.flatten()    #make a row vector
    u = u.flatten()

    if any(m < 0) or any(m > 1):
        raise TypeError('M must be in the range 0 <= M <= 1.')

    I = np.where((m != 1)&(m != 0))[0]
    if len(I) != 0:
        mu,J,K = np.unique(m[I], return_index=True, return_inverse=True)   #extracts unique values from m
        #K = uint32(K)
        mumax = len(mu)
        signU = np.sign(u[I])

        #pre-allocate space and augment if needed
        chunk = 7
        a = np.zeros((chunk,mumax))
        c = copy(a) 
        b = copy(a)
        a[0,:] = np.ones(mumax)
        c[0,:] = np.sqrt(mu)
        b[0,:] = np.sqrt(1-mu)
        n = np.zeros((1,mumax), dtype=np.int32)
        i = 0
        while any(abs(c[i,:]) > tol):                                    #Arithmetic-Geometric Mean of A, B and C
            i = i + 1
            if i >= np.shape(a)[0]:
                a = np.concatenate((a,np.zeros((2,mumax))))
                b = np.concatenate((b,np.zeros((2,mumax))))
                c = np.concatenate((c,np.zeros((2,mumax))))
            
            a[i,:] = 0.5 * (a[i-1,:] + b[i-1,:])
            b[i,:] = np.sqrt(a[i-1,:] * b[i-1,:])
            c[i,:] = 0.5 * (a[i-1,:] - b[i-1,:])
            II = np.where((abs(c[i,:]) <= tol) & (abs(c[i-1,:]) > tol))[0]
            
            if len(II) != 0:
                n[II] = np.ones(np.shape(II))*(i-1)
            
            if i>100:
                raise ValueError

        mmax = len(I)
        mn = np.max(n)
        phin = np.zeros(mmax)     
        C  = np.zeros(mmax)    
        Cp = copy(C)  
        e  = np.int32(C)  
        phini = signU*u[I]
        phin = phini.reshape(phin.shape)
        i = 0  
        c2 = c**2
        while i < mn:                                                    #Descending Landen Transformation 
            i = i + 1
            II = np.where(n[K] > i)[0]
            if len(II) != 0:     
                phin[II] = np.arctan(b[i,K[II]]/a[i,K[II]]*np.tan(phin[II])) + np.pi*np.ceil(phin[II]/np.pi - 0.5) + phin[II]
                e[II] = 2**(i-1) 
                C[II] = C[II]  + float(e[II[0]])*c2[i,K[II]]
                Cp[II]= Cp[II] + c[i+1,K[II]]*np.sin(phin[II])  
        
        Ff = phin / (a[mn,K]*float(e)*2)                                                      
        F[I] = Ff*signU                                               #Incomplete Ell. Int. of the First Kind
        Z[I] = Cp*signU                                               #Jacobi Zeta Function
        E[I] = (Cp + (1 - 1/2*C) * Ff)*signU                         #Incomplete Ell. Int. of the Second Kind

    #Special cases: m == {0, 1}
    m0 = np.where(m == 0)[0]
    if len(m0) != 0:
        F[m0] = u[m0]
        E[m0] = u[m0]
        Z[m0] = 0

    m1 = np.where(m == 1)[0]
    um1 = abs(u[m1]) 
    if len(m1) != 0: 
        N = np.floor( (um1+np.pi/2)/np.pi )  
        M = np.where(um1 < np.pi/2)[0]              
        
        F[m1[M]] = np.log(np.tan(np.pi/4 + u[m1[M]]/2))
        ii = np.where(um1 >= np.pi/2)[0]   
        F[m1[ii]] = np.inf*np.sign(u[m1[ii]])
        
        E[m1] = ((-1)**N * np.sin(um1) + 2*N)*np.sign(u[m1]) 
        
        Z[m1] = (-1)**N * np.sin(u[m1])                      
    
    return F,E,Z
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
    return [(0,0),(0,1),(0,2),[0,0],[0,1],[1,1]][i]
    
def Esh_sol(inp, computeDisp=False, computeStress=False, computeStrain=False, ncpus = 1):
    
    
    return output

def run_all(pars, args, ncpus):
    if ncpus != 1:
        p = Pool(ncpus)
        outs = p.map(run_one, zip([p[-1] for p in pars], [args for i in range(len(pars))]))
    else:
        outs = [run_one([p[-1], args]) for p in pars]
    
    return outs

def run_one(inps):
    x,args = inps
    R, Rb, R_i, Rb_i, vm, dim, eigen, computeDisp, computeD4, computeStress, computeStrain, eigen, incstrain, incstress,epsvec,stressvec,Cm = args

    pos = np.dot(np.dot(R_i,Rb),x)
    out = Esh(vm, dim, pos, eigen, computeDisp, computeD4)
    d = np.zeros(3)
    sn = np.zeros(6)
    sr = np.zeros(6)
    
    if computeDisp and computeD4:
        rd4=Cmatrix(out[0])
        d = np.dot(np.dot(R,Rb_i),out[1])
    elif computeDisp:
        rd4=Cmatrix(out)
    elif computeD4:
        d = np.dot(np.dot(R,Rb_i),out)

    if computeD4:
        if pos[0]**2/dim[0]**2+pos[1]**2/dim[1]**2+pos[2]**2/dim[2]**2 <= 1: # for interior points
            if computeStrain:
                sn = incstrain
            if computeStress:
                sr = incstress
        else:
            if computeStrain:
                strainr = epsvec+np.dot(np.squeeze(rd4),eigen)
                sn = rotate(strainr, R, Rb_i)
            if computeStress:
                stressr = stressvec+np.dot(np.dot(Cm,np.squeeze(rd4)),eigen)
                sr = rotate(stressr, R, Rb_i)

    return d, sn, sr

def Esh_D4(vm, a, x, eigen):
    return Esh(vm, a, x, eigen, computeD4=True)

def Esh_disp(vm, a, x, eigen):
    return Esh(vm, a, x, eigen, computeDisp=True)
    
def Esh_D4_disp(vm, a, x, eigen):
    return Esh(vm, a, x, eigen, computeDisp=True, computeD4=True)

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
    # k=((a[0]**2-a[1]**2)/(a[0]**2-a[2]**2))**(1/2) # the elliptic modulus
    m = (a[0]**2-a[1]**2)/(a[0]**2-a[2]**2) # m=k**2 is the parameter
    F, E, Z = elliptic12(theta, m) #this sets the tolerance to eps, add a third argument to set to a larger tol

    #******************************************************************#
    #Calculation of I's
    #******************************************************************#

    Ifir = np.zeros(3)
    Isec = np.zeros((3,3))
    if a[0]==a[1] and a[0]==a[2]:
        # Spherical Case
        delta = np.sqrt(np.prod(a**2+lmb))
        # can simplify to del3=sqrt((a[0]**2+lmb)**3) for sphere
        Ifir=(4/3)*np.pi*a[0]**3/(np.sqrt(a[0]**2+lmb))**3*np.ones(3)
        Isec=(4/5)*np.pi*a[0]**3/np.sqrt(a[0]**2+lmb)*np.ones((3,3))  # todo: i changed the 5/2 to 1/2 to make units right--not sure if correct

    elif a[0]>a[1] and a[2]==a[1]:
        #fprintf('Prolate case..\n')
        
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
        #fprntf('Oblate case...\n')
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
        #fprintf('triaxial ellipsoid case ..\n')
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


if __name__ == "__main__":
    inc = Inclusion()
    inc.gridx = np.array([0,])
    inc.gridy = np.arange(-25,25,2.5)
    inc.gridz = np.arange(-25,25,2.5)

    inc.solve(computeDisp=True, computeStress=True, computeStrain=True, ncpus=2)

    #inc.sol = Esh_sol(inc, computeDisp=True, computeStress=True, computeStrain=True, ncpus=2)
    
    f,ax = plt.subplots(1,1,figsize=(8,8))
    x = inc.sol.x.squeeze()
    y = inc.sol.y.squeeze()
    z = inc.sol.z.squeeze()
    u = inc.sol.disp.squeeze()[:,:,0]
    umin = u.min()
    umax = u.max()
    cax = ax.contourf(y,z,u, levels = np.linspace(umin, umax,11))
    plt.colorbar(cax)
    plt.show()