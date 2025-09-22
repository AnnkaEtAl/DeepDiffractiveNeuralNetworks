# TensorFlow and tf.keras
import numpy as np
from functools import *
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras.layers import Resizing, Flatten, Reshape, Permute # type: ignore
import sys
import scipy.io


##Not thoroughly tested yet!
#To Do: UPSCALE! Lens for smooth phase
#See initial tests in test_layers
#Possibly there is a more concise version without the need to apply free space propagation twice

class Lens(tf.keras.layers.Layer):
    """
    A class to compute the interaction with a spherical lens. The transferfunction is computed according to the Fomo script.
    Attributes:
        lam(float): wavelength
        x (float[Nx]): meshing of computational domain in x direction
        y (float[Ny]): meshing of computational domain in y direction
        z (float): propagation distance before lens
        z (float): propagation distance after lens
        f (float): focal length
        R (float): radius of lens (physical size of aperture, default None), 
                   for r>R bigger than R the light is blocked

    Methods:
        call:   takes scalar input field of the form tensor(complex[batch, Ny, Nx] 
                and returns the field after a lens tensor(complex[batch, Ny, Nx])

    """
    def __init__(self, x, y, z1, z2, lam, f, R = None):
        super(Lens, self).__init__()
        self.lam = lam
        self.x  = x
        self.y  = y
        self.z1  = z1
        self.z2  = z2
        self.f  = f
        self.R  = R

        self.prop_in = FreeSpacePropagation(x, y, z1, lam)
        self.prop_out = FreeSpacePropagation(x, y, z2, lam)

    def apply_lens(self, inputs):

        # Transferfunction thin lens tL
        X,Y = np.meshgrid(self.x, self.y, indexing = 'ij')
        r   = np.sqrt(X**2 +Y**2)
        k   = 2*np.pi/self.lam
        tL  = np.exp(-1j*k/(2*self.f)*(X**2 +Y**2))

        #Apply Aperture
        if self.R:
            tL[r>self.R] = 0

        # Apply lens including propagation
        return inputs * tf.cast(tL, inputs.dtype) 


    def call(self, inputs):

        if self.z1>0:
            inputs = self.prop_in(inputs) #propagate to lens
        inputs = tf.map_fn(self.apply_lens, inputs) 
        if self.z2>0:
            inputs = self.prop_out(inputs)    #propagate to imaging plane
        return inputs
    

#SCALING NOT CORRECT!   
class opticalFFT(tf.keras.layers.Layer):
    """
    A class to compute an optical FFT, based on FOMO-script
    Attributes:
        lam(float): wavelength
        x (float[Nx]): meshing of computational domain in x direction
        y (float[Ny]): meshing of computational domain in y direction
        f (float): focal length

    Methods:
        call:   takes scalar input field of the form tensor(complex[batch, Ny, Nx] 
                and returns the field after a lens tensor(complex[batch, Ny, Nx])

    """
    def __init__(self, x, y, lam, f, pad = 0, **kwargs):
        super(opticalFFT, self).__init__(**kwargs)
        self.lam = lam
        self.pad = pad
        self.x  = x
        self.y  = y
        self.f  = f
        self.Nx = np.shape(self.x)[0] #amount of dots in the computational domain
        self.Ny = np.shape(self.y)[0]

    #@tf.function
    def Setup2f(self, inputs):

        # Wavevector
        k     =  2 * np.pi/self.lam

        # Sampling
        x = self.x
        y = self.y
        xin, yin, xout, yout = np.meshgrid(x, y, x, y, indexing='ij')

        # Exponent
        exp = np.exp(1j * 2*np.pi *k/self.f*(xin*xout + yin*yout))
        exp = tf.cast(exp, dtype=inputs.dtype)
        exp = tf.transpose(tf.reshape(exp, (self.Nx *self.Ny, self.Nx *self.Ny)))
        #exp = tf.transpose(tf.reshape(exp, (self.Nx *self.Ny, self.Nx *self.Ny)))

        # the image needs to be repeated for every output pixel
        # since we want to sum over the whole image for each pixel
        u0 = tf.expand_dims(tf.reshape(inputs, -1), axis=0)
        u0 = tf.transpose(u0)

        # Prefactor
        fac = tf.convert_to_tensor(-1j * (2 * np.pi) * k/self.f * np.exp(1j * 2 * k * self.f), dtype=inputs.dtype)

        # 2f-Result
        u2f = fac *tf.matmul(exp, u0)
        u2f = tf.reshape(u2f, (self.Ny, self.Nx))

        return u2f#/tf.reduce_max(u2f)

    @staticmethod
    def inbuildFFT(in_):
        res = tf.signal.fft2d(tf.signal.fftshift(in_))
        norm = tf.math.reduce_max(tf.math.abs(Flatten()(res)))
        norm = tf.cast(norm, dtype=tf.complex64)
        res = res / norm
        res = tf.signal.fftshift(res)
        return res
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "lam":self.lam,
            "pad":self.pad, 
            "x":self.x,  
            "y":self.y,  
            "f":self.f,  
            "Nx":self.Nx, 
            "Ny":self.Ny, 
            })
        return config
    
    def call(self, inputs):
        #out = tf.map_fn(self.Setup2f, inputs)
        out = tf.map_fn(self.inbuildFFT, inputs)
        return  out



class FreeSpacePropagation(tf.keras.layers.Layer):
    """
    A class to compute the free space propagation of light.
    Attributes:
        lam(float): wavelength
        x (float[Nx]): meshing of computational domain in x direction
        y (float[Ny]): meshing of computational domain in y direction
        z (float): propagation distance
        Nx (int) : number of meshing points in x-direction
        Ny (int) : number of meshing points in y-direction
        method (string): 'RS_FFT', 'RS_Conv', 'RS_AS', RS_FFT_ver2', RS_conv_old'
        mode : True if you want to apply simson rule to 'RS_FFT' method. False by default.
        pad  : Padding applied in 'RS_AS' method 

    Arguments:
        lam(float): wavelength
        x (float[Nx]): meshing of computational domain in x direction
        y (float[Ny]): meshing of computational domain in y direction
        z (float): propagation distance
        method (string): 'RS_FFT', 'RS_Conv', 'RS_AS', RS_FFT_ver2', RS_conv_old'
        mode : True if you want to apply simson rule to 'RS_FFT' method. False by default.
        pad  : Padding applied in 'RS_AS' method 

    Methods:
        call:   takes scalar input field of the form tensor(complex[batch, Ny, Nx] 
                and returns propagated field tensor(complex[batch, Ny, Nx])

    """
    def __init__(self, x, y, z, lam, method = 'RS_FFT', mode = False, pad = 0, **kwargs):

        super(FreeSpacePropagation, self).__init__(**kwargs)
        self.lam = lam
        self.x  = x
        self.y  = y
        self.z  = z
        self.Nx = np.shape(self.x)[0]
        self.Ny = np.shape(self.y)[0]
        self.method     = method
        self.propagator = {'RS_FFT': self.RS_FFT, 'RS_Conv': self.RS_Conv, 'RS_Conv_old': self.RS_Conv_old,
                           'RS_AS': self.RS_AS, 'RS_FFT_ver2': self.RS_FFT_ver2}

        
        # The weighted correction only exists for the RS_FFT, throw a warning if mode True with other method
        # Also the simson rule can only applied to odd N throw a warning if N is not odd
        # Also throw an error if an invalid method is chosen
        if (mode)  and (method == 'RS_FFT') and (self.Nx%2==1) and (self.Ny%2==1):
            self.simpsonRule = True
        elif (mode) and not(method == 'RS_FFT') and not((self.Nx%2==1) and (self.Ny%2==1)):
            self.simpsonRule = False
            print("WARNING: The Simson Rule (mode=True) can only be applied for uneaven Nx and Ny and the RS_FFT method, setting mode = False")
        else:
            self.simpsonRule = False

        if method == 'RS_AS' or method == 'RS_FFT_ver2':
            self.pad = pad
        elif not(method == 'RS_AS') and pad:
            print("WARNING: padding can only be added to RS_AS and RS_FFT_ver2 method")

               
    @tf.function
    def RS_FFT(self, inputs):
        """
        Best for small Apertures at large distances.
        The propagator is expressed by the analytical form in real space. 
        Rayleigh Sommerfield diffraction, as described by:
        Rayleigh-Sommerfeld (RS) - Shen, F., and Wang, A. (2006). Fast-Fourier-transform based numerical integration method 
        for the Rayleigh-Sommerfeld diffraction formula. Applied Optics, 45(6), 1102–1110. https://doi.org/10.1364/AO.45.001102.add()
        For mode = True, the simspon rule is applied to improve errors by integration.
        """
        #Apply the simpson rule if requested, this should improve the error
        if self.simpsonRule:
            a  = np.ones((1,self.Ny))
            b  = np.ones((1,self.Nx))
            a[:,1:-1:2] ,  a[:,2:-1:2] =  (a[:,1:-1:2] +3 ,  a[:,2:-1:2]+1)
            b[:,1:-1:2] ,  b[:,2:-1:2] =  (b[:,1:-1:2] +3 ,  b[:,2:-1:2]+1)
            wSimpson = (b.T*a)/9 #*self.dx*self.dy
            inputs   = inputs * tf.cast(wSimpson, dtype =  inputs.dtype)

        #Expand X and Y
        X  = np.concatenate((self.x[0]-self.x[-1:0:-1], self.x- self.x[0]))
        Y  = np.concatenate((self.y[0]-self.y[-1:0:-1], self.y- self.y[0]))
    
        # Expand x and y to catch all differences 
        dx, dy = (X[1]-X[0],Y[1]-Y[0])
        px = (self.Nx-1)
        py = (self.Ny-1)
        
        # Pad input to 2N-1 match the sampling
        paddings = tf.constant([[0, py], [0, px]])
        u0       = tf.pad(inputs, paddings, 'CONSTANT')

        # Computing the impulse response    
        r     = np.sqrt(self.z**2 + X**2 + np.expand_dims(Y, axis = 1)**2)  #equistance between points          
        k     = 2*np.pi/self.lam                                          # wave vector
        h     = 1/(np.pi*2)*((1/r)-(1.j*k))*np.exp(1.j*k*r)*self.z/r**2  # impulse response
        h     = tf.cast(h, inputs.dtype)

        # proagated field in the fourier somain
        FT_u0  = tf.signal.fft2d(u0)
        FT_h   = tf.signal.fft2d(h)

        # propagated field in the spatial domain
        Uz    = tf.signal.ifft2d(tf.multiply(FT_u0 , FT_h))
        Uz    = Uz* dx * dy

        return Uz[py:, px:] 

    @tf.function
    def RS_FFT_ver2(self, inputs):
        """
        DEPRECIATED: Version of RS FFT that allows padding. 
        However, accuracy can directly be improved by changing the spatial resolution dx, dy.
        """
        #Apply the simpson rule if requested, this should improve the error
        if self.simpsonRule:
            a  = np.ones((1,self.Ny))
            b  = np.ones((1,self.Nx))
            a[:,1:-1:2] ,  a[:,2:-1:2] =  (a[:,1:-1:2] +3 ,  a[:,2:-1:2]+1)
            b[:,1:-1:2] ,  b[:,2:-1:2] =  (b[:,1:-1:2] +3 ,  b[:,2:-1:2]+1)
            wSimpson = (b.T*a)/9 #*self.dx*self.dy
            inputs   = inputs * tf.cast(wSimpson, dtype =  inputs.dtype)


        #Expand X and Y
        X  = self.x#self.x#np.fft.fftshift(x)
        Y  = self.y#np.fft.fftshift(y)
    
        # Expand x and y to catch all differences 
        dx, dy = (self.x[1]-self.y[0],self.x[1]-self.y[0])
        
        # Pad input to 2N-1 match the sampling
        paddings = tf.constant([[self.pad, self.pad], [self.pad, self.pad]])
        u0       = tf.pad(inputs, paddings, 'CONSTANT')

        # Computing the impulse response    
        r     = np.sqrt(self.z**2 + X**2 + np.expand_dims(Y, axis = 1)**2)  #equistance between points          
        k     = 2*np.pi/self.lam                                          # wave vector
        h     = 1/(np.pi*2)*((1/r)-(1.j*k))*np.exp(1.j*k*r)*self.z/r**2  # impulse response
        h     = tf.cast(h, inputs.dtype)
        h     = tf.pad(h, paddings, 'CONSTANT')

        # proagated field in the fourier somain
        FT_u0  = tf.signal.fft2d(u0)
        FT_h   = tf.signal.fft2d(h)

        # propagated field in the spatial domain
        uz     = tf.signal.fftshift(tf.multiply(FT_u0 , FT_h))
        Uz     = tf.signal.ifftshift(tf.signal.ifft2d(uz))
        Uz     = Uz* dx * dy

        if self.pad:
            #print("here")
            return Uz[self.pad:-self.pad, self.pad:-self.pad] 
        else:
            return Uz
       
    @tf.function
    def RS_Conv_old(self, inputs):
        '''
        DEPRECIATED: Rayleigh-Sommerfield Diffraction. 
        Implementation via direct convolution of field with impulse response.
        Depreciated because FFT method is the same mathematically, yielding equivalent 
        results with significant speedup and RAM efficiency
        The propagator is expressed by the analytical form in real space. 
        '''

        # Creating convolutional 4d mesh
        xg1, yg1, xg2, yg2 = np.meshgrid(self.x, self.y, self.x, self.y)

        # length of vectors connecting input and output plane at each point
        r = np.sqrt((xg1 - xg2) ** 2 + (yg1 - yg2) ** 2 + self.z ** 2)

        # pixel sizes
        dx, dy = (self.x[1]-self.x[0], self.y[1]-self.y[0])

        # impulse response
        k     = 2*np.pi/self.lam   
        h     = 1/(np.pi*2)*((1/r)-(1.j*k))*np.exp(1.j*k*r)*self.z/r**2
        h     = tf.cast(h.reshape(self.Nx*self.Ny, self.Nx*self.Ny), dtype =  inputs.dtype)

        # direct convolution in spatial domain
        Uz    = tf.matmul(Flatten()(tf.expand_dims(inputs, axis = 0)), h*dx*dy)

        # Fixing the dimensions, as currently we have a flattened input
        Uz = Permute([2, 1])(Reshape((self.Ny, self.Nx))(Uz))

        return Uz
    
    @tf.function
    def RS_Conv(self, inputs):
        '''
        DEPRECIATED: Rayleigh-Sommerfield Diffraction. 
        Implementation via direct convolution of field with impulse response.
        Same as RS_Conv_old but iteratively, saving RAM. Incredibly slow.
        '''

        # Creating vector r in input and output plane
        X, Y = np.meshgrid(self.x, self.y, indexing = 'ij')

        # pixel sizes
        dx, dy = (self.x[1]-self.x[0], self.y[1]-self.y[0])

        # convolution
        k     = 2*np.pi/self.lam  
        Uz = tf.zeros(0, inputs.dtype)
        for xout in self.x:
            for yout in self.y:
                # distance to a certain output puxel
                r     = np.sqrt((X- xout) ** 2 + (Y - yout) ** 2 + self.z ** 2)
                # respective implse response (propagator)
                h     = 1/(np.pi*2)*((1/r)-(1.j*k))*np.exp(1.j*k*r)*self.z/r**2
                h     = tf.cast(h,inputs.dtype)
                temp  = tf.reduce_sum(h*inputs)* dx*dy
                Uz    = tf.concat([Uz, [temp]],0)

        Uz = tf.reshape(Uz, (self.Ny, self.Nx))
        return Uz

    @tf.function
    def RS_AS(self, inputs):
        '''
        Best for big Apertures at small distances.
        Rayleigh-Sommerfiedl diffraction with direct integration. 
        The propagator is expressed by the analytical form in fourierspace. 
        '''
        #Mesh in spatial domain
        dx, dy = (self.x[1]-self.x[0],self.y[1]-self.y[0])

        # Analytical Impulse response    
        k  = 2*np.pi/self.lam                                        
        fx = np.fft.fftshift(np.fft.fftfreq(self.Nx+2*self.pad, d = dx))
        fy = np.fft.fftshift(np.fft.fftfreq(self.Ny+2*self.pad, d = dy))
        fxx, fyy = np.meshgrid(fx, fy, indexing='ij')
        arg  =(2 * np.pi)**2 * ((1. / self.lam) ** 2 - fxx**2-fyy**2)
        kz = np.sqrt(np.abs(arg))
        kz = np.where(arg >= 0, kz, 1j*kz)

        # proagated field in the fourier domain

        paddings = tf.constant([[self.pad, self.pad], [self.pad, self.pad]])
        inputs   = tf.pad(inputs, paddings, 'CONSTANT')
        FT_u0  = tf.signal.fftshift(tf.signal.fft2d(inputs))
        FT_h   = tf.cast(tf.exp(1j * kz * self.z),  inputs.dtype)

        # increase accuracy
        Uz     = tf.multiply(FT_u0 , FT_h)
        # propagated field in the spatial domain
        Uz    = tf.signal.ifft2d(tf.signal.ifftshift(Uz))

        if self.pad:
            return Uz[self.pad:-self.pad, self.pad:-self.pad] 
        else:
            return Uz


    def get_config(self):
        config = super().get_config()
        return config

    def call(self, inputs):
        """
        Computes the modulated field after interaction with the metasurface
        Args:
            inputs (complex[batch_size, N, N]):  field before propagation
        Returns:
            complex[batch_size, N, N]: field after propagation
        """    

        # Computes the Free space propagation

        #print(inputs.dtype)
        res = tf.map_fn(self.propagator[self.method], inputs)
        
        return res


class Meta(tf.keras.layers.Layer):

    """
    A neural network layer that represents the metasurface, it manipulates the amplitude and phase of the incident field.
    By default only the phase is trainable. In Neural Network terminology it represents an a linear activation.
    The factor m in f(x)=m*x is determined by the phase and amplitude modulation by the metasurface at each pixel.

    Attributes:
        lam(float): wavelength
        x (float[Nx]): meshing of computational domain in x direction
        y (float[Ny]): meshing of computational domain in y direction
        N_meta (int) :   Number of metasurface pixels in one direction
        Dx_meta (float): lateral size of metasurface in x direction
        Dy_meta (float): lateral size of metasurface in y direction
        amplitude (float32[N_meta,N_meta]):    variable containing the amplitude manipulation due to the metasurface
        theta (float32[N_meta,N_meta]):        trainable variable containing the phase shift due to the metasurface
        delx(float), dely(float):   shift of metasurface in lateral directionwith respect to the center, 
                                    if you choose this make sure to also choose an approriate padding to image the whole metasurface

    Arguments:
        lam(float): wavelength
        x (float[Nx]): meshing of computational domain in x direction
        y (float[Ny]): meshing of computational domain in y direction
        N_meta (int) :   Number of metasurface pixels in one direction
        Dx_meta (float): lateral size of metasurface in x direction
        Dy_meta (float): lateral size of metasurface in y direction

    Methods:
        call:  computes the modulated field after interaction with the metasurface
    """

    def __init__(self, x, y, lam, N_Neurons, Dx_meta, Dy_meta, embedding = 0,  **kwargs):

        super(Meta, self).__init__(**kwargs)
        self.lam = lam
        # Computational Domain
        self.y,self.x      = (y,x)
        self.Dx, self.Dy   = (x[-1]-x[0], y[-1]-y[0])
        Nx, Ny   = (np.shape(x)[0], np.shape(y)[0])
        # Metasurface
        self.N_Neurons     = N_Neurons
        self.Dx_meta       = Dx_meta
        self.Dy_meta       = Dy_meta
        

        #Make sure metasurface fits into computational domain
        if (Dx_meta  > self.Dx) or (Dy_meta > self.Dy):
            sys.exit("Metasurface dimensions exceed the computational domain")

        # Computes the number of metasurface pixels on our defined mesh
        ymeta = y[np.round(np.abs(y),10)<=np.round(Dy_meta/2,10)]
        xmeta = x[np.round(np.abs(x),10)<=np.round(Dx_meta/2,10)]
        self.Nx_meta, self.Ny_meta  = (len(xmeta),len(ymeta)) 
        
        if (self.N_Neurons  > self.Nx_meta) or (self.N_Neurons  > self.Ny_meta) :
            sys.exit("More Neurons than Meshing points, choose a finer meshing, less Neurons or a larger computational domain")    
                
        # Metasurface and domain should both be even or odd
        if not(Nx%2 == self.Nx_meta%2):
            self.Nx_meta = self.Nx_meta - 1
            print("Update the number of meshing points, i.e the dimensions of the metasurface are adjusted")
        if not(Ny%2 == self.Ny_meta%2):
            self.Ny_meta = self.Ny_meta - 1
            print("Update the number of meshing points, i.e the dimensions of the metasurface are adjusted")
            print((Ny-self.Ny_meta)/2)
            
        # Update the number of meshing points, i.e the dimensions of the metasurface are adjusted

        # Resizing of metasurface to fit the mesh of the domain
        self.Upscale = Resizing(self.Ny_meta, self.Nx_meta, interpolation  = 'nearest')                

        # Embedd Metasurface into computational domain
        self.embedding = embedding
        py, px = ((Ny-self.Ny_meta)//2, (Nx-self.Nx_meta)//2)
        #print(self.Ny_meta+ 2*py, len(x), self.Ny_meta, py)
        self.padding   = [[py,py],[px,px]]

        #Training variables
        self.amplitude = tf.Variable(
            tf.ones((self.N_Neurons,self.N_Neurons)),
            trainable = False,
            name = "amplitude")
        self.theta = tf.Variable(
            initial_value = tf.ones((self.N_Neurons,self.N_Neurons))*0.5,
            trainable = True,
            name = "theta",
            constraint=lambda t: tf.clip_by_value(t, 0., 1.))

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, inputs):
        """
        Computes the modulated field after interaction with the metasurface
        Args:
            inputs (complex[batch_size, N, N]):  complex amplitude of input field
        Returns:
            complex[batch_size, N, N]: complex amplitude by metasurface phase-shifted input field
        """
    
        amp_mod    = tf.nn.relu(self.amplitude) / tf.reduce_max(tf.nn.relu(self.amplitude))
        amp_mod    = tf.squeeze(self.Upscale(tf.expand_dims(amp_mod, axis = -1)))
        amp_mod    = tf.cast(amp_mod , inputs.dtype)
        
        phase_mod  = tf.squeeze(self.Upscale(tf.expand_dims(self.theta, axis = -1)))
        phase_mod  = tf.complex(tf.cos(2*np.pi*phase_mod), tf.sin(2*np.pi*phase_mod))
        phase_mod  = tf.cast(phase_mod, inputs.dtype)
       
        embedding   = tf.cast(self.embedding, inputs.dtype)
        metasurface = tf.multiply(amp_mod, phase_mod)
        metaLayer   = tf.pad(metasurface , self.padding, constant_values = embedding, mode = 'constant')

        return inputs * metaLayer


class Meta_q_phase(tf.keras.layers.Layer):

    """
    A neural network layer that represents the metasurface, it manipulates the amplitude and phase of the incident field.
    By default only the phase is trainable. In Neural Network terminology it represents an a linear activation.
    The factor m in f(x)=m*x is determined by the phase and amplitude modulation by the metasurface at each pixel.

    Attributes:
        lam(float): wavelength
        x (float[Nx]): meshing of computational domain in x direction
        y (float[Ny]): meshing of computational domain in y direction
        N_meta (int) :   Number of metasurface pixels in one direction
        Dx_meta (float): lateral size of metasurface in x direction
        Dy_meta (float): lateral size of metasurface in y direction
        amplitude (float32[N_meta,N_meta]):    variable containing the amplitude manipulation due to the metasurface
        theta (float32[N_meta,N_meta]):        trainable variable containing the phase shift due to the metasurface
        delx(float), dely(float):   shift of metasurface in lateral directionwith respect to the center, 
                                    if you choose this make sure to also choose an approriate padding to image the whole metasurface

    Arguments:
        lam(float): wavelength
        x (float[Nx]): meshing of computational domain in x direction
        y (float[Ny]): meshing of computational domain in y direction
        N_meta (int) :   Number of metasurface pixels in one direction
        Dx_meta (float): lateral size of metasurface in x direction
        Dy_meta (float): lateral size of metasurface in y direction

    Methods:
        call:  computes the modulated field after interaction with the metasurface
    """

    def __init__(self, x, y, lam, N_Neurons, Dx_meta, Dy_meta, q_mode='f', embedding = 0,  **kwargs):

        super(Meta_q_phase, self).__init__(**kwargs)
        self.lam = lam
        # Computational Domain
        self.y,self.x      = (y,x)
        self.Dx, self.Dy   = (x[-1]-x[0], y[-1]-y[0])
        Nx, Ny   = (np.shape(x)[0], np.shape(y)[0])
        # Metasurface
        self.N_Neurons     = N_Neurons
        self.Dx_meta       = Dx_meta
        self.Dy_meta       = Dy_meta
        if q_mode == 'f':
          self.plevels = np.load('phases_library.npz')['arr_0']
        else:
          self.plevels = np.load('phases.npz')['arr_1'] 
          
        self.plevels = tf.convert_to_tensor(self.plevels, dtype=tf.float32)
        
        

        #Make sure metasurface fits into computational domain
        if (Dx_meta  > self.Dx) or (Dy_meta > self.Dy):
            sys.exit("Metasurface dimensions exceed the computational domain")

        # Computes the number of metasurface pixels on our defined mesh
        ymeta = y[np.round(np.abs(y),10)<=np.round(Dy_meta/2,10)]
        xmeta = x[np.round(np.abs(x),10)<=np.round(Dx_meta/2,10)]
        self.Nx_meta, self.Ny_meta  = (len(xmeta),len(ymeta)) 
        
        if (self.N_Neurons  > self.Nx_meta) or (self.N_Neurons  > self.Ny_meta) :
            sys.exit("More Neurons than Meshing points, choose a finer meshing, less Neurons or a larger computational domain")    
                
        # Metasurface and domain should both be even or odd
        if not(Nx%2 == self.Nx_meta%2):
            self.Nx_meta = self.Nx_meta - 1
            print("Update the number of meshing points, i.e the dimensions of the metasurface are adjusted")
        if not(Ny%2 == self.Ny_meta%2):
            self.Ny_meta = self.Ny_meta - 1
            print("Update the number of meshing points, i.e the dimensions of the metasurface are adjusted")
            print((Ny-self.Ny_meta)/2)
            
        # Update the number of meshing points, i.e the dimensions of the metasurface are adjusted

        # Resizing of metasurface to fit the mesh of the domain
        self.Upscale = Resizing(self.Ny_meta, self.Nx_meta, interpolation  = 'nearest')                

        # Embedd Metasurface into computational domain
        self.embedding = embedding
        py, px = ((Ny-self.Ny_meta)//2, (Nx-self.Nx_meta)//2)
        #print(self.Ny_meta+ 2*py, len(x), self.Ny_meta, py)
        self.padding   = [[py,py],[px,px]]

        #Training variables
        self.amplitude = tf.Variable(
            tf.ones((self.N_Neurons,self.N_Neurons)),
            trainable = False,
            name = "amplitude")
        self.theta = tf.Variable(
            initial_value = tf.ones((self.N_Neurons,self.N_Neurons))*0.5,
            trainable = True,
            name = "theta",
            constraint=lambda t: tf.clip_by_value(t, 0., 1.))

    def get_config(self):
        config = super().get_config()
        return config

    def quantize_theta(self,theta, levels):
        # For each element in theta, find the nearest allowed level
        # One way: 
        theta_expanded = tf.expand_dims(theta, -1)          # shape [N, N, 1]
        dist = tf.abs(theta_expanded - levels)              # shape [N, N, num_levels]
        indices = tf.argmin(dist, axis=-1)                  # find closest index for each theta value
        # Gather the quantized values
        theta_quant = tf.gather(levels, indices)
        return theta_quant

    def call(self, inputs):
        """
        Computes the modulated field after interaction with the metasurface
        Args:
            inputs (complex[batch_size, N, N]):  complex amplitude of input field
        Returns:
            complex[batch_size, N, N]: complex amplitude by metasurface phase-shifted input field
        """

            # Quantize theta
        theta_quant = self.quantize_theta(self.theta, self.plevels)
            # Apply straight-through estimator
        theta_ste = self.theta + tf.stop_gradient(theta_quant - self.theta)
    
        amp_mod    = tf.nn.relu(self.amplitude) / tf.reduce_max(tf.nn.relu(self.amplitude))
        amp_mod    = tf.squeeze(self.Upscale(tf.expand_dims(amp_mod, axis = -1)))
        amp_mod    = tf.cast(amp_mod , inputs.dtype)
        
        phase_mod  = tf.squeeze(self.Upscale(tf.expand_dims(theta_ste, axis = -1)))
        phase_mod  = tf.complex(tf.cos(2*np.pi*phase_mod), tf.sin(2*np.pi*phase_mod))
        phase_mod  = tf.cast(phase_mod, inputs.dtype)
       
        embedding   = tf.cast(self.embedding, inputs.dtype)
        metasurface = tf.multiply(amp_mod, phase_mod)
        metaLayer   = tf.pad(metasurface , self.padding, constant_values = embedding, mode = 'constant')

        return inputs * metaLayer   
    
        
        
class MetaBilinearLUT(tf.keras.layers.Layer):
    """
    Meta-surface layer: bilinear interpolation in (a,b) for amplitude & phase LUT.
    Returns (U_meta, a_params, b_params) so directivity layer can reuse a_params/b_params.
    """
    def __init__(self, x, y, lam, N_meta, Dx_meta, Dy_meta, lut_mat_file, norm_amp=False, embedding=0.0, **kwargs):
        super().__init__(**kwargs)
        self.x = tf.constant(x, tf.float32)
        self.y = tf.constant(y, tf.float32)
        self.Dx, self.Dy   = (x[-1]-x[0], y[-1]-y[0])
        self.lam = lam
        Nx, Ny = len(x), len(y)
        self.N_meta = N_meta
        self.norm_amp = norm_amp

        # load LUT
        mat = scipy.io.loadmat(lut_mat_file)
        phase = mat['phase_EySH']
        amp   = mat['amp_EySH']
        #phase = mat['phase'][:,:,0]
        #amp   = mat['amplitude'][:,:,0]
        phase_lut_cor = np.where(phase < 0, phase + 2 * np.pi, phase)
        phase_lut_cor = phase_lut_cor/(2*np.pi)

        self.phase_map = tf.constant(phase_lut_cor, dtype=tf.float32)
        self.amp_map   = tf.constant(amp,   dtype=tf.float32)  # [A,B]
        coords_a = mat['RA'][0]*1e9
        coords_b = mat['RB'][0]*1e9
        #coords_a = mat['RA'][:,0]
        #coords_b = mat['RB'][0]
        self.coords_a = tf.constant(coords_a, dtype=tf.float32)
        self.coords_b = tf.constant(coords_b, dtype=tf.float32)

        self.Dx_meta       = Dx_meta
        self.Dy_meta       = Dy_meta
        

        #Make sure metasurface fits into computational domain
        if (Dx_meta  > self.Dx) or (Dy_meta > self.Dy):
            sys.exit("Metasurface dimensions exceed the computational domain")

        # Computes the number of metasurface pixels on our defined mesh
        ymeta = y[np.round(np.abs(y),10)<=np.round(Dy_meta/2,10)]
        xmeta = x[np.round(np.abs(x),10)<=np.round(Dx_meta/2,10)]
        self.Nx_meta, self.Ny_meta  = (len(xmeta),len(ymeta)) 
        
        if (self.N_meta  > self.Nx_meta) or (self.N_meta  > self.Ny_meta) :
            sys.exit("More Neurons than Meshing points, choose a finer meshing, less Neurons or a larger computational domain")    
                
        # Metasurface and domain should both be even or odd
        if not(Nx%2 == self.Nx_meta%2):
            self.Nx_meta = self.Nx_meta - 1
            print("Update the number of meshing points, i.e the dimensions of the metasurface are adjusted")
        if not(Ny%2 == self.Ny_meta%2):
            self.Ny_meta = self.Ny_meta - 1
            print("Update the number of meshing points, i.e the dimensions of the metasurface are adjusted")
            print((Ny-self.Ny_meta)/2)
            
        # Update the number of meshing points, i.e the dimensions of the metasurface are adjusted

        # Resizing of metasurface to fit the mesh of the domain
        self.Upscale = Resizing(self.Ny_meta, self.Nx_meta, interpolation  = 'nearest')                

        # Embedd Metasurface into computational domain
        self.embedding = embedding
        py, px = ((Ny-self.Ny_meta)//2, (Nx-self.Nx_meta)//2)
        #print(self.Ny_meta+ 2*py, len(x), self.Ny_meta, py)
        self.padding   = [[py,py],[px,px]]

        # trainable a,b
        self.a_params = self.add_weight(
            'a_params', shape=(N_meta,N_meta),
            initializer=tf.random_uniform_initializer(coords_a[0], coords_a[-1]),
            trainable=True)
        self.b_params = self.add_weight(
            'b_params', shape=(N_meta,N_meta),
            initializer=tf.random_uniform_initializer(coords_b[0], coords_b[-1]),
            trainable=True)

    def _bilinear_1d(self, vals, grid):
        """Return idx0, idx1, w0, w1 for bilinear in 1D."""
        v = tf.reshape(vals, [-1])
        idx1 = tf.searchsorted(grid, v, 'left')
        idx1 = tf.clip_by_value(idx1, 1, tf.shape(grid)[0]-1)
        idx0 = idx1 - 1
        g0 = tf.gather(grid, idx0)
        g1 = tf.gather(grid, idx1)
        w1 = (v - g0)/(g1 - g0)
        w0 = 1 - w1
        return idx0, idx1, w0, w1

    def call(self, inputs):
        # bilinear in a and b
        ai0, ai1, wa0, wa1 = self._bilinear_1d(self.a_params, self.coords_a)
        bi0, bi1, wb0, wb1 = self._bilinear_1d(self.b_params, self.coords_b)

        M2 = self.N_meta*self.N_meta
        # gather four corners for amp & phase
        def gather2(m, i, j):
            idx = tf.stack([i, j], 1)
            return tf.gather_nd(m, idx)

        A00 = gather2(self.amp_map, ai0, bi0)
        A01 = gather2(self.amp_map, ai0, bi1)
        A10 = gather2(self.amp_map, ai1, bi0)
        A11 = gather2(self.amp_map, ai1, bi1)
        P00 = gather2(self.phase_map, ai0, bi0)
        P01 = gather2(self.phase_map, ai0, bi1)
        P10 = gather2(self.phase_map, ai1, bi0)
        P11 = gather2(self.phase_map, ai1, bi1)

        # bilinear blend in a,b
        amp_flat = A00*wa0*wb0 + A01*wa0*wb1 + A10*wa1*wb0 + A11*wa1*wb1
        ph_flat  = P00*wa0*wb0 + P01*wa0*wb1 + P10*wa1*wb0 + P11*wa1*wb1

        amp = tf.reshape(amp_flat, [self.N_meta, self.N_meta])
        ph  = tf.reshape(ph_flat,  [self.N_meta, self.N_meta])

        # upsample & pad
        #amp_mod    = tf.nn.relu(amp) / tf.reduce_max(tf.nn.relu(amp))
        amp_mod    = tf.squeeze(self.Upscale(tf.expand_dims(amp, axis = -1)))
        amp_mod    = tf.cast(amp_mod , inputs.dtype)
        
        phase_mod  = tf.squeeze(self.Upscale(tf.expand_dims(ph, axis = -1)))
        phase_mod  = tf.complex(tf.cos(2*np.pi*phase_mod), tf.sin(2*np.pi*phase_mod))
        phase_mod  = tf.cast(phase_mod, inputs.dtype)
       
        embedding   = tf.cast(self.embedding, inputs.dtype)
        metasurface = tf.multiply(amp_mod, phase_mod)
        metaLayer   = tf.pad(metasurface , self.padding, constant_values = embedding, mode = 'constant')
   
        return inputs * metaLayer, self.a_params, self.b_params
        
        
class DirectivityBilinearLUT(tf.keras.layers.Layer):
    """
    Propagate with per-atom directivity LUT [A,B,θ,φ] via bilinear in all four dims.
    """
    def __init__(self, x, y, z, lam, lut_file, N_meta, **kwargs):
        super().__init__(**kwargs)
        # physical coords
        self.x = tf.constant(x, tf.float32)
        self.y = tf.constant(y, tf.float32)
        self.z = tf.constant(z, tf.float32)
        self.k = tf.constant(2.*np.pi/lam, tf.complex64)
        self.Nx, self.Ny = len(x), len(y)
        self.N_meta     = N_meta

        # load 4D LUT
        data = np.load(lut_file)
        a = np.squeeze(data['a_vals']); b = np.squeeze(data['b_vals'])
        th = np.squeeze(data['theta']);   ph = np.squeeze(data['phi'])
        D4 = data['D_LUT']  # shape [A,B,Th,Ph]

        # store as 1-D tensors
        self.a_vals     = tf.constant(a, tf.float32)   # [A]
        self.b_vals     = tf.constant(b, tf.float32)   # [B]
        self.theta_grid = tf.constant(th,tf.float32)   # [Th]
        self.phi_grid   = tf.constant(ph,tf.float32)   # [Ph]
        self.D_LUT      = tf.constant(D4,tf.float32) # [A,B,Th,Ph]

        # build list of each atom's flat index & (x,y)
        py = (self.Ny - N_meta)//2
        px = (self.Nx - N_meta)//2
        m_idxs, n_idxs = np.meshgrid(
            np.arange(N_meta), np.arange(N_meta), indexing='ij'
        )
        rows = (py + m_idxs).ravel()
        cols = (px + n_idxs).ravel()
        self.atom_flat_idx = tf.constant(rows*self.Nx + cols, tf.int32)  # [M2]
        self.xs_flat = tf.constant(x[cols], tf.float32)  # [M2]
        self.ys_flat = tf.constant(y[rows], tf.float32)  # [M2]

        # observation grid flattened
        Yg, Xg = np.meshgrid(y, x, indexing='ij')
        self.Xf = tf.constant(Xg.ravel(), tf.float32)    # [Nx*Ny]
        self.Yf = tf.constant(Yg.ravel(), tf.float32)

    def _bilinear_1d(self, vals, grid):
        # vals: [K], grid: [G]
        # returns (i0,i1,w0,w1) each [K]
        G = tf.shape(grid)[0]
        # if grid only length-1, always return index 0 with w0=1
        def single():
            K = tf.size(vals)
            zeros = tf.zeros([K], tf.int32)
            ones  = tf.ones([K],  tf.float32)
            return zeros, zeros, ones, tf.zeros([K],tf.float32)
        def multi():
            v = tf.reshape(vals, [-1])
            i1 = tf.searchsorted(grid, v, 'left')
            i1 = tf.clip_by_value(i1, 1, G-1)
            i0 = i1 - 1
            g0 = tf.gather(grid, i0)
            g1 = tf.gather(grid, i1)
            w1 = (v - g0) / (g1 - g0)
            w0 = 1 - w1
            return i0, i1, w0, w1
        return tf.cond(G>1, multi, single)

    def bilinear2d(self, D2, theta_q, phi_q):
        # D2: [Th,Ph], theta_q/phi_q: [Nobs]
        ti0, ti1, wt0, wt1 = self._bilinear_1d(theta_q, self.theta_grid)
        pi0, pi1, wp0, wp1 = self._bilinear_1d(phi_q,   self.phi_grid)

        def g(i,j):
            idx = tf.stack([i,j], axis=1)  # [Nobs,2]
            return tf.gather_nd(D2, idx)   # [Nobs]

        D00 = g(ti0, pi0); D01 = g(ti0, pi1)
        D10 = g(ti1, pi0); D11 = g(ti1, pi1)

        return (D00*wt0*wp0 + D01*wt0*wp1 +
                D10*wt1*wp0 + D11*wt1*wp1)

    def call(self, U_meta, a_params, b_params):
        """
        U_meta:   [B,Ny,Nx] complex64
        a_params: [M,M] float32
        b_params: [M,M]
        """
        B = tf.shape(U_meta)[0]
        M2 = self.N_meta * self.N_meta

        # flatten a/b once per sample
        a_flat = tf.reshape(a_params, [-1])
        b_flat = tf.reshape(b_params, [-1])
        ai0, ai1, wa0, wa1 = self._bilinear_1d(a_flat, self.a_vals)
        bi0, bi1, wb0, wb1 = self._bilinear_1d(b_flat, self.b_vals)

        def _prop(u):
            # u: [Ny,Nx] complex64
            u0 = tf.reshape(u, [-1])                 # [Nx*Ny]
            Uout = tf.zeros_like(self.Xf, tf.complex64)
            for k in tf.range(M2):
                amp = u0[self.atom_flat_idx[k]]      # scalar complex
                # skip padded zeros
                if tf.abs(amp) < 1e-8:
                    continue

                xs = self.xs_flat[k]; ys = self.ys_flat[k]
                dx = self.Xf - xs; dy = self.Yf - ys   # [Nx*Ny]
                r  = tf.sqrt(self.z*self.z + dx*dx + dy*dy)
                sph = (tf.exp(1j*self.k*tf.cast(r,tf.complex64))
                       / tf.cast(r,tf.complex64))

                theta_q = tf.acos(self.z / r)         # [Nx*Ny]
                phi_q   = tf.atan2(dy, dx)

                # four (a,b) corner patterns
                D00 = self.bilinear2d(self.D_LUT[ai0[k], bi0[k]],
                                      theta_q, phi_q)
                D01 = self.bilinear2d(self.D_LUT[ai0[k], bi1[k]],
                                      theta_q, phi_q)
                D10 = self.bilinear2d(self.D_LUT[ai1[k], bi0[k]],
                                      theta_q, phi_q)
                D11 = self.bilinear2d(self.D_LUT[ai1[k], bi1[k]],
                                      theta_q, phi_q)

                # weights in (a,b)
                w00 = wa0[k]*wb0[k]; w01 = wa0[k]*wb1[k]
                w10 = wa1[k]*wb0[k]; w11 = wa1[k]*wb1[k]
                D_flat = (D00*w00 + D01*w01 +
                          D10*w10 + D11*w11)

                Uout += amp * sph * tf.cast(D_flat, tf.complex64)

            return tf.reshape(Uout, [self.Ny, self.Nx])

        # map over the B batch dimension
        return tf.map_fn(_prop, U_meta)

class Meta_bilinear_lookup(tf.keras.layers.Layer):

    """
    A neural network layer that represents the metasurface, it manipulates the amplitude and phase of the incident field.
    By default only the phase is trainable. In Neural Network terminology it represents an a linear activation.
    The factor m in f(x)=m*x is determined by the phase and amplitude modulation by the metasurface at each pixel.

    Attributes:
        lam(float): wavelength
        x (float[Nx]): meshing of computational domain in x direction
        y (float[Ny]): meshing of computational domain in y direction
        N_meta (int) :   Number of metasurface pixels in one direction
        Dx_meta (float): lateral size of metasurface in x direction
        Dy_meta (float): lateral size of metasurface in y direction
        amplitude (float32[N_meta,N_meta]):    variable containing the amplitude manipulation due to the metasurface
        theta (float32[N_meta,N_meta]):        trainable variable containing the phase shift due to the metasurface
        delx(float), dely(float):   shift of metasurface in lateral directionwith respect to the center, 
                                    if you choose this make sure to also choose an approriate padding to image the whole metasurface

    Arguments:
        lam(float): wavelength
        x (float[Nx]): meshing of computational domain in x direction
        y (float[Ny]): meshing of computational domain in y direction
        N_meta (int) :   Number of metasurface pixels in one direction
        Dx_meta (float): lateral size of metasurface in x direction
        Dy_meta (float): lateral size of metasurface in y direction

    Methods:
        call:  computes the modulated field after interaction with the metasurface
    """

    def __init__(self, x, y, lam, N_Neurons, Dx_meta, Dy_meta, norm_amp = False, embedding = 0,  **kwargs):

        super(Meta_bilinear_lookup, self).__init__(**kwargs)
        self.lam = lam
        # Computational Domain
        self.y,self.x      = (y,x)
        self.Dx, self.Dy   = (x[-1]-x[0], y[-1]-y[0])
        Nx, Ny   = (np.shape(x)[0], np.shape(y)[0])
        # Metasurface
        self.N_Neurons     = N_Neurons
        self.Dx_meta       = Dx_meta
        self.Dy_meta       = Dy_meta
        self.norm_amp = norm_amp

        #Upload LUT data
        mat = scipy.io.loadmat('../LUT_data/Iena_lut/LUT_half_cylinder_on_sapphire_left.mat')
        
        ### map phase values from [-pi, pi] to [0,1]
        phase_lut = mat['phase'][:,:,0]
        phase_lut_cor = np.where(phase_lut < 0, phase_lut + 2 * np.pi, phase_lut)
        phase_lut_cor = phase_lut_cor/(2*np.pi)

        self.phase_map = tf.constant(phase_lut_cor, dtype=tf.float32)
        self.amplitude_map = tf.constant(mat['amplitude'][:,:,0], dtype=tf.float32)
        coords_a = mat['RA'][:,0]
        coords_b = mat['RB'][0]
        self.coords_a = tf.constant(coords_a.squeeze(), dtype=tf.float32)  # shape [A]
        self.coords_b = tf.constant(coords_b.squeeze(), dtype=tf.float32)  # shape [B]


        

        #Make sure metasurface fits into computational domain
        if (Dx_meta  > self.Dx) or (Dy_meta > self.Dy):
            sys.exit("Metasurface dimensions exceed the computational domain")

        # Computes the number of metasurface pixels on our defined mesh
        ymeta = y[np.round(np.abs(y),10)<=np.round(Dy_meta/2,10)]
        xmeta = x[np.round(np.abs(x),10)<=np.round(Dx_meta/2,10)]
        self.Nx_meta, self.Ny_meta  = (len(xmeta),len(ymeta)) 
        
        if (self.N_Neurons  > self.Nx_meta) or (self.N_Neurons  > self.Ny_meta) :
            sys.exit("More Neurons than Meshing points, choose a finer meshing, less Neurons or a larger computational domain")    
                
        # Metasurface and domain should both be even or odd
        if not(Nx%2 == self.Nx_meta%2):
            self.Nx_meta = self.Nx_meta - 1
            print("Update the number of meshing points, i.e the dimensions of the metasurface are adjusted")
        if not(Ny%2 == self.Ny_meta%2):
            self.Ny_meta = self.Ny_meta - 1
            print("Update the number of meshing points, i.e the dimensions of the metasurface are adjusted")
            print((Ny-self.Ny_meta)/2)
            
        # Update the number of meshing points, i.e the dimensions of the metasurface are adjusted

        # Resizing of metasurface to fit the mesh of the domain
        self.Upscale = Resizing(self.Ny_meta, self.Nx_meta, interpolation  = 'nearest')                

        # Embedd Metasurface into computational domain
        self.embedding = embedding
        py, px = ((Ny-self.Ny_meta)//2, (Nx-self.Nx_meta)//2)
        #print(self.Ny_meta+ 2*py, len(x), self.Ny_meta, py)
        self.padding   = [[py,py],[px,px]]

        #Training variables
        self.a_params = tf.Variable(tf.random.uniform((self.N_Neurons,self.N_Neurons), 
                                                      minval=self.coords_a[0], 
                                                      maxval=self.coords_a[-1]),
                                    trainable = True,
                                    name = 'a_params')

        self.b_params = tf.Variable(tf.random.uniform((self.N_Neurons,self.N_Neurons), 
                                                      minval=self.coords_b[0], 
                                                      maxval=self.coords_b[-1]),
                                    trainable = True,
                                    name = 'b_params')

    def get_config(self):
        config = super().get_config()
        return config

    def bilinear_lookup(self, a_params, b_params, coords_a, coords_b, amplitude_map, phase_map):
        # a_params, b_params: shape [N, N], continuous parameters
        # coords_a: [A], coords_b: [B]
        # amplitude_map, phase_map: [A, B]

        # Find indices where a_params and b_params would be inserted
        # Flatten a_params
        a_params_flat = tf.reshape(a_params, [-1])  # shape [N*N]

        # Perform search
        a_idx_flat = tf.searchsorted(coords_a, a_params_flat, side='left')  # shape [N*N]

        # Reshape back
        a_idx = tf.reshape(a_idx_flat, a_params.shape)  # shape [N, N]
        # Flatten a_params
        b_params_flat = tf.reshape(b_params, [-1])  # shape [N*N]

        # Perform search
        b_idx_flat = tf.searchsorted(coords_b, b_params_flat, side='left')  # shape [N*N]

        # Reshape back
        b_idx = tf.reshape(b_idx_flat, b_params.shape)  # shape [N, N]

        # Clip to valid range, ensuring we have neighbors on both sides
        a_idx = tf.clip_by_value(a_idx, 1, tf.shape(coords_a)[0]-1)
        b_idx = tf.clip_by_value(b_idx, 1, tf.shape(coords_b)[0]-1)

        # Identify neighbors
        a_idx0 = a_idx - 1
        a_idx1 = a_idx
        b_idx0 = b_idx - 1
        b_idx1 = b_idx

        a0 = tf.gather(coords_a, a_idx0)
        a1 = tf.gather(coords_a, a_idx1)
        b0 = tf.gather(coords_b, b_idx0)
        b1 = tf.gather(coords_b, b_idx1)

        # Compute interpolation weights
        wa1 = (a_params - a0) / (a1 - a0)  # fraction along a
        wa0 = 1.0 - wa1
        wb1 = (b_params - b0) / (b1 - b0) # fraction along b
        wb0 = 1.0 - wb1

        # Gather the four corner values
        corners = lambda m, ai, bi: tf.gather_nd(m, tf.stack([bi, ai], axis=-1))
        amp_a0b0 = corners(amplitude_map, a_idx0, b_idx0)
        amp_a0b1 = corners(amplitude_map, a_idx0, b_idx1)
        amp_a1b0 = corners(amplitude_map, a_idx1, b_idx0)
        amp_a1b1 = corners(amplitude_map, a_idx1, b_idx1)

        phase_a0b0 = corners(phase_map, a_idx0, b_idx0)
        phase_a0b1 = corners(phase_map, a_idx0, b_idx1)
        phase_a1b0 = corners(phase_map, a_idx1, b_idx0)
        phase_a1b1 = corners(phase_map, a_idx1, b_idx1)

        # Bilinear interpolation
        amplitude = (amp_a0b0 * wa0 * wb0 +
                    amp_a0b1 * wa0 * wb1 +
                    amp_a1b0 * wa1 * wb0 +
                    amp_a1b1 * wa1 * wb1)

        phase = (phase_a0b0 * wa0 * wb0 +
                phase_a0b1 * wa0 * wb1 +
                phase_a1b0 * wa1 * wb0 +
                phase_a1b1 * wa1 * wb1)

        return amplitude, phase

    def call(self, inputs):
        """
        Computes the modulated field after interaction with the metasurface
        Args:
            inputs (complex[batch_size, N, N]):  complex amplitude of input field
        Returns:
            complex[batch_size, N, N]: complex amplitude by metasurface phase-shifted input field

        """
        amplitude, phase = self.bilinear_lookup(self.a_params, self.b_params, self.coords_a, self.coords_b, self.amplitude_map, self.phase_map)

        if self.norm_amp:
            amp_mod    = tf.nn.relu(amplitude) / tf.reduce_max(tf.nn.relu(amplitude))
            amp_mod    = tf.squeeze(self.Upscale(tf.expand_dims(amp_mod, axis = -1)))
        else: 
            amp_mod    = tf.squeeze(self.Upscale(tf.expand_dims(amplitude, axis = -1)))
        amp_mod    = tf.cast(amp_mod , inputs.dtype)
        
        phase_mod  = tf.squeeze(self.Upscale(tf.expand_dims(phase, axis = -1)))
        phase_mod  = tf.complex(tf.cos(2*np.pi*phase_mod), tf.sin(2*np.pi*phase_mod))
        phase_mod  = tf.cast(phase_mod, inputs.dtype)
       
        embedding   = tf.cast(self.embedding, inputs.dtype)
        metasurface = tf.multiply(amp_mod, phase_mod)
        metaLayer   = tf.pad(metasurface , self.padding, constant_values = embedding, mode = 'constant')

        return inputs * metaLayer

class DirectivityFarField(tf.keras.layers.Layer):
    """
    Takes U_meta [B, Ny, Nx] (after Meta mask) and per-atom a_params,b_params [M,M],
    plus a 4D LUT D_LUT[A,B,Th,Ph], and returns the far-field pattern
    E_out [B, Th, Ph].
    """
    def __init__(self, x, y, z, lam, lut_file, N_meta, **kwargs):
        super().__init__(**kwargs)
        # physical coords of global grid
        self.x = tf.constant(x, tf.float32)     # [Nx]
        self.y = tf.constant(y, tf.float32)     # [Ny]
        self.z = tf.constant(z, tf.float32)     # propagation distance
        self.k = tf.constant(2*np.pi/lam, tf.float32)
        self.Nx, self.Ny = len(x), len(y)
        self.N_meta    = N_meta
        M2 = N_meta * N_meta

        # load the full 4D directivity LUT
        data = np.load(lut_file)
        a_vals = np.squeeze(data['a_vals'])*1e9
        b_vals = np.squeeze(data['b_vals'])*1e9
        theta  = np.squeeze(data['theta'])
        phi    = np.squeeze(data['phi'])
        D4     = data['D_LUT'].astype(np.complex64)  # shape [A,B,Th,Ph]

        # store as TF constants
        self.a_vals     = tf.constant(a_vals, tf.float32)   # [A]
        self.b_vals     = tf.constant(b_vals, tf.float32)   # [B]
        self.theta_grid = tf.constant(theta,  tf.float32)   # [Th]
        self.phi_grid   = tf.constant(phi,    tf.float32)   # [Ph]
        self.D_LUT      = tf.constant(D4,      tf.complex64) # [A,B,Th,Ph]

        # build list of each meta-atom’s flat index into the big [Ny*Nx] field
        py = (self.Ny - N_meta)//2
        px = (self.Nx - N_meta)//2
        m_idxs, n_idxs = np.meshgrid(np.arange(N_meta),
                                     np.arange(N_meta),
                                     indexing='ij')
        rows = (py + m_idxs).ravel()   # [M2]
        cols = (px + n_idxs).ravel()   # [M2]
        self.atom_flat_idx = tf.constant(rows*self.Nx + cols, tf.int32)
        # their physical (x,y) positions
        self.x_flat = tf.constant(x[cols], tf.float32)  # [M2]
        self.y_flat = tf.constant(y[rows], tf.float32)  # [M2]

        # precompute sinθ·cosφ and sinθ·sinφ grids
        sinθ = tf.sin(self.theta_grid)[:,None]   # [Th,1]
        cosφ = tf.cos(self.phi_grid)[None,:]     # [1,Ph]
        sinφ = tf.sin(self.phi_grid)[None,:]
        self.sinθ_cosφ = sinθ * cosφ            # [Th,Ph]
        self.sinθ_sinφ = sinθ * sinφ            # [Th,Ph]

    def call(self, U_meta, a_params, b_params):
        """
        U_meta:   [B,Ny,Nx] complex64
        a_params: [M_meta,M_meta] float32
        b_params: [M_meta,M_meta]
        returns:
        E_out:    [B, Th, Ph] complex64
        """
        B = tf.shape(U_meta)[0]
        M2 = self.N_meta * self.N_meta

        # 1) Gather each atom’s field amplitude from the padded U_meta
        u_flat = tf.reshape(U_meta, [B, -1])  # [B, Ny*Nx]
        # u_atoms[b,k] = field at atom k
        u_atoms = tf.gather(u_flat, self.atom_flat_idx, axis=1)  # [B,M2]

        # 2) Quantize (a,b) → nearest grid indices (no interpolation here)
        a_flat = tf.reshape(a_params, [-1])  # [M2]
        b_flat = tf.reshape(b_params, [-1])
        # find insertion point
        ia = tf.searchsorted(self.a_vals, a_flat, 'left')
        ib = tf.searchsorted(self.b_vals, b_flat, 'left')
        # clip to valid interval
        ia = tf.clip_by_value(ia, 1, tf.shape(self.a_vals)[0]-1)
        ib = tf.clip_by_value(ib, 1, tf.shape(self.b_vals)[0]-1)
        # pick nearest neighbor
        aL = tf.gather(self.a_vals, ia-1);  aR = tf.gather(self.a_vals, ia)
        bL = tf.gather(self.b_vals, ib-1);  bR = tf.gather(self.b_vals, ib)
        ia = tf.where(tf.abs(a_flat - aR) < tf.abs(a_flat - aL), ia, ia-1)
        ib = tf.where(tf.abs(b_flat - bR) < tf.abs(b_flat - bL), ib, ib-1)

        # 3) For each atom k, pull its entire [Th,Ph] slice of directivity
        idx2 = tf.stack([ia, ib], axis=1)   # [M2,2]
        D_atom = tf.gather_nd(self.D_LUT, idx2)  # [M2, Th, Ph]

        # 4) Compute the Fraunhofer phase term for each atom×angle
        #    argument = k*( x_k sinθ cosφ + y_k sinθ sinφ )
        #    shape [M2,Th,Ph]
        phase_arg = (self.x_flat[:,None,None] * self.sinθ_cosφ[None,...]
                   + self.y_flat[:,None,None] * self.sinθ_sinφ[None,...])
        sph_phase = tf.exp(1j * tf.cast(self.k * phase_arg, tf.complex64))

        # 5) Combine u_atoms, D_atom, sph_phase and sum over atoms
        #    broadcast u_atoms over [Th,Ph]
        u_bcast = tf.reshape(u_atoms, [B, M2, 1, 1])        # [B,M2,1,1]
        D_bcast = tf.reshape(D_atom, [1, M2, *D_atom.shape[1:]])  # [1,M2,Th,Ph]
        sph     = tf.reshape(sph_phase, [1, M2, *D_atom.shape[1:]])

        E = tf.reduce_sum(u_bcast * D_bcast * sph, axis=1) # [B,Th,Ph]
        return E

class DirectivityLensPropagation(tf.keras.layers.Layer):
  """
  Fully vectorized propagation with per-atom directivity LUT [A,B,Th,Ph].
  1) Build angular spectrum E[Th,Ph] without Python loops
  2) Lens via 2D IFFT back to spatial [Ny,Nx]
  """
  def __init__(self, x, y, z, lam, lut_file, N_meta, **kwargs):
        super().__init__(**kwargs)
        # spatial coords
        self.x = tf.constant(x, tf.float32)    # [Nx]
        self.y = tf.constant(y, tf.float32)    # [Ny]
        self.z = tf.constant(z, tf.float32)
        self.k = tf.constant(2*np.pi/lam, tf.float32)
        self.Nx, self.Ny = len(x), len(y)
        self.M = N_meta
        M2 = N_meta * N_meta

        # load 4D LUT
        data = np.load(lut_file)
        self.a_vals = tf.constant(np.squeeze(data['a_vals'])*1e9, tf.float32)  # [A]
        self.b_vals = tf.constant(np.squeeze(data['b_vals'])*1e9, tf.float32)  # [B]
        self.theta = tf.constant(np.squeeze(data['theta']), tf.float32)    # [Th]
        self.phi   = tf.constant(np.squeeze(data['phi']),   tf.float32)    # [Ph]
        self.D_LUT = tf.constant(data['D_LUT'].astype(np.complex64))       # [A,B,Th,Ph]
        self.Th, self.Ph = self.theta.shape[0], self.phi.shape[0]

        # atom flat indices & positions
        py = (self.Ny - N_meta) // 2
        px = (self.Nx - N_meta) // 2
        m, n = np.meshgrid(np.arange(N_meta), np.arange(N_meta), indexing='ij')
        rows = (py + m).ravel()
        cols = (px + n).ravel()
        self.atom_idx = tf.constant(rows * self.Nx + cols, tf.int32)  # [M2]
        self.xs = tf.constant(x[cols], tf.float32)                    # [M2]
        self.ys = tf.constant(y[rows], tf.float32)                    # [M2]

        # precompute angular phasors
        sinθ = tf.sin(self.theta)[:, None]     # [Th,1]
        cosφ = tf.cos(self.phi)[None, :]       # [1,Ph]
        sinφ = tf.sin(self.phi)[None, :]       # [1,Ph]
        self.sinθ_cosφ = sinθ * cosφ           # [Th,Ph]
        self.sinθ_sinφ = sinθ * sinφ           # [Th,Ph]

  def call(self, U_meta, a_params, b_params):
        B = tf.shape(U_meta)[0]
        M2 = self.M * self.M

        # extract each atom's field amplitude
        Uflat = tf.reshape(U_meta, [B, -1])                 # [B,Ny*Nx]
        Uatoms = tf.gather(Uflat, self.atom_idx, axis=1)    # [B,M2]

        # nearest neighbor in (a,b)
        af = tf.reshape(a_params, [-1])                      # [M2]
        bf = tf.reshape(b_params, [-1])
        ia = tf.argmin(tf.abs(af[:,None] - self.a_vals[None,:]), axis=1)  # [M2]
        ib = tf.argmin(tf.abs(bf[:,None] - self.b_vals[None,:]), axis=1)

        # gather directivity slices
        idx2 = tf.stack([ia, ib], axis=1)                    # [M2,2]
        D_atom = tf.gather_nd(self.D_LUT, idx2)              # [M2,Th,Ph]

        # compute spherical phase for each atom & angle
        # shape: [M2,Th,Ph]
        phase_arg = self.k * (
            self.xs[:, None, None] * self.sinθ_cosφ[None, ...] +
            self.ys[:, None, None] * self.sinθ_sinφ[None, ...]
        )
        sph_phase = tf.exp(1j * tf.cast(phase_arg, tf.complex64))

        # build angular spectrum: [B,Th,Ph]
        # Uatoms: [B,M2], D_atom: [M2,Th,Ph], sph_phase: [M2,Th,Ph]
        U_expanded = tf.reshape(Uatoms, [B, M2, 1, 1])
        D_expanded = tf.reshape(D_atom, [1, M2, self.Th, self.Ph])
        S_expanded = tf.reshape(sph_phase, [1, M2, self.Th, self.Ph])

        E_ang = tf.reduce_sum(U_expanded * D_expanded * S_expanded, axis=1)  # [B,Th,Ph]

        # lens via IFFT2D back to spatial [Ny,Nx]
        E_shift = tf.signal.ifftshift(E_ang, axes=(1,2))
        U_img = tf.signal.ifft2d(E_shift)
        U_img = tf.signal.fftshift(U_img, axes=(1,2))

        # if angular grid != spatial grid, resize
        if self.Th != self.Ny or self.Ph != self.Nx:
            real = tf.image.resize(
                tf.math.real(U_img)[...,None],
                [self.Ny, self.Nx], method='bilinear')
            imag = tf.image.resize(
                tf.math.imag(U_img)[...,None],
                [self.Ny, self.Nx], method='bilinear')
            U_img = tf.complex(tf.squeeze(real,-1), tf.squeeze(imag,-1))

        return U_img

class DirectivityPropagation(tf.keras.layers.Layer):
    """
    Brute-force propagation including per-atom directivity LUT [A,B,Th,Ph].
    Input:  
        U_meta   [B, Ny, Nx] complex64  -- field after metasurface mask  
        a_params [M, M] float32         -- trainable a for each atom  
        b_params [M, M] float32         -- trainable b for each atom  
    Output:
        U_out    [B, Ny, Nx] complex64  -- propagated field with directivity
    """
    def __init__(self, x, y, z, lam, lut_file, N_meta, **kwargs):
        super().__init__(**kwargs)
        self.x = tf.constant(x, tf.float32)   # [Nx]
        self.y = tf.constant(y, tf.float32)   # [Ny]
        self.z = tf.constant(z, tf.float32)
        self.k = tf.constant(2*np.pi/lam, tf.complex64)
        self.Nx, self.Ny = len(x), len(y)
        self.M = N_meta
        M2 = N_meta * N_meta

        # --- load 4D directivity LUT ---
        data = np.load(lut_file)
        a_vals = np.squeeze(data['a_vals'])*1e9
        b_vals = np.squeeze(data['b_vals'])*1e9
        theta  = np.squeeze(data['theta'])
        phi    = np.squeeze(data['phi'])
        D4     = data['D_LUT'].astype(np.complex64)  # [A,B,Th,Ph]

        # store as TF constants
        self.a_vals     = tf.constant(a_vals, tf.float32)      # [A]
        self.b_vals     = tf.constant(b_vals, tf.float32)      # [B]
        self.theta_g    = tf.constant(theta,  tf.float32)      # [Th]
        self.phi_g      = tf.constant(phi,    tf.float32)      # [Ph]
        self.D_LUT      = tf.constant(D4,      tf.complex64)   # [A,B,Th,Ph]

        # build atom→flat index and atom positions
        py = (self.Ny - N_meta)//2
        px = (self.Nx - N_meta)//2
        m_idxs, n_idxs = np.meshgrid(np.arange(N_meta),
                                     np.arange(N_meta),
                                     indexing='ij')
        rows = (py + m_idxs).ravel()   # [M2]
        cols = (px + n_idxs).ravel()
        self.atom_idx = tf.constant(rows*self.Nx + cols, tf.int32)  # flat index
        self.xs = tf.constant(x[cols], tf.float32)  # [M2]
        self.ys = tf.constant(y[rows], tf.float32)

        # full (x,y) grid for outputs
        Yg, Xg = np.meshgrid(y, x, indexing='ij')
        self.Xg = tf.constant(Xg, tf.float32)  # [Ny,Nx]
        self.Yg = tf.constant(Yg, tf.float32)

    def _nearest_ab(self, flat_vals, grid):
        """
        Nearest‐neighbor quantization of flat_vals [M2] into grid [G].
        """
        # compute absolute differences [M2,G]
        diffs = tf.abs(flat_vals[:,None] - grid[None,:])
        return tf.argmin(diffs, axis=1)  # [M2]

    def _bilinear2d(self, D2, theta, phi):
        """
        Bilinear interp of D2[Th,Ph] at coords theta [Ny,Nx], phi [Ny,Nx].
        Returns [Ny,Nx] complex.
        """
        # flatten
        th = tf.reshape(theta,[-1])
        ph = tf.reshape(phi,  [-1])
        # 1D for θ
        i1 = tf.searchsorted(self.theta_g, th, 'left')
        i1 = tf.clip_by_value(i1, 1, tf.shape(self.theta_g)[0]-1)
        i0 = i1 - 1
        t0 = tf.gather(self.theta_g, i0)
        t1 = tf.gather(self.theta_g, i1)
        w1t = (th - t0)/(t1 - t0);  w0t = 1 - w1t
        # 1D for φ
        j1 = tf.searchsorted(self.phi_g, ph, 'left')
        j1 = tf.clip_by_value(j1, 1, tf.shape(self.phi_g)[0]-1)
        j0 = j1 - 1
        p0 = tf.gather(self.phi_g, j0)
        p1 = tf.gather(self.phi_g, j1)
        w1p = (ph - p0)/(p1 - p0);  w0p = 1 - w1p

        # gather four corners
        def g(ii, jj):
            idx = tf.stack([ii, jj], axis=1)   # [Ny*Nx,2]
            vals = tf.gather_nd(D2, idx)       # [Ny*Nx]
            return vals

        D00 = g(i0, j0);  D01 = g(i0, j1)
        D10 = g(i1, j0);  D11 = g(i1, j1)

        # blend
        # compute float weights then cast
        w00 = tf.cast(w0t * w0p, tf.complex64)
        w01 = tf.cast(w0t * w1p, tf.complex64)
        w10 = tf.cast(w1t * w0p, tf.complex64)
        w11 = tf.cast(w1t * w1p, tf.complex64)

        Df = D00*w00 + D01*w01 + D10*w10 + D11*w11
        # reshape back
        return tf.reshape(Df, tf.shape(self.Xg))  # [Ny,Nx]

    def call(self, U_meta, a_params, b_params):
        B = tf.shape(U_meta)[0]
        M2 = self.M * self.M

        # flatten meta-output per batch
        Uflat = tf.reshape(U_meta, [B, -1])  # [B,Ny*Nx]
        # gather each atom’s field amplitude
        Uatoms = tf.gather(Uflat, self.atom_idx, axis=1)  # [B,M2]

        # quantize (a,b) once
        af = tf.reshape(a_params, [-1])  # [M2]
        bf = tf.reshape(b_params, [-1])
        ia = self._nearest_ab(af, self.a_vals)  # [M2]
        ib = self._nearest_ab(bf, self.b_vals)

        def propagate_one(u_atoms):
            # u_atoms: [M2] complex
            Uout = tf.zeros_like(self.Xg, tf.complex64)  # [Ny,Nx]
            for k in tf.range(M2):
                amp = u_atoms[k]
                # skip padding zeros
                if tf.abs(amp) < 1e-8:
                    continue

                # spherical wave
                dx = self.Xg - self.xs[k]
                dy = self.Yg - self.ys[k]
                r  = tf.sqrt(self.z*self.z + dx*dx + dy*dy)
                sph = tf.exp(1j * self.k * tf.cast(r,tf.complex64)) / tf.cast(r,tf.complex64)

                # angles
                th = tf.acos(self.z / r)
                ph = tf.atan2(dy, dx)

                # directivity slice
                D2 = self.D_LUT[ia[k], ib[k]]  # [Th,Ph]
                Df = self._bilinear2d(D2, th, ph)

                Uout += amp * sph * tf.cast(Df, tf.complex64)

            return Uout

        # map over batch
        return tf.map_fn(propagate_one, Uatoms)


class ToDetector(tf.keras.layers.Layer):
    """
    This class represents the process of detection.
    The input field hits the detector array of N-detectors = N_classes and we measure the intensity.

    Arguments:
        detector_template (float[N_classes, N, N]):
            A series of masks, each representing one class. It should contain only zeros and ones, the patches of
            ones represent the area over which we integrate the detected intensity.
            One could also understand this as an array of single detectors placed at the locations defined by the
            detector template at the plane of interest. N must match the computational domain!
        **kwargs:
            grants access to variables of the keras base layer class

    Methods:
         call: computes the measured intensity for each class
    """
    def __init__(self, detector, norm = True, **kwargs):

        super(ToDetector,self).__init__(**kwargs)
        self.norm = norm
        self.Ny = np.shape(detector)[1]
        self.Nx = np.shape(detector)[2]
        self.detector  = tf.transpose(tf.reshape(detector, shape = (-1, self.Ny*self.Nx)))
        
    def get_config(self):
        config = super().get_config()
        return config
    
    def call(self, inputs):
        """
        Computes the measured intensity for each class

        Args:
            inputs (complex[batch_size, N, N]):  complex Amplitude of input field

        Returns:
            float[batch_size, N, N]: intensity at each pixel of each detector or intensity at detector below mask as defined above
        """
        readout  = Flatten()(tf.abs(inputs)**2)
        readout  = tf.matmul(tf.cast(readout, dtype = self.detector.dtype), self.detector)
        #readout = tf.matmul(tf.abs(inputs),tf.transpose(self.d)) 
        if self.norm:
            norm     = tf.math.reduce_max(readout, axis = -1, keepdims = True)
            return    readout/norm
        else:
            return readout
    
class ToDetector_mse(tf.keras.layers.Layer):
    """
    This class represents the process of detection.
    The input field hits the detector array of N-detectors = N_classes and we measure the intensity.

    Arguments:
        detector_template (float[N_classes, N, N]):
            A series of masks, each representing one class. It should contain only zeros and ones, the patches of
            ones represent the area over which we integrate the detected intensity.
            One could also understand this as an array of single detectors placed at the locations defined by the
            detector template at the plane of interest. N must match the computational domain!
        **kwargs:
            grants access to variables of the keras base layer class

    Methods:
         call: computes the measured intensity for each class
    """
    def __init__(self, detector, **kwargs):

        super(ToDetector_mse,self).__init__(**kwargs)

        self.Ny = np.shape(detector)[1]
        self.Nx = np.shape(detector)[2]
        self.detector  = tf.transpose(tf.reshape(detector, shape = (-1, self.Ny*self.Nx)))
        self.total_input_intensity = None
        
        
    def get_config(self):
        config = super().get_config()
        return config
    
    def call(self, inputs):
        """
        Computes the measured intensity for each class

        Args:
            inputs (complex[batch_size, N, N]):  complex Amplitude of input field

        Returns:
            (float[batch_size, N, N], float[batch_size,]): intensity at each pixel of each detector and total intensity on the camera
        """
                # Compute intensity
        intensity = tf.square(tf.abs(inputs))  # Shape: (batch_size, N, N)
        intensity_flat = tf.reshape(intensity, [tf.shape(inputs)[0], -1])  # Shape: (batch_size, N*N)

        # Compute detected intensities
        readout = tf.matmul(intensity_flat, self.detector)  # Shape: (batch_size, N_classes)

        # Total input intensity (sum over the entire computational domain)
        total_input_intensity = tf.reduce_sum(intensity_flat, axis=1, keepdims=True)  # Shape: (batch_size, 1)
        self.total_input_intensity = total_input_intensity
        norm     = tf.math.reduce_max(readout, axis = -1, keepdims = True)
        
        return readout/norm
    
    

class ToDetector_reg(tf.keras.layers.Layer):
    """
    This class represents the process of detection.
    The input field hits the detector array of N-detectors = N_classes and we measure the intensity.

    Arguments:
        detector_template (float[N_classes, N, N]):
            A series of masks, each representing one class. It should contain only zeros and ones, the patches of
            ones represent the area over which we integrate the detected intensity.
            One could also understand this as an array of single detectors placed at the locations defined by the
            detector template at the plane of interest. N must match the computational domain!
        **kwargs:
            grants access to variables of the keras base layer class

    Methods:
         call: computes the measured intensity for each class
    """
    def __init__(self, detector,reg= 0.001, **kwargs):

        super(ToDetector_reg,self).__init__(**kwargs)

        self.Ny = np.shape(detector)[1]
        self.Nx = np.shape(detector)[2]
        self.detector = tf.transpose(tf.reshape(detector, shape=(-1, self.Ny * self.Nx)))
        self.reg = reg
        
    def get_config(self):
        config = super().get_config()
        return config
    
    def call(self, inputs):
        # Compute the intensity of the input field
        intensity = tf.square(tf.abs(inputs))  # Shape: (batch_size, N, N)
        intensity_flat = tf.reshape(intensity, [tf.shape(inputs)[0], -1])  # Shape: (batch_size, N*N)

        # Compute the intensity detected by each class's detector
        readout = tf.matmul(tf.cast(intensity_flat, dtype = self.detector.dtype), self.detector)  # Shape: (batch_size, N_classes)

        total_input_intensity = tf.reduce_sum(intensity, axis=[1, 2])  # Shape: (batch_size,)
        # Compute total detected intensity
        total_detected_intensity = tf.reduce_sum(readout, axis=1)  # Shape: (batch_size,)

        # Compute intensity outside detectors
        intensity_outside = (total_input_intensity - total_detected_intensity)/total_input_intensity

        # Regularization loss
        
        regularization_loss = self.reg* tf.square(intensity_outside)

        # Add the regularization loss to the model
        self.add_loss(tf.reduce_mean(regularization_loss))
        norm     = tf.math.reduce_max(readout, axis = -1, keepdims = True)

        # Return logits
        return readout /norm
    

class ToDetector_conc(tf.keras.layers.Layer):
    """
    This class represents the process of detection.
    The input field hits the detector array of N-detectors = N_classes and we measure the intensity.

    Arguments:
        detector_template (float[N_classes, N, N]):
            A series of masks, each representing one class. It should contain only zeros and ones, the patches of
            ones represent the area over which we integrate the detected intensity.
            One could also understand this as an array of single detectors placed at the locations defined by the
            detector template at the plane of interest. N must match the computational domain!
        **kwargs:
            grants access to variables of the keras base layer class

    Methods:
         call: computes the measured intensity for each class
    """
    def __init__(self, detector, reg=0.001, reg_conc=1.0, **kwargs):
        super(ToDetector_conc, self).__init__(**kwargs)
        self.Ny = np.shape(detector)[1]
        self.Nx = np.shape(detector)[2]
        self.detector = tf.transpose(tf.reshape(detector, shape=(-1, self.Ny * self.Nx)))
        self.reg = reg  # For outside intensity regularization
        self.reg_conc = reg_conc  # For concentration loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'reg': self.reg,
            'reg_conc': self.reg_conc,
            # Include other parameters if necessary
        })
        return config

    def call(self, inputs, y_true=None):
        # Compute intensity
        intensity = tf.square(tf.abs(inputs))  # Shape: (batch_size, N, N)
        intensity_flat = tf.reshape(intensity, [tf.shape(inputs)[0], -1])  # Shape: (batch_size, N*N)

        # Compute detected intensities
        readout = tf.matmul(intensity_flat, self.detector)  # Shape: (batch_size, N_classes)

        # Total input intensity on the detector
        total_input_intensity = tf.reduce_sum(intensity_flat, axis=1) + 1e-8  # Shape: (batch_size,)

        # Total detected intensity
        total_detected_intensity = tf.reduce_sum(readout, axis=1)  # Shape: (batch_size,)

        # Intensity outside detectors
        intensity_outside = (total_input_intensity - total_detected_intensity) / total_input_intensity

        # Regularization loss for percent of the intensity outside detectors 
        regularization_loss = self.reg * tf.square(intensity_outside)

        # Add the regularization loss to the model
        self.add_loss(tf.reduce_mean(regularization_loss))

        norm     = tf.math.reduce_max(readout, axis = -1, keepdims = True)


        # If `y_true` is provided, compute concentration loss (maximize the intensity in the correct detector's sub-array)
        if y_true is not None:
            # Ensure y_true is of integer type
            y_true = tf.cast(y_true, tf.int32)

            # Create a mask for the correct classes
            num_classes = tf.shape(readout)[1]
            mask = tf.one_hot(y_true, depth=num_classes)  # Shape: [batch_size, num_classes]

            # Compute correct intensity
            correct_intensity = tf.reduce_sum(readout * mask, axis=1)  # Shape: [batch_size,]

            # Compute concentration ratio
            concentration_ratio = correct_intensity / total_input_intensity  # Shape: [batch_size,]

            # Compute concentration loss
            epsilon = 1e-8
            concentration_loss = -(concentration_ratio + epsilon)
            concentration_loss = self.reg_conc * concentration_loss  # Apply regularization factor

            # Add the concentration loss to the model
            self.add_loss(tf.reduce_mean(concentration_loss))

        # Return logits (raw readout)
        return readout 


class DetectorWrapper(tf.keras.layers.Layer):
    def __init__(self, detector_layer, **kwargs):
        super(DetectorWrapper, self).__init__(**kwargs)
        self.detector_layer = detector_layer

    def call(self, inputs):
        inputs, y_true = inputs
        return self.detector_layer(inputs, y_true=y_true)

class Activation(tf.keras.layers.Layer):
    """
    This class is a collection of activation functions which can be applied to the input plane.

    Methods:
        SHG(inputs)
            computes a non-linear activation based on non-depleted optical second harmonic generation.
            I.e the input field is squared at each point. Pay attention that after this
            layer the wavelength has to be set to be half of the fundamental mode.
    """

    @staticmethod
    def SHG(inputs):
        """
        computes the nonlinear effect of second harmonic generation on th complex input field

        Args:
            inputs (complex[batch_size, N, N]): complex input field

        Returns:
            complex[batch_size, N, N]: field after crystal

        """

        return tf.square(inputs)
    
    @staticmethod
    def phase_relu(inputs):
        """
        computes relu for phase and amplitude of a complex number

        Args:
            inputs (complex[batch_size, N, N]): complex input field

        Returns:
            complex[batch_size, N, N]: field after crystal

        """

        A = tf.cast(tf.abs(inputs), dtype = tf.float32)

        phi     = tf.cast(tf.math.angle(inputs), dtype = tf.float32)
        phi     =  tf.keras.activations.relu(phi)
        return tf.cast(tf.complex(A * tf.cos(phi), A * tf.sin(phi)) , dtype = inputs.dtype) 
                                                    
    @staticmethod
    def phase_tanh(inputs):
        """
        computes relu for phase and amplitude of a complex number

        Args:
            inputs (complex[batch_size, N, N]): complex input field

        Returns:
            complex[batch_size, N, N]: field after crystal

        """
        A       = tf.cast(tf.abs(inputs), dtype = tf.float32)
        phi     = tf.cast(tf.math.angle(inputs), dtype = tf.float32)
        phi     = tf.keras.activations.tanh(phi)
        return tf.cast(tf.complex(A * tf.cos(phi), A * tf.sin(phi)) , dtype = inputs.dtype) 
                          
    @staticmethod
    def phase_amplitude_tanh(inputs):
        """
        computes tanh for phase and amplitude of a complex number

        Args:
            inputs (complex[batch_size, N, N]): complex input field

        Returns:
            complex[batch_size, N, N]: field after crystal

        """
        A       = tf.cast(tf.abs(inputs), dtype = tf.float32)
        phi     = tf.cast(tf.math.angle(inputs), dtype = tf.float32)
        A       = tf.keras.activations.tanh(A)
        phi     = tf.keras.activations.tanh(phi)
        return tf.cast(tf.complex(A * tf.cos(phi), A * tf.sin(phi)) , dtype = inputs.dtype) 
    
    @staticmethod
    def amplitude_tanh(inputs):
        """
        computes tanh for phase and amplitude of a complex number

        Args:
            inputs (complex[batch_size, N, N]): complex input field

        Returns:
            complex[batch_size, N, N]: field after crystal

        """
        A   = tf.cast(tf.abs(inputs), dtype = tf.float32)
        phi  = tf.cast(tf.math.angle(inputs), dtype = tf.float32)
        A    = tf.keras.activations.tanh(A)
        return tf.cast(tf.complex(A * tf.cos(phi), A * tf.sin(phi)) , dtype = inputs.dtype) 

    @staticmethod
    def amplitude_relu(inputs):
        """
        computes tanh for phase and amplitude of a complex number

        Args:
            inputs (complex[batch_size, N, N]): complex input field

        Returns:
            complex[batch_size, N, N]: field after crystal

        """
        A     = tf.cast(tf.abs(inputs), dtype = tf.float32)
        phi   = tf.cast(tf.math.angle(inputs), dtype = tf.float32)
        A     = tf.keras.activations.relu(A)
        return tf.cast(tf.complex(A * tf.cos(phi), A * tf.sin(phi)) , dtype = inputs.dtype) 
                                                     
    @staticmethod
    def intensity_tanh(inputs):
        """
        computes tanh for the intensity as it probably would be done if handled electro opticalle

        Args:
            inputs (complex[batch_size, N, N]): complex input field

        Returns:
            complex[batch_size, N, N]: field after crystal

        """

        return tf.complex(tf.keras.activations.tanh(tf.abs(inputs)**2), tf.zeros_like(inputs, dtype = tf.math.real(inputs).dtype))
    
    @staticmethod
    def intensity_relu(inputs):
        """
        computes tanh for the intensity as it probably would be done if handled electro opticalle

        Args:
            inputs (complex[batch_size, N, N]): complex input field

        Returns:
            complex[batch_size, N, N]: field after crystal

        """

        return tf.complex(tf.keras.activations.relu(tf.abs(inputs)**2), tf.zeros_like(inputs, dtype = tf.math.real(inputs).dtype))
    
##########################
class embedd2CP(tf.keras.layers.Layer):
    def __init__(self, Dx, Dy, x, y, **kwargs):

        super(embedd2CP,self).__init__(**kwargs)
        if Dx > (x[-1]-x[0]) or  Dy >( y[-1]-y[0]):
            sys.exit("Input dimensions exceed the computational domain")

        # compute resized size
        newSizeY = len(y[np.round(np.abs(y),8)<=np.round(Dy/2,8)]) 
        newSizeX = len(x[np.round(np.abs(x),8)<=np.round(Dx/2,8)])
        Nx, Ny   = (np.shape(x)[0], np.shape(y)[0])

        # takes care of centering
        if not((Ny-newSizeY)%2==0) or not((Nx-newSizeX)%2==0):
            newSizeY = newSizeY-1
            newSizeX = newSizeX-1  

        #padding    
        padx     = (Nx-newSizeX)//2
        pady     = (Ny-newSizeY)//2
        self.padding  = [[0,0],[pady,pady],[padx,padx]]

        #resizing
        self.resize = Resizing(Ny, Nx)
        
    def get_config(self):
        config = super().get_config()
        return config
    
    def call(self, inputs):
        data     = tf.transpose(self.resize(tf.transpose(inputs)))
        return tf.squeeze(tf.pad(data, self.padding))
    
##########################
# For preprocessing
from skimage.transform import resize
import pandas as pd
import os

# Function to read .ubyte files
def read_ubyte(file_path):
    with open(file_path, 'rb') as f:
        # Read the magic number and dimensions
        magic_number = struct.unpack('>I', f.read(4))[0]
        
        if magic_number == 2051:  # Magic number for image files
            num_images = struct.unpack('>I', f.read(4))[0]
            rows = struct.unpack('>I', f.read(4))[0]
            cols = struct.unpack('>I', f.read(4))[0]
            
            # Read the image data
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
            return images

        elif magic_number == 2049:  # Magic number for label files
            num_labels = struct.unpack('>I', f.read(4))[0]
            
            # Read the label data
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels
        else:
            raise ValueError("Invalid magic number. This is not a valid .ubyte file.")

# Preprocessing
def loadData(type = "digits"):
    '''
    Dx, Dy [float]: lateral size of input, beaware to set this smaller than the computational domain
    x, y array[float]: meshing (computational domain)
    '''

    #Load Trainings and validation data
    if type == "digits":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif type == "fashion":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif type == "cifar":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif type == "emnist": #28x28!
        base_path = os.path.dirname(os.path.abspath(__file__))
        x_train = read_ubyte(os.path.join(base_path,"datasets", "emnist","emnist-balanced-train-images-idx3-ubyte"))
        x_test = read_ubyte(os.path.join(base_path,"datasets","emnist","emnist-balanced-test-images-idx3-ubyte"))
        y_train = read_ubyte(os.path.join(base_path,"datasets","emnist","emnist-balanced-train-labels-idx1-ubyte"))
        y_test = read_ubyte(os.path.join(base_path,"datasets","emnist","emnist-balanced-test-labels-idx1-ubyte"))

    else:
        print("invalid dataset choice")
    
    # Normalize 
    x_train, x_test = (x_train/255., x_test/255.)
    # Embedd
    # DONT FECKIN DO THAT IT TAKES TO MUCH SPACE IN MEMORY TO RESIZE THE WHOLE DATASET
    #x_train   = embedd_into_computational_domain(Dx, Dy, x ,y, x_train)
    #x_test    = embedd_into_computational_domain(Dx, Dy, x ,y, x_test)

    return x_train, y_train, x_test, y_test
def loadDetector(Dx, Dy, x ,y, file, mock=False, penalty=10):
    '''
    Dx, Dy [float]: lateral size of input, beaware to set this smaller than the computational domain
    x, y array[float]: meshing (computational domain)
    file = "square" for square detector areas, "radial" for radial one (10 classes each)
    '''
    # Load Datector
    data = np.load(file, allow_pickle = True)
    # Resize
    '''if mock:
        mock     = 1-np.expand_dims(np.sum(detector, axis = 0), axis = 0)
        print(np.shape(mock))
        detector = np.vstack([detector,mock])
        print(np.shape(detector))
        detector = np.array([detector[i]/np.sum(detector[i]) for i in range(len(detector))])
        detector[-1] = detector[-1] *penalty
'''
    detector = embedd_into_computational_domain(Dx, Dy, x ,y, data)

    if mock:
        N=np.shape(detector)[1]
        mock_mask=np.ones((N, N))
        sum=np.sum(detector, axis=0)
        max_width=np.max(np.argwhere(sum==1))-np.min(np.argwhere(sum==1))
        aperture_width=round(max_width+20)
        if aperture_width%2!=0:
            aperture_width+=1
        mock_mask[N//2-aperture_width//2:N//2+aperture_width//2, N//2-aperture_width//2:N//2+aperture_width//2 ]=0
        mock_class=np.abs(sum+mock_mask-1)

        detector=np.concatenate([detector, mock_class[np.newaxis, :, :]], axis=0)

        mock_sum = np.sum(detector[-1])
        patch_sum = np.sum(detector[0])
        if mock_sum > 0:
            detector[-1] *= (patch_sum / mock_sum) * penalty
        else:
            detector[-1] = 0  

    return detector

#Newest version, works like a charm!
def embedd_into_computational_domain(Dx, Dy, x ,y, data):

    # start with embedding relative to size
    size_x = x[-1]-x[0] # computational domain in x direction
    size_y = y[-1]-y[0] # computational domain in y direction
    Nx, Ny = (len(x), len(y))
    # embedding element into computational domain
    if Dx > size_x or  Dy >size_y:
        sys.exit("Input dimensions exceed the computational domain")
    elif Dx < size_x or  Dy < size_y:
        magx, magy     = (size_x/Dx, size_y/Dy)
        Nx_new, Ny_new = int(np.round((Nx/magx))), int(np.round((Ny/magy)))
        # When we pad we want the data to be embedded symeetrically thus we need to make sure this will be the case after shrinking
        if not((Nx-Nx_new)%2==0):
            print("to ensure symmetric embedding the metasurface size is adjusted")
            Nx_new +=-1
        if not((Ny-Ny_new)%2==0):
            print("to ensure symmetric embedding the metasurface size is adjusted")
            Ny_new  +=-1
        # Shrinking D to fit mesh
        data = np.transpose(Resizing(Ny_new, Nx_new, interpolation = "nearest")(np.transpose(data)))
        # Padding
        padx, pady = ((Nx-Nx_new)//2, (Ny-Ny_new)//2)
        padding    = [[0,0],[int(pady),int(pady)],[int(padx),int(padx)]]
        data       = np.pad(data, padding)
        # adapt size to computational domain
    elif (len(data[1]) != Nx or  len(data[0]) != Ny) and Dx == size_x and Dy == size_y:
         data = np.transpose(Resizing(Ny, Nx, interpolation = "nearest")(np.transpose(data)))
    return np.squeeze(data)

def myResize(Nx, Ny, input):
    return tf.transpose(Resizing(Ny, Nx)(tf.transpose(input)))

def makeDetector(angles, size, r_patches, N=28, mock = False, mock_type='single', penalty = 1):
    """
    Creates a detector mask with multiple square patches at specified angles around the center.

    Parameters:
    - angles (list of float): Angles in degrees for patch placement (from center).
    - size (int): Size of each square patch (in pixels).
    - r_patches (float): Radial distance (as a fraction of image size) for placing patches.
    - N (int): Size of the image (NxN). Default is 344.
    - mock (bool): If True, add a mock class that is the negative of all patches, normalized.

    Returns:
    - detector (np.ndarray): Array of shape (n_patches + mock, N, N) with 1s in patch regions.
    """

    # Number of classes: patches + mock class if enabled
    num_classes = len(angles)
    if mock:
        if mock_type == "single":
            num_classes += 1
        elif mock_type == "multiple":
            num_classes += len(angles)
    detector = np.zeros((num_classes, N, N))
    half_size = int(np.round(size / 2))
    center_x, center_y = N // 2, N // 2

    for i, alpha in enumerate(angles):
        # Convert angle to radians
        radians = np.deg2rad(alpha)
        # Compute patch location relative to center
        loc_x = int(np.round(center_x + np.sin(radians) * r_patches / 2 * N))
        loc_y = int(np.round(center_y + np.cos(radians) * r_patches / 2 * N))
        
        # Ensure patch fits within bounds
        x_start, x_end = max(0, loc_x - half_size), min(N, loc_x + half_size)
        y_start, y_end = max(0, loc_y - half_size), min(N, loc_y + half_size)

        detector[i, x_start:x_end, y_start:y_end] = 1

        
    mock_offset = len(angles)

    # Add mock class if enabled
    if mock and mock_type=="single":
        # Create the mock class as the negative of the other patches
        detector[-1] = 1 - np.sum(detector[:-1], axis=0)

        # Create central hole in mock class
        aperture = np.ones((N, N))

        # Define center and outer radii for the donut-shaped hole
        r_center = int(r_patches / 2 * N - 2*size)  # Inner radius (center hole)
        r_center_außen = int(r_patches / 2 * N + 2*size)  # Outer radius (boundary)

        # Iterate over each pixel to apply the donut-shaped aperture
        for i in range(N):
            for j in range(N):
                # Compute the radial distance from the center
                dist = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                
                # If the distance is less than the inner radius, set the value to zero (center hole)
                if dist < r_center:
                    aperture[i, j] = 1  # inner hole
                elif dist > r_center_außen:
                    aperture[i, j] = 0  # outside of donut
                else:
                    aperture[i, j] = 1  # donut area
        detector[-1] = detector[-1] * aperture[np.newaxis, :, :]

    if mock and mock_type=='multiple':
        half_size_outer=int(np.round(half_size*2.2))
        print(half_size_outer)
        half_size_inner=half_size
        np.shape(detector)
        for i, alpha in enumerate(angles):
            radians = np.deg2rad(alpha)
            detector[i+mock_offset, :, :]=0
            # Compute patch location relative to center
            loc_x = int(np.round(center_x + np.sin(radians) * r_patches / 2 * N))
            loc_y = int(np.round(center_y + np.cos(radians) * r_patches / 2 * N))
            
            x_start_outer, x_end_outer = max(0, loc_x - half_size_outer), min(N, loc_x + half_size_outer)
            y_start_outer, y_end_outer = max(0, loc_y - half_size_outer), min(N, loc_y + half_size_outer)
            
            detector[i+mock_offset, x_start_outer:x_end_outer, y_start_outer:y_end_outer] = 1

            x_start_inner, x_end_inner = max(0, loc_x - half_size_inner), min(N, loc_x + half_size_inner)
            y_start_inner, y_end_inner = max(0, loc_y - half_size_inner), min(N, loc_y + half_size_inner)
            
            detector[i+mock_offset, x_start_inner:x_end_inner, y_start_inner:y_end_inner] = 0



    #Normalization so that each patch sums up to 1 : whether it is a mock class or a detector class
    
    patch_sum = np.sum(detector[0])  # Use the first real detector as reference
    for i in range(mock_offset, detector.shape[0]):
        mock_sum = np.sum(detector[i])
        if mock_sum > 0:
            detector[i] *= (patch_sum / mock_sum) * penalty
        else:
            detector[i] = 0  # Avoid division by zero
    
    return detector
    

# first pad, then resize
# something off
def embedd_into_computational_domain_new01(Dx, Dy, x ,y, data, order = 1):

    # start with embedding relative to size
    size_x = x[-1]-x[0] # computational domain in x direction
    size_y = y[-1]-y[0] # computational domain in y direction

    # embedding element into computational domain
    if Dx > size_x or  Dy >size_y:
        sys.exit("Input dimensions exceed the computational domain")
    elif Dx < size_x or  Dy < size_y:
        magx, magy   = (size_x/Dx, size_y/Dy)
        padx, pady   = (np.round((magx*len(x)-len(x))/2), np.round((magy*len(y)-len(y))/2))
        padding  = [[0,0],[int(pady),int(pady)],[int(padx),int(padx)]]
        data     = np.pad(data, padding)
    # resizing this to the right number of points to match x
    data     = np.transpose(Resizing(len(y), len(x))(np.transpose(data)))

    return np.squeeze(data)


@tf.keras.utils.register_keras_serializable()

class LossFunctions(tf.keras.losses.Loss):
    """
    A collection of custom loss functions tailored for optical neural network training.

    Attributes:
        detector (tf.Tensor): The detector patterns/tensors corresponding to classes,
            flattened internally for indexing by label.
        T (float): Temperature scaling parameter used in some losses to sharpen
            output distributions (default: 10.0).
        eta (float): A scaling factor used in the PE loss (to set it properly: find the convergence
        value for when only PELoss with eta=1 is there, the exponent of the negative
        value of this is the eta one should set).
        loss_name (str): The name of the loss function to use. Must be one of:
            'PixelwiseMSELoss', 'CustomSCELoss', 'PELoss', 'JointLoss', 'JointLossPlus'.

    Methods:

        PixelwiseMSELoss(output_label, output_layer):
            Computes mean squared error between normalized output intensity and
            corresponding detector patterns.

        CustomSCELoss(output_label, output_layer):
            Computes sparse categorical cross-entropy loss after scaling and
            normalizing output intensities
        This one is the same as the default tensorflow one but deals with the 
        output layer which is not yet masked
        so when the toDetector is NOT the last layer in the model)

        PELoss(output_label, output_layer):
            Calculates a power efficiency (PE)-based loss encouraging output
            intensity overlap with the target detector pattern.

        JointLoss(output_label, output_layer):
            A weighted sum of CustomSCELoss and PELoss, combining classification
            and power efficiency objectives.
        based on the article Joint loss function design 
        in diffractive optical neural network classifiers 
        for high power efficiency  https://doi.org/10.1364/OE.547572

        JointLossPluss(output_label, output_layer):
            A weighted sum of CustomSCELoss, PELoss (so the JointLoss) which combining classification
            and power efficiency objectives and signal contrast - the ratio of the 
            intensity within a detector patch to the sum of the intensity in other detector paches


        available():
            Returns a list of available loss function names.

        get_config():
            Returns the configuration of the loss function instance
            (for serialization support, currently defaults to superclass).
    """



    def __init__(self, detector, loss_name, T=10., eta=0.746, factor=2.5):
        super().__init__()
        self.factor=factor
        self.bigger_detector = self.get_bigger_detector(detector[0:10])
        self.bigger_detector=Flatten()(self.bigger_detector)
        self.detector=Flatten()(detector)
        self.T=T
        self.eta=eta
        self.loss_name=loss_name
        self._loss_map={
            'PixelwiseMSELoss': self.PixelwiseMSELoss,
            'CustomSCELoss': self.CustomSCELoss,
            'PELoss': self.PELoss,
            'JointLoss':self.JointLoss,
            'JointLossPlus': self. JointLossPlus,
            'JointLossPlusPlus': self. JointLossPlusPlus
        }
    def call(self, output_label, output_layer, **kwargs):
        loss_fn = self._loss_map.get(self.loss_name)
        if loss_fn is None:
            raise ValueError(f"Loss '{self.loss_name}' not found")
        return loss_fn(output_label, output_layer)
    
    def get_bigger_detector(self, detector):
        """
        Given a detector as a NumPy array of shape (num_classes, N, N), 
        return a new array
        where each patch is centered in the same place, but the '1' region is
        twice as large in both dimensions.
        """
        num_classes, N, _=detector.shape
        bigger_detector=np.zeros_like(detector)
        for i in range(num_classes):
            rows, cols = np.where(detector[i] == 1)
            row_min, row_max = rows.min(), rows.max()
            col_min, col_max = cols.min(), cols.max()

            center_row = int(round((row_min + row_max) / 2))
            center_col = int(round((col_min + col_max) / 2))

            size = max(row_max - row_min + 1, col_max - col_min + 1)
            new_size = int(size * self.factor)
            half = new_size // 2

            r_start = max(center_row - half, 0)
            r_end = min(center_row + half, N)
            c_start = max(center_col - half, 0)
            c_end = min(center_col + half, N)
            

            bigger_detector[i, r_start:r_end, c_start:c_end] = 1.

        return bigger_detector
    
    def PixelwiseMSELoss(self, output_label, output_layer):
        readout  = Flatten()(tf.abs(output_layer)**2)           
        detector = self.detector                                    # Flatten detector (nClasses, NxN)
        norm     = tf.reduce_max(readout, axis=-1, keepdims=True)         # Normalize each output to maximally 1
        readout  = readout / norm                                         # Prevent divide-by-zero
        readout  = tf.cast(readout, dtype=np.float64)     
        detector = tf.cast(detector , dtype=np.float64)  
        output_label=tf.cast(output_label, tf.int32)              # Cast to same type (float) as detector, tensorflow is REALLY whiny about datatypes                                        
        selected_detectors = tf.gather(detector, output_label) 
            # Select the targetimage by label, selected detectors is now of shape (Batch, NxN)
        loss     = tf.reduce_mean(tf.square(readout - selected_detectors), axis = -1) # Computes MSE-difference (Batch)
        return tf.reduce_mean(loss) 
        
    def CustomSCELoss(self, output_label, output_layer):
        readout = Flatten()(tf.abs(output_layer)**2)
        toDetector=tf.transpose(self.detector)
        readout  = tf.matmul(tf.cast(readout, dtype = self.detector.dtype), toDetector)
        #now readout contains intensity sum within each class patch
        norm = tf.reduce_max(readout, axis=-1, keepdims=True)
        readout = (readout *self.T) / norm
        #tf.print("First 10 values from each batch sample:\n", readout[0, :10])


        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
            output_label, readout)
   
    def PELoss(self, output_label, output_layer):
        readout = Flatten()(tf.abs(output_layer)**2)
        norm = tf.reduce_max(readout, axis=-1, keepdims=True)
        readout = readout / norm 

        readout = tf.cast(readout, dtype=tf.float32)
        detector = tf.cast(self.detector, dtype=tf.float32)
        output_label = tf.cast(output_label, tf.int32)
        selected_detectors = tf.gather(detector, output_label)
        collected_power = tf.reduce_sum(readout * selected_detectors, axis=-1)
        total_power = tf.reduce_sum(readout, axis=-1)
        PE = collected_power / total_power 

        PE = -1 * tf.math.log(PE / self.eta)
        PE = 0.5 * (PE + tf.abs(PE))  # if PE is negative, it is set to zero, if it is positive, it is kept as is

        return tf.reduce_mean(PE)
    
    def SCLoss(self, output_label, output_layer):
        readout=Flatten()(tf.abs(output_layer)**2)
        norm=tf.reduce_max(readout, axis=-1, keepdims=True)
        readout=readout/norm
        readout=tf.cast(readout, dtype=tf.float32)
        detector = tf.cast(self.detector, dtype=tf.float32)
        output_label = tf.cast(output_label, tf.int32)
        selected_detectors = tf.gather(detector, output_label)
        collected_power = tf.reduce_sum(readout * selected_detectors, axis=-1)
        all_detectors=tf.reduce_sum(detector, axis=0)
        total_detector_power=tf.reduce_sum(readout * all_detectors, 
                                                             axis=-1)
        signal_contrast=collected_power/total_detector_power #normalization constant here maybe?


        SC=-1*tf.math.log(signal_contrast)
        SC=0.5*(SC+tf.abs(SC))

        return tf.reduce_mean(SC)

    def JointLoss(self, output_label, output_layer):
        loss_sce = self.CustomSCELoss(output_label, output_layer)
        loss_pe = self.PELoss(output_label, output_layer)
        loss_sc=self.SCLoss(output_label, output_layer)
        if loss_pe!=0:
            ratio_pe=loss_pe/loss_sce
        else:
            ratio_pe=1.

        if loss_sc!=0:
            ratio_sc=loss_sc/loss_sce
        else:
            ratio_sc=1.
        joint_loss=loss_sce +loss_pe/tf.stop_gradient(ratio_pe)
        
        return joint_loss
    def LeakageLoss(self, output_label, output_layer):
        readout=Flatten()(tf.abs(output_layer)**2)
        norm=tf.reduce_max(readout, axis=-1, keepdims=True)
        readout = readout / norm
        readout = tf.cast(readout, dtype=tf.float32)
        detector = tf.cast(self.detector, dtype=tf.float32)  # (num_classes, H*W)
        bigger_detector = tf.cast(self.bigger_detector, dtype=tf.float32)
        
        output_label = tf.cast(output_label, tf.int32)
        selected_detectors = tf.gather(detector, output_label)              # (batch, H*W)
        selected_bigger_detectors = tf.gather(bigger_detector, output_label)
        
        collected_power = tf.reduce_sum(readout * selected_detectors, axis=-1)
        total_power = tf.reduce_sum(readout * selected_bigger_detectors, axis=-1)
        leakage = collected_power / total_power

        LL=-1*tf.math.log(leakage)
        LL=0.5*(LL+tf.abs(LL))
        return tf.reduce_mean(LL)

    def JointLossPlus(self, output_label, output_layer):
        loss_sce = self.CustomSCELoss(output_label, output_layer)
        loss_sc=self.SCLoss(output_label, output_layer)
        joint_loss=self.JointLoss(output_label, output_layer)
        loss_ll=self.LeakageLoss(output_label, output_layer)
        ratio_sc = loss_sc / loss_sce if loss_sc != 0 else 1.
        ratio_ll=loss_ll/loss_sce if loss_ll!=0 else 1.

        joint_loss_plus=joint_loss+loss_sc/tf.stop_gradient(ratio_sc) 
        return joint_loss_plus

    def JointLossPlusPlus(self, output_label, output_layer):
        loss_sce = self.CustomSCELoss(output_label, output_layer)
        joint_loss_plus=self.JointLossPlus(output_label, output_layer)
        loss_ll=self.LeakageLoss(output_label, output_layer)
        ratio_ll=loss_ll/loss_sce if loss_ll!=0 else 1.

        joint_loss_plusplus=joint_loss_plus+loss_ll/tf.stop_gradient(ratio_ll) 
        return joint_loss_plusplus
    
    def available(self):
        return list(self._loss_map.keys())
    

"""
    Metrics
"""
@tf.keras.utils.register_keras_serializable()
class PowerEfficiency(tf.keras.metrics.Metric):
    def __init__(self, detector, name='power_efficiency', **kwargs):
        super(PowerEfficiency, self).__init__(name=name, **kwargs)
        self.detector = Flatten()(detector)
        self.detector = tf.cast(self.detector, dtype=tf.float64)
        self.total = self.add_weight(name='total', initializer='zeros', dtype=tf.float64)
        self.count = self.add_weight(name='count', initializer='zeros', dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        output_label = tf.cast(y_true, tf.int32)
        output_layer = tf.abs(y_pred) ** 2
        readout = Flatten()(output_layer)

        norm = tf.reduce_max(readout, axis=-1, keepdims=True)
        readout = readout / norm
        readout = tf.cast(readout, dtype=tf.float64)

        selected_detectors = tf.gather(self.detector, output_label)
        collected_power = tf.reduce_sum(readout * selected_detectors, axis=-1)
        total_power = tf.reduce_sum(readout, axis=-1)
        PE = collected_power / total_power

        self.total.assign_add(tf.reduce_sum(PE))
        self.count.assign_add(tf.cast(tf.shape(PE)[0], tf.float64))

    def result(self):
        return self.total / self.count

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)
    def get_config(self):
        config = super().get_config()
        # Save detector as list for serialization
        config.update({
            'detector': self.detector.numpy().tolist()
        })
        return config

    @classmethod
    def from_config(cls, config):
        detector = config.pop('detector')
        return cls(detector=detector, **config)

@tf.keras.utils.register_keras_serializable()
class ClassContrast(tf.keras.metrics.Metric):
    def __init__(self, detector, name='class_contrast', **kwargs):
        super(ClassContrast, self).__init__(name=name, **kwargs)
        self.detector = Flatten()(detector)
        self.detector = tf.cast(self.detector, dtype=tf.float64)
        self.total = self.add_weight(name='total', initializer='zeros', dtype=tf.float64)
        self.count = self.add_weight(name='count', initializer='zeros', dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        output_label = tf.cast(y_true, tf.int32)
        output_layer = tf.abs(y_pred) ** 2
        readout = Flatten()(output_layer)

        norm = tf.reduce_max(readout, axis=-1, keepdims=True)
        readout = readout / norm
        readout = tf.cast(readout, dtype=tf.float64)

        selected_detectors = tf.gather(self.detector, output_label)
        collected_power = tf.reduce_sum(readout * selected_detectors, axis=-1)

        all_detectors=tf.reduce_sum(self.detector, axis=0)
        total_detector_power = tf.reduce_sum(readout * all_detectors, 
                                                             axis=-1)
        SC=collected_power/total_detector_power

        self.total.assign_add(tf.reduce_sum(SC))
        self.count.assign_add(tf.cast(tf.shape(SC)[0], tf.float64))

    def result(self):
        return self.total / self.count

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)
    def get_config(self):
        config = super().get_config()
        # Save detector as list for serialization
        config.update({
            'detector': self.detector.numpy().tolist()
        })
        return config

    @classmethod
    def from_config(cls, config):
        detector = config.pop('detector')
        return cls(detector=detector, **config)


"""
    for live plotting of power efficiency
"""

"""
from IPython.display import clear_output  # for Jupyter/live plotting

class LivePowerEfficiencyPlot(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.pe_values = []
        self.acc_values = []

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            if 'power_efficiency' in logs:
                self.pe_values.append(logs['power_efficiency'])
            if 'custom_sparse_categorical_accuracy' in logs:
                self.acc_values.append(logs['custom_sparse_categorical_accuracy'])

            clear_output(wait=True)
            fig, ax1 = plt.subplots(figsize=(8, 4))

            # Left y-axis: Power Efficiency
            ax1.plot(self.pe_values, 'b-', label='Power Efficiency')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Power Efficiency', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.grid(True)

            # Right y-axis: Accuracy
            ax2 = ax1.twinx()
            ax2.plot(self.acc_values, 'r--', label='Accuracy')
            ax2.set_ylabel('Accuracy', color='r')
            ax2.tick_params(axis='y', labelcolor='r')

            # Title and legends
            fig.tight_layout()
            fig.legend(loc='lower right')
            plt.show()
"""

"""
    custom SCE function for when MSE loss is used. 
    Basically when the last layer is opticalFFT and not toDetector
"""
@tf.keras.utils.register_keras_serializable()
class CustomSparseCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, detector, name='custom_sparse_categorical_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.detector = Flatten()(detector)# Flatten if needed
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Compute readout
        readout = tf.abs(y_pred) ** 2
        readout = tf.reshape(readout, [tf.shape(readout)[0], -1])
        readout = tf.cast(readout, dtype=self.detector.dtype)
        readout = tf.matmul(readout, tf.transpose(self.detector))

        # Normalize
        norm = tf.reduce_max(readout, axis=-1, keepdims=True)
        readout = readout / norm

        # Compute accuracy
        acc = tf.keras.metrics.sparse_categorical_accuracy(y_true, readout)

        self.total.assign_add(tf.reduce_sum(acc))
        self.count.assign_add(tf.cast(tf.size(acc), self.count.dtype))

    def result(self):
        return self.total / (self.count + 1e-8)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)
    def get_config(self):
        config = super().get_config()
        # Save detector as numpy array
        config.update({
            "detector": self.detector.numpy().tolist()
        })
        return config
    @classmethod
    def from_config(cls, config):
        detector = config.pop("detector")
        return cls(detector=detector, **config)
    
@tf.keras.utils.register_keras_serializable()
class LightConcentration(tf.keras.metrics.Metric):
    """
    Custom Keras metric to compute how much light is around the detector patch, namely light leakage.
    Light leakage is defined as the ratio of intensity that is around the original patch
      in the area twice as large as the original patch
    to the total intensity in the larger patch.

    """
    def __init__(self, detector, factor=2,name="light_concentration", **kwargs):
        super().__init__(name=name, **kwargs)
        self.factor=factor
        self.bigger_detector = self.get_bigger_detector(detector)
        self.bigger_detector=Flatten()(self.bigger_detector)
        self.bigger_detector = tf.cast(self.bigger_detector, dtype=tf.float64)

        self.detector = Flatten()(detector)
        self.detector = tf.cast(self.detector, dtype=tf.float64)
        # Compute the "bigger detector" (doubled patch size around same center)
        

        self.total = self.add_weight(name='total', initializer='zeros', dtype=tf.float64)
        self.count = self.add_weight(name='count', initializer='zeros', dtype=tf.float64)
    
    def get_bigger_detector(self, detector):
        """
        Given a detector as a NumPy array of shape (num_classes, N, N), 
        return a new array
        where each patch is centered in the same place, but the '1' region is
        twice as large in both dimensions.
        """
        num_classes, N, _=detector.shape
        bigger_detector=np.zeros_like(detector)
        for i in range(num_classes):
            rows, cols = np.where(detector[i] == 1)
            row_min, row_max = rows.min(), rows.max()
            col_min, col_max = cols.min(), cols.max()

            center_row = int(round((row_min + row_max) / 2))
            center_col = int(round((col_min + col_max) / 2))

            size = max(row_max - row_min + 1, col_max - col_min + 1)
            new_size = int(size * self.factor)
            half = new_size // 2

            r_start = max(center_row - half, 0)
            r_end = min(center_row + half, N)
            c_start = max(center_col - half, 0)
            c_end = min(center_col + half, N)
            

            bigger_detector[i, r_start:r_end, c_start:c_end] = 1.

        return bigger_detector

    def update_state(self, y_true, y_pred, sample_weight=None):
        output_label=tf.cast(y_true, tf.int32)
        output_layer=tf.abs(y_pred)**2
        readout=Flatten()(output_layer)

        norm=tf.reduce_max(readout, axis=-1, keepdims=True)
        readout=readout/norm
        readout=tf.cast(readout, dtype=tf.float64)

        selected_detectors=tf.gather(self.detector, output_label)
        collected_power=tf.reduce_sum(readout*selected_detectors, axis=-1)

        selected_bigger_detectors=tf.gather(self.bigger_detector, output_label)
        total_power=tf.reduce_sum(readout*selected_bigger_detectors, axis=-1)

        leakage=collected_power/total_power
        self.total.assign_add(tf.reduce_sum(leakage))
        self.count.assign_add(tf.cast(tf.shape(leakage)[0], tf.float64))

    def result(self):
        return self.total / (self.count)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = super().get_config()
        # Note: detector can't be saved as tensor directly, consider saving as list or handle separately
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)