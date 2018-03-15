def homomorphic_filter(im, cutoff, order=2, lowgain=0.5, highgain=2):
    """
    Applies a homomorphic filter to an image using a Butterworth filter,
    """
    from numpy import log, exp, meshgrid
    from numpy.fft import fft2, ifft2, fftshift, ifftshift
    im = im.astype(float)
    im[im==0] = 1 # prevent taking the log of 0
    lg = log(im)
    ft = fftshift(fft2(lg))
    h,w = im.shape
    y,x = meshgrid(range(-(w//2), (w+1)//2), range(-(h//2), (h+1)//2))
    bw_fltr = 1/(1+0.414*((x*x+y*y)/(cutoff*cutoff))**order)
    fltr = lowgain + (highgain - lowgain) * (1 - bw_fltr)
    fltred = fltr * ft
    return exp(ifft2(ifftshift(fltred)).real).clip(0, 255).astype('uint8')