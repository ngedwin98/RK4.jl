
### Here we compare the relative performance of odes that contain numeric parameters via closures vs. fixed constants vs. ode parameter pointer

# In[3]:

function make_my_ode_closure(kappa, gamma, chi, epsilon)
    function my_ode!(t, z, zdot, params=nothing)
        zdot[1] = -.5kappa * z[1] + chi * z[3] * conj(z[2])
        zdot[2] = -.5kappa * z[2] + chi * z[3] * conj(z[1])
        zdot[3] = -.5gamma * z[3] - chi * z[1] * z[2] + epsilon
    end
end

type my_odeparams
    kappa::Float64
    gamma::Float64
    chi::Float64
    epsilon::Float64
end

function eps_threshold(kappa, gamma, chi)
    kappa * gamma /(4chi)
end

function my_ode2!(t, z, zdot, params::my_odeparams)
    zdot[1] = -.5params.kappa * z[1] + params.chi * z[3] * conj(z[2])
    zdot[2] = -.5params.kappa * z[2] + params.chi * z[3] * conj(z[1])
    zdot[3] = -.5params.gamma * z[3] - params.chi * z[1] * z[2] + params.epsilon
end


# Out[3]:

#     my_ode2! (generic function with 1 method)

# In[4]:

kappa = 10.
gamma = 100.
chi = 1.

et = eps_threshold(kappa, gamma, chi)
eps = 1.3 * et


# Out[4]:

#     325.0

# In[5]:

my_ode1! = make_my_ode_closure(kappa, gamma, chi, eps)
my_ode2_params = my_odeparams(kappa, gamma, chi, eps)

function my_ode3!(t, z, zdot, params=nothing)
    zdot[1] = -.5 * 10. * z[1] + 1. * z[3] * conj(z[2])
    zdot[2] = -.5 * 10. * z[2] + 1. * z[3] * conj(z[1])
    zdot[3] = -.5 * 100. * z[3] - 1. * z[1] * z[2] + 325.
end


# Out[5]:

#     my_ode3! (generic function with 2 methods)

# In[39]:

tlist = linspace(0,10,2001)
z0 = zeros(Complex128, 3)
z0[1] += .1
hmax = .001
@time rk4solve(my_ode1!, z0, tlist, hmax, nothing)
@time rk4solve(my_ode2!, z0, tlist, hmax, my_ode2_params)
@time data_det = rk4solve(my_ode3!, z0, tlist, hmax, nothing)


# Out[39]:

#     elapsed time: 0.008763427 seconds (4003824 bytes allocated)
#     elapsed time: 0.036594139 seconds (4003888 bytes allocated)
#     elapsed time: 0.012820318 seconds (4003824 bytes allocated)
# 

#     3x2001 Array{Complex{Float64},2}:
#      0.1+0.0im    0.0975317+0.0im  …  8.66025+0.0im  8.66025+0.0im
#      0.0+0.0im  0.000365167+0.0im     8.66025+0.0im  8.66025+0.0im
#      0.0+0.0im      1.43779+0.0im         5.0+0.0im      5.0+0.0im

### Compare performance to Sundials solver

# In[40]:

using Sundials
function wrap_complex(ode)
    function real_ode(t, y, ydot)
        ode(t, reinterpret(Complex128, y), reinterpret(Complex128,ydot))
    end
end

@time Sundials.ode(wrap_complex(my_ode1!), reinterpret(Float64, z0), tlist)'
        


# Out[40]:

#     elapsed time: 0.048252799 seconds (1271600 bytes allocated)
# 

#     6x2001 Array{Float64,2}:
#      0.1  0.0975317    0.0951323   …  8.66031  8.66031  8.66031  8.66031
#      0.0  0.0          0.0            0.0      0.0      0.0      0.0    
#      0.0  0.000364961  0.00131923     8.66031  8.66031  8.66031  8.66031
#      0.0  0.0          0.0            0.0      0.0      0.0      0.0    
#      0.0  1.43789      2.55698        4.99998  4.99998  4.99998  4.99998
#      0.0  0.0          0.0         …  0.0      0.0      0.0      0.0    

### Solve a stochastic problem

# In[41]:

function make_my_sde_closure(kappa, gamma, chi, epsilon)
    function my_sde!(t, z, w, zdot, params=nothing)
        zdot[1] = -.5kappa * z[1] + chi * z[3] * conj(z[2]) - sqrt(kappa)/2. * (w[1] + 1im*w[2])
        zdot[2] = -.5kappa * z[2] + chi * z[3] * conj(z[1]) - sqrt(kappa)/2. * (w[3] + 1im*w[4])
        zdot[3] = -.5gamma * z[3] - chi * z[1] * z[2] + epsilon - sqrt(gamma)/2. * (w[5] + 1im*w[6])
    end
end


# Out[41]:

#     make_my_sde_closure (generic function with 1 method)

# In[53]:

tlist = linspace(0, 100, 20001)
my_sde! = make_my_sde_closure(kappa, gamma, chi, eps)
@time data_det = rk4solve(my_ode1!, z0, tlist, hmax, nothing)
@time data_stoch, w_stoch = rk4solve_stochastic(my_sde!, z0, tlist, hmax, 6, nothing)


# Out[53]:

#     elapsed time: 0.173285705 seconds (37678528 bytes allocated)
#     elapsed time: 0.246445816 seconds (73687840 bytes allocated)
# 

#     (
#     3x20001 Array{Complex{Float64},2}:
#      0.1+0.0im  0.0663593+0.0302668im  …   -4.52191+6.45538im
#      0.0+0.0im    0.165425-0.160761im      -4.03559-7.27252im
#      0.0+0.0im     1.17517+0.171727im     4.68364+0.0415094im,
#     
#     6x20000 Array{Float64,2}:
#        19.8866    7.93405  -38.1622  -72.5182  …   64.4042    40.5654   -66.3385 
#       -19.3125  -52.204     35.3003   19.9148     -71.8084   -52.8304    54.1936 
#      -106.423   -60.3538    12.1902   46.1044      26.9748   -25.9346   -85.6279 
#       102.263   105.18     -11.093   -37.3462       5.96316   33.6985    -8.34943
#        63.1335  -66.3701    68.2258   71.0337     165.941     75.6323   -11.9125 
#       -38.8264  -54.6555   105.712    60.3185  …   65.0092    -2.79253  -40.2567 )

# In[54]:

using PyPlot


# In[56]:

plot(tlist, real(data_stoch[3,:]'))
plot(tlist, real(data_det[3,:]'))


# Out[56]:

# image file:

#     1-element Array{Any,1}:
#      PyObject <matplotlib.lines.Line2D object at 0x1159cfad0>

# In[57]:

plot(tlist, real(data_stoch[1,:]'))
plot(tlist, real(data_det[1,:]'))


# Out[57]:

# image file:

#     1-element Array{Any,1}:
#      PyObject <matplotlib.lines.Line2D object at 0x11956ef10>

# In[58]:

plot(tlist, imag(data_stoch[1,:]'))
plot(tlist, imag(data_det[1,:]'))


# Out[58]:

# image file:

#     1-element Array{Any,1}:
#      PyObject <matplotlib.lines.Line2D object at 0x119ac9950>

# In[59]:

plot(real(data_stoch[1,:]'),imag(data_stoch[1,:]'))


# Out[59]:

# image file:

#     1-element Array{Any,1}:
#      PyObject <matplotlib.lines.Line2D object at 0x119af81d0>

# In[62]:

plot(tlist, real(data_stoch[1,:] .* data_stoch[2,:])')
plot(tlist, real(data_det[1,:] .* data_det[2,:])')


# Out[62]:

# image file:

#     1-element Array{Any,1}:
#      PyObject <matplotlib.lines.Line2D object at 0x119620e90>

# In[ ]:



