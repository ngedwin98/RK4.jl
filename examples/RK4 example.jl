
# In[1]:

using RK4


# In[8]:

function my_ode!(t, x, xdot, params=nothing)
    for kk=2:length(x)
    # some crazy ass ODE
        xdot[kk] = -cos(sum(abs(x))) * x[kk-1]
    end
    xdot[1] = -cos(sum(abs(x))) * x[end]  + sin(t)
end


x0 = zeros(5)
tlist = linspace(0, 100., 1001)

xts = rk4solve(my_ode!, x0, tlist, .01)


# Out[8]:

#     5x1001 Array{Float64,2}:
#      0.0   0.00499583    0.0199334   …  0.471826  0.419368  0.374404
#      0.0  -0.000166582  -0.00133054     0.441665  0.449207  0.455343
#      0.0   4.16524e-6    6.65672e-5     1.71957   1.72711   1.73411 
#      0.0  -8.33041e-8   -2.66355e-6     1.29509   1.32427   1.35104 
#      0.0   1.38786e-9    8.8799e-8      0.607237  0.629415  0.650109

# In[9]:

using PyPlot
plot(tlist, xts')


# Out[9]:

# image file:

#     5-element Array{Any,1}:
#      PyObject <matplotlib.lines.Line2D object at 0x11065a490>
#      PyObject <matplotlib.lines.Line2D object at 0x11065a710>
#      PyObject <matplotlib.lines.Line2D object at 0x11065a950>
#      PyObject <matplotlib.lines.Line2D object at 0x11065ab10>
#      PyObject <matplotlib.lines.Line2D object at 0x11065acd0>
