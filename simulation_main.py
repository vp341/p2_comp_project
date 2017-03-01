import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


#creates individual ring of particles with specified relative density
def create_ring(radius, relative_density):
	no_of_particles = radius*particle_density*relative_density
	velocity = np.sqrt(1/radius)
	angles = np.linspace(0,2*np.pi,no_of_particles)
	positions = radius*np.exp(1j*angles)
	velocities = velocity*1j*np.exp(1j*angles)
	coords = np.column_stack((np.real(positions),np.imag(positions),np.real(velocities),np.imag(velocities)))
	return [no_of_particles,coords]

#creates a set of rings given an array of radius,density pairs
def create_ring_set(radius_density_pairs):
	rings = np.zeros((1,4))
	critical_values = []
	tot = 0
	#only looping over 6 values, so no need for numpy loop
	for [radius,density] in radius_density_pairs:
		ring = create_ring(radius,density)
		rings = np.append(rings,ring[1],axis=0)
		tot += ring[0]
		critical_values.append(tot)
	rings = np.delete(rings,0,0)
	return [critical_values,rings]

#plots a ring set with individual colours for each ring
def plot_ring_set(critical_values, rings, mass_sol):
	plot = []
	plt.gca().set_aspect('equal', adjustable='box')
	rings = np.split(rings,critical_values)
	for i in range(0,int(len(mass_sol[0])/5)):
		plt.plot(mass_sol[:,0+5*i],mass_sol[:,1+5*i],c="g")
		plt.plot(np.nan,np.nan)

	#only looping over 6 values so no need for numpy loop
	for i,ring in enumerate(rings):
		if(len(ring)):
			plot.append(plt.plot(ring[:,0],ring[:,1],".",c=colours[i])[0])
	masses = np.reshape(mass_sol[0],(-1,5))
	np.shape
	plot.append(plt.plot(masses[:,0],masses[:,1],"o",c="k")[0])
	plt.show()	
	#returns plot to allow live plotting
	return plot

#plots a ring set with individual colours for each ring
def plot_ring_live(critical_values,rings,masses,graph):
	rings = np.split(rings,critical_values)
	# #only looping over 6 values so no need for numpy loop
	for i,ring in enumerate(rings):
		if(len(ring)):
			graph[i].set_xdata(ring[:,0])
			graph[i].set_ydata(ring[:,1])
			graph[i].set_mfc(colours[i])
	graph[len(graph)-1].set_xdata(masses[:,0])
	graph[len(graph)-1].set_ydata(masses[:,1])
	plt.draw()
	plt.pause(0.0001)

def plot_live_full(vals,ring_sol,mass_sol):
	this_rings = np.reshape(ring_sol[0],(-1,4))
	plt.gca().set_aspect('equal', adjustable='box')
	plt.ion()
	# plt.xlim([-40,40])
	# plt.ylim([-40,40])
	graph = plot_ring_set(vals,this_rings,mass_sol)
	plt.pause(0.3)
	for i in range(1,len(sol)):
		plot_ring_live(vals,np.reshape(ring_sol[i],(-1,4)),np.reshape(mass_sol[i],(-1,5)),graph)


def g(masses, rings):
	x = rings[:,0]
	y = rings[:,1]
	toReturn = np.zeros_like(rings)
	for mass in masses:
		delta_x = x-mass[0]
		delta_y = y-mass[1]
		r = np.sqrt(np.power(delta_x,2)+np.power(delta_y,2))
		r3 = np.power(r,-3)
		#r3 = np.clip(r3,0.0,1.0e4)
		toReturn[:,2] -= np.multiply(r3,delta_x)*mass[4]
		toReturn[:,3] -= np.multiply(r3,delta_y)*mass[4]
	toReturn[:,0] = rings[:,2]
	toReturn[:,1] = rings[:,3]
	toReturn = np.nan_to_num(toReturn)
	return toReturn

def g_mass(masses):
	x = masses[:,0]
	y = masses[:,1]
	toReturn = np.zeros_like(masses)
	for mass in masses:
		delta_x = x-mass[0]
		delta_y = y-mass[1]
		r = np.sqrt(np.power(delta_x,2)+np.power(delta_y,2))
		r3 = np.power(r,-3)
		#r3 = np.clip(r3,0.0,1.0e4)
		toReturn[:,2] -= np.multiply(r3,delta_x)*mass[4]
		toReturn[:,3] -= np.multiply(r3,delta_y)*mass[4]
	toReturn[:,0] = masses[:,2]
	toReturn[:,1] = masses[:,3]
	toReturn = np.nan_to_num(toReturn)
	return toReturn


#the differential step for the ODE solver
def ode_step(full_set,t,ring_no,thing):
	rings = full_set[:ring_no]
	masses = full_set[ring_no:]
	rings = np.reshape(rings,(-1,4))
	masses = np.reshape(masses,(-1,5))
	dringsdt = g(masses,rings).flatten()
	dmassesdt = g_mass(masses).flatten()
	toReturn = np.append(dringsdt,dmassesdt)
	return toReturn

colours = ["#FF0000","#cc0040","#990080","#6600BF","#3300FF","#FFFFFF","#000000"]
particle_density = 5
#ring_set = create_ring_set([[1,5],[2,5]])
ring_set = create_ring_set([[2,12],[3,18],[4,24],[5,30],[6,36]])
#masses = [[0.0,0.0,0.0,0.0,1.0],[0.0,10.0,0.447,0.0,2.0]]
masses = [[0.0,0.0,0.0,0.0,1.0],[0.0,20.0,0.31,0.0,2.0]]
#masses = [[0.0,0.0,0.0,0.0,1.0]]
masses = np.reshape(masses,(-1,5))
masses_f = masses.flatten()

rings = ring_set[1]
vals = ring_set[0]
#plot_ring_set(vals,rings)
thing = 0
rings_f = rings.flatten()
ring_no = len(rings_f)
full_set_f = np.append(rings_f,masses_f)
totalTime = 50
noOfSteps = 100
t = np.linspace(0,totalTime,noOfSteps)
sol = integrate.odeint(ode_step,full_set_f,t,args=(ring_no,thing))
ring_sol = sol[:,:ring_no]
mass_sol = sol[:,ring_no:]
plot_live_full(vals,ring_sol,mass_sol)

