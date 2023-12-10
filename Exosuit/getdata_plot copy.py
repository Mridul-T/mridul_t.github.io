import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import comp_filter as cf

# Create figure for plotting
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []

#Setting up MPU0650
gyro = 250      # 250, 500, 1000, 2000 [deg/s]
acc = 2         # 2, 4, 7, 16 [g]
tau = 0.95
mpu = cf.MPU(gyro, acc, tau)

    # Set up sensor and calibrate gyro with N points
mpu.setUp()
mpu.calibrateGyro(500)

# This function is called periodically from FuncAnimation
def animate(i, xs, ys):

    # Read temperature (Celsius) from TMP102
    
    
    f = open("data.txt", "a")
    x,y,z=mpu.compFilter()
    # Add x and y to lists
    xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
    ys.append(x)

    xs = xs[-100:]
    ys = ys[-100:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('R vs Time')
    plt.ylabel('R')
    f.close()

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=10)
plt.show()