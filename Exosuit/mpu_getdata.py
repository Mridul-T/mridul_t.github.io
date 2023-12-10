import comp_filter as cf
import time

def run():
    gyro = 250      # 250, 500, 1000, 2000 [deg/s]
    acc = 2         # 2, 4, 7, 16 [g]
    tau = 0.98
    mpu = cf.MPU(gyro, acc, tau)

    # Set up sensor and calibrate gyro with N points
    mpu.setUp()
    mpu.calibrateGyro(500)
    l=[]
    f = open("data.txt", "a")
    # Run for 20 secounds
    startTime = time.time()
    while(time.time() < (startTime + 20)):
        x,y,z=mpu.compFilter()
        f.write(str(time.time())+","+str(x)+"\n")
    # End
    f.close()
    print("Closing")
run()